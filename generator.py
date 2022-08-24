from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from misc.utils_sampling import top_k_top_p_filtering, ngram_copy_filtering
from misc.utils_gpu import MemoryLogger, get_device
from math import ceil, floor
import torch
import threading

"""
The code is from Keep it Simple https://github.com/tingofurro/keep_it_simple/
"""

def pad(data, padval=0):
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=padval)

class Generator:

    def __init__(self, model_card="benjamin/gerpt2", max_input_length=512, max_output_length=300, print_memory=False, device=None):
        self.device = get_device(gpu_idx=0)

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_card, max_length=max_input_length)
        self.tokenizer.pad_token = "!"
        
        self.model = GPT2LMHeadModel.from_pretrained(model_card, pad_token_id=self.tokenizer.eos_token_id, max_length=(max_input_length+max_output_length))
        self.model.to(self.device)

        self.tokenizer.add_special_tokens({'sep_token':'<|sep|>'})
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.start_id = self.tokenizer.bos_token_id
        self.end_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.model.train()

        self.mlog = MemoryLogger(printing=print_memory)

        self.parallel_outputs = []

    def get_tokenizer_vocabsize(self):
        if self.sep_id == self.tokenizer.vocab_size:
            return self.tokenizer.vocab_size + 1
        return self.tokenizer.vocab_size

    def save(self, to_file):
        torch.save(self.model.state_dict(), to_file)

    def reload(self, from_file, strict=True):
        loaded_dict = torch.load(from_file, map_location=torch.device(self.device))
        loaded_dict = {k.replace("module.module.", ""): v for k, v in loaded_dict.items()}
        print(self.model.load_state_dict(loaded_dict, strict=strict))

    def run(self, text, sample_size=1, top_k=1, top_p=0, temperature=1.1):
        output = self.generate_torch_samples(text, parallel=False, max_output_len=self.max_output_length, sample_size=sample_size, top_k=top_k, top_p=top_p, temperature=temperature)
        return [o['output_text'] for o in output]

    def run_typical(self, text, sample_size, p=0.95, temperature=1):
        output = self.typical_sampling(text, max_output_len=self.max_output_length, sample_size=sample_size, p=p, temperature=temperature)
        return [o['output_text'] for o in output]

    def typical_sampling(self, text, sample_size=5, p=0.9, max_output_len=230, no_repeat_ngram=4, temperature=1):
        tokenizer_outs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=self.max_input_length)
        encs = tokenizer_outs["input_ids"]
        encs = torch.cat((encs, torch.LongTensor([[self.sep_id]])), dim=1).to(self.device) 
        sample_outputs = self.model.generate(
            encs,
            do_sample=True, 
            typical_p=p,
            max_new_tokens=max_output_len, 
            num_return_sequences=sample_size,
            no_repeat_ngram_size=no_repeat_ngram,
            temperature=temperature
        )
        outputs = {}
        outputs['output_text'], outputs["output_tokens"] = self.toks2text_batch(sample_outputs, return_tokens=True, skip_special_tokens=False)
        
        # Split at sep token
        for i in range(sample_size):
            sep_idx = outputs["output_tokens"][i].index(self.sep_id)
            outputs["output_tokens"][i] = outputs["output_tokens"][i][sep_idx+1:]
            outputs["output_text"][i] = outputs["output_text"][i].split(self.tokenizer.sep_token)[1]
        outputs_list = [{k: outputs[k][i] for k in outputs} for i in range(sample_size)]

        return outputs_list
        
    def train(self):
        self.model.train()
        self.mode = 'train'

    def eval(self):
        self.model.eval()
        self.mode = 'eval'

    def preprocess_input(self, texts, device=None):
        tokenizer_outs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding="longest")
        encs = tokenizer_outs["input_ids"]
        attention_mask = tokenizer_outs["attention_mask"]
        
        d = self.device
        if device:
            d=device

        encs = encs[:, :self.max_input_length].to(d)
        attention_mask = attention_mask[:, :self.max_input_length].to(d)
        
        return encs, attention_mask

    def preprocess_batch(self, encoded, decoded):
        encs = self.preprocess_input(encoded)

        decs = [self.tokenizer.encode(dec, add_special_tokens=False) for dec in decoded]

        decs = [dec[:(self.max_output_length-1)] for dec in decs] # We cut short, but we want the end token at the end

        decs_inp = pad([torch.LongTensor([self.sep_id]+dec) for dec in decs], padval=0).to(self.device)
        decs_out = pad([torch.LongTensor(dec+[self.tokenizer.eos_token_id]) for dec in decs], padval=-1).to(self.device)
        return encs, decs_inp, decs_out

    def encode(self, encoded_texts, model=None):
        input_ids = encoded_texts
        if model:
            model_outs = model(input_ids=input_ids, past_key_values=None)
        else:
            model_outs = self.model(input_ids=input_ids, past_key_values=None)
        return model_outs["past_key_values"]

    def decode(self, decoded_texts, past=None, decoded_targets=None):
        model_outs = self.model(input_ids=decoded_texts, past_key_values=past)
        return model_outs["logits"]

    def decode_fast(self, build_up, past, model=None):
        decoded_so_far = build_up[:, -1].view(-1, 1)
        if model:
            model_outputs = model(input_ids=decoded_so_far, 
                        past_key_values=past)
        else:
            model_outputs = self.model(input_ids=decoded_so_far, 
                        past_key_values=past)
        return model_outputs["logits"], model_outputs["past_key_values"]

    def decode_slow(self, build_up, past, model=None):
        if model:
            model_outputs = model(input_ids=build_up, 
                        past_key_values=past, use_cache=False)
        else:
            model_outputs = self.model(input_ids=build_up, 
                        past_key_values=past)
        next_token_logits = model_outputs["logits"][:, -1, :] 
        return next_token_logits

    def toks2text_batch(self, tokens_batch, return_tokens=False, skip_special_tokens=True):
        end_id = self.tokenizer.eos_token_id

        tokens_batch = [tokens[1:].tolist() + [end_id] for tokens in tokens_batch] # Add the end_id just in case
        tokens_batch = [tokens[:tokens.index(end_id)] for tokens in tokens_batch] # Cut at the end token

        # texts = [self.tokenizer.decode(tokens) for tokens in tokens_batch]
        texts = self.tokenizer.batch_decode(tokens_batch, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False) #TODO: testen...

        if not return_tokens:
            return texts 
        else:
            return texts, tokens_batch

    def train_batch(self, encoded, decoded=None, decoded_tokenized=None, no_preinput=False, input_past=None, return_logits=False):
        # if self.mode != 'train':
        #     print("BEWARE. Model is not in train mode.")
        assert decoded is not None or decoded_tokenized is not None, "Train batch should either receive decoded (list of text), or decoded_tokenized (list of list of token integers)."
        
        if decoded_tokenized is not None:
            # We are forcing to re-use pre-tokenized stuff
            encs = self.preprocess_input(encoded)
            decs_inp = pad([torch.LongTensor([self.sep_id]+dec) for dec in decoded_tokenized], padval=0).to(self.device)
            decs_out = pad([torch.LongTensor(dec+[self.tokenizer.eos_token_id]) for dec in decoded_tokenized], padval=-1).to(self.device)
        else:
            encs, decs_inp, decs_out = self.preprocess_batch(encoded, decoded)
        
        past = None
        if input_past is not None:
            past = input_past
        elif not no_preinput:
            past = self.encode(encs[0])

        crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
        logits = self.decode(decs_inp, past=past)
        if return_logits:
            return logits
        loss = crit(logits.view(-1, self.get_tokenizer_vocabsize()), decs_out.contiguous().view(-1))
        return loss

    # function to generate samples using the model.generate function
    def torch_sampling(self, encs, model, sample_size, past_key_values=None, max_output_len=100, top_k=10, top_p=0.9, no_repeat_ngram=4, temperature=1.0):
        if past_key_values:
            encs = None
        sample_outputs = model.generate(
            encs,
            do_sample=True, 
            max_new_tokens=max_output_len, 
            top_k=top_k, 
            top_p=top_p, 
            num_return_sequences=sample_size,
            no_repeat_ngram_size=no_repeat_ngram,
            temperature=temperature,
            past_key_values=past_key_values
        )
        outputs = {}
        outputs['output_text'], outputs["output_tokens"] = self.toks2text_batch(sample_outputs, return_tokens=True, skip_special_tokens=False)
        
        if past_key_values == None:
            # Split at sep token
            for i in range(sample_size):
                sep_idx = outputs["output_tokens"][i].index(self.sep_id)
                outputs["output_tokens"][i] = outputs["output_tokens"][i][sep_idx+1:]
                outputs["output_text"][i] = outputs["output_text"][i].split(self.tokenizer.sep_token)[1]
    
        outputs_list = [{k: outputs[k][i] for k in outputs} for i in range(sample_size)]
        self.parallel_outputs.extend(outputs_list)

    # function to generate samples parallely using the model.generate function
    def generate_torch_samples(self, text, parallel=False, max_output_len=100, sample_size=8, top_k=10, top_p=0.9, no_repeat_ngram=5, temperature=1.0):
        self.parallel_outputs = []
        
        tokenizer_outs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=self.max_input_length)
        encs = tokenizer_outs["input_ids"]
        encs = torch.cat((encs, torch.LongTensor([[self.sep_id]])), dim=1).to(self.device) 
        #past = self.encode(encs)
        past = None

        if parallel:
            # replicate model on gpu-2
            self.mlog.start_id("replicating")
            replicas = torch.nn.parallel.replicate(self.model, ["cuda:0", "cuda:1"])
            self.mlog.end_id("replicating")

            # sample schritt für jedes modell 
            self.mlog.start_id("parallel_sample")

            kwargs = {'past_key_values':past, 'max_output_len': max_output_len, 'top_k': top_k, 'top_p': top_p, 'no_repeat_ngram': no_repeat_ngram}
            
            n1 = ceil(sample_size/2)
            n2 = floor(sample_size/2)
            x1 = threading.Thread(target=self.torch_sampling, args=(encs, replicas[0], n1), kwargs=kwargs)
            x2 = threading.Thread(target=self.torch_sampling, args=(encs.to(replicas[1].device), replicas[1], n2), kwargs=kwargs)
            x1.start()
            x2.start() 
            x1.join()
            x2.join()

            self.mlog.end_id("parallel_sample")
        else:
            self.torch_sampling(encs, self.model, sample_size, past_key_values=past, max_output_len=max_output_len, top_k=top_k, top_p=top_p, no_repeat_ngram=no_repeat_ngram, temperature=temperature)

        return self.parallel_outputs

    def sampling(self, text, model, sample_size, split=None, max_output_len=100, top_k=0, top_p=1.0, no_copy_ngram=7, no_repeat_ngram=5, temperature=1.0):
        device = model.device
        gpu_idx = int(str(device)[-1])
        
        if split:
            Ns = [split] * int(sample_size/split)
            print(Ns)
        else:
            Ns = [sample_size]
        
        for N in Ns:
            outputs_list = []

            bodies = [text] * N
            
            inputs = self.preprocess_input(bodies, device=device)
            past = self.encode(inputs[0], model=model)
            
            build_up = torch.LongTensor([self.sep_id]).repeat(N, 1).to(device) # took sep token instead of bos in KiS
            
            seq_logprobs = torch.zeros((N)).to(device)

            finished_func = lambda build_up: all([self.end_id in build for build in build_up[:, 1:]])

            self.mlog.start_id("gen_samples_loop", gpu_idx=gpu_idx)
            while build_up.shape[1] <= max_output_len and not finished_func(build_up):
                self.mlog.end_id("gen_samples_loop", gpu_idx=gpu_idx)
                self.mlog.start_id("gen_samples_loop", gpu_idx=gpu_idx)
                    
                logits, past = self.decode_fast(build_up, past, model=model)
                logits = logits.view(N, -1)

                logits = ngram_copy_filtering(build_up, inputs[0], logits, n_gram=no_copy_ngram)
                logits = ngram_copy_filtering(build_up, build_up, logits, n_gram=no_repeat_ngram)
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

                probs = torch.nn.functional.softmax(logits/temperature, dim=-1).squeeze(1)
                distrib = torch.distributions.categorical.Categorical(probs)
                current = distrib.sample().unsqueeze(-1)

                current = current.view(-1, 1)
                build_up = torch.cat((build_up, current), dim=1)

                not_finished = (1-torch.any(build_up[:, 1:]==self.end_id, dim=1).float()).to(device)
                seq_logprobs += not_finished * logprobs[torch.arange(N), current.view(N)].view(N)
                
            outputs = {}
            outputs['output_text'], outputs["output_tokens"] = self.toks2text_batch(build_up, return_tokens=True)
            outputs['logprob'] = seq_logprobs.tolist()

            outputs_list.extend([{k: outputs[k][i] for k in outputs} for i in range(N)])
            self.parallel_outputs.extend(outputs_list)


    def generate_samples(self, text, parallel=False, split=None, max_output_len=100, sample_size=4, top_k=0, top_p=1.0, no_copy_ngram=7, no_repeat_ngram=5, temperature=1.0):
        self.parallel_outputs = []
        
        #inputs = self.preprocess_input(bodies, device=self.device)
        #past = self.encode(inputs[0], model=model) #TODO

        if parallel:
            # replicate model on gpu-2
            self.mlog.start_id("replicating")
            replicas = torch.nn.parallel.replicate(self.model, ["cuda:0", "cuda:1"])
            self.mlog.end_id("replicating")

            # sample schritt für jedes modell 
            self.mlog.start_id("parallel_sample")

            kwargs = {'split': split, 'max_output_len': max_output_len, 'top_k': top_k, 'top_p': top_p, 'no_copy_ngram': no_copy_ngram, 'no_repeat_ngram': no_repeat_ngram, 'temperature': temperature}
            
            n1 = ceil(sample_size/2)
            n2 = floor(sample_size/2)
            x1 = threading.Thread(target=self.sampling, args=(text, replicas[0], n1), kwargs=kwargs)
            x2 = threading.Thread(target=self.sampling, args=(text, replicas[1], n2), kwargs=kwargs)
            x1.start()
            x2.start() 
            x1.join()
            x2.join()

            self.mlog.end_id("parallel_sample")
        else:
            self.sampling(text, self.model, sample_size, split=split, max_output_len=max_output_len, top_k=top_k, top_p=top_p, no_copy_ngram=no_copy_ngram, no_repeat_ngram=no_repeat_ngram, temperature=temperature)

        return self.parallel_outputs


if __name__ == '__main__':
    None


