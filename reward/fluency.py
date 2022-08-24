import torch
import numpy as np
import wandb
import tqdm

from transformers import AutoTokenizer, AutoModelWithLMHead, BertTokenizerFast,  AutoModelForSequenceClassification
from transformers.optimization import AdamW
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.cuda.amp import autocast
from collections import Counter
from sklearn.metrics import f1_score
from misc.utils_gpu import get_device


"""
The code is from Keep it Simple https://github.com/tingofurro/keep_it_simple/
"""

# The score for measuring the LM-Fluency
class Fluency:
    def __init__(self, model_card="dbmdz/bert-base-german-cased", model_file=None, device_idx=1):
        self.device = get_device(gpu_idx=device_idx)
        print("Fluency model on %s" % self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.model = AutoModelWithLMHead.from_pretrained(model_card).to(self.device)
        self.model.eval()

        if model_file is not None:
            self.reload_model(model_file)

    def reload_model(self, model_file, strict=True):
        print(self.model.load_state_dict(torch.load(model_file, map_location=self.device), strict=strict))

    def preprocess_batch(self, decoded):
        # We cut short, but we want the end token at the end
        max_output_length = 240
        decs = [self.tokenizer.encode(dec) for dec in decoded]
        decs = [dec[:(max_output_length-1)] for dec in decs]
        
        decs_inp = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(dec) for dec in decs], batch_first=True, padding_value=0).to(self.device)
        decs_out = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(dec) for dec in decs], batch_first=True, padding_value=-1).to(self.device)
        #decs_inp = torch.nn.utils.rnn.pad_sequence([torch.LongTensor([self.tokenizer.bos_token_id]+dec) for dec in decs], batch_first=True, padding_value=0)
        #decs_out = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(dec+[self.tokenizer.eos_token_id]) for dec in decs], batch_first=True, padding_value=-1)

        return decs_inp, decs_out

    def text2loss(self, texts):
        txt_inp, txt_out = self.preprocess_batch(texts)

        with torch.no_grad():
            model_outputs = self.model(input_ids=txt_inp)#, past_key_values=None)

            crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none').cuda() if torch.cuda.is_available() else torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = crit(model_outputs["logits"].view(-1, self.tokenizer.vocab_size), txt_out.view(-1)).view(txt_out.shape)
            mask = (txt_inp != torch.LongTensor([0]).to(self.device)).float()
            non_pad_count = torch.sum(mask, dim=1)
            loss_per = torch.sum(loss, dim=1) / non_pad_count
        return loss_per

    def score(self, sources, generateds, lamb=0.12, printing=False):
        sources_score = self.text2loss(sources)
        generateds_score = self.text2loss(generateds)
        scores = (lamb + sources_score - generateds_score) / lamb
        scores = torch.clamp(scores, 0.001, 1.0).tolist()

        if printing:
            print("[fluency]", scores)
        #return {"scores": scores, "sources_loss": sources_score, "generateds_loss": generateds_score}
        return scores


# The discriminator, predicting if a text is generated or human-written
class TextDiscriminator:

    def __init__(self, retrain_every=4000, fp16=False, log_wandb=False):
        # retrain_every: once the cache reaches that amount, the model is retrained.
        # fp16: Use half-precision for training

        self.fp16 = fp16
        self.device = get_device(gpu_idx=1)
        print("TextDiscriminator model on %s" % self.device)

        self.tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-german-cased")
        self.discriminator = None
        self.optimizer = None
        self.optim_every = 2

        self.trained = False

        self.last_val_f1 = 0.0
        self.retrain_every = retrain_every
        self.cache_sources, self.cache_generateds = [], []

        self.learning_rate = 1e-5 #10^(-5)
        self.n_epochs = 5
        self.batch_size = 6
        
        self.log_wandb = log_wandb

        if self.log_wandb:
            w_config = dict(epochs = self.n_epochs, batch_size = self.batch_size, learning_rate = self.learning_rate)
            wandb.init(project="discriminator", config=w_config)


    def train_from_dataset(self, texts, labels, n_epochs=5):
        self.n_epochs = n_epochs

        # Process data
        toks = [torch.LongTensor(self.tokenizer.encode(text))[:260] for text in texts]
        toks = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=0)

        dataset = TensorDataset(torch.LongTensor(toks), torch.LongTensor(labels))
        N_dev = min(100, int(0.1*len(dataset)))
        N_train = len(dataset) - N_dev

        d_train, d_dev = torch.utils.data.dataset.random_split(dataset, [N_train, N_dev])
        print("Num train: %d, num dev: %d; Label Count: %s" %(len(d_train), len(d_dev), str(Counter(labels))))

        train_sampler, dev_sampler = RandomSampler(d_train), RandomSampler(d_dev)

        train_dataloader = DataLoader(d_train, sampler=train_sampler, batch_size=self.batch_size)
        dev_dataloader = DataLoader(d_dev, sampler=dev_sampler, batch_size=N_dev)

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Load model:
        self.discriminator = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-german-cased", num_labels=2)
        self.discriminator.to(self.device)
        
        # Optimizer
        param_optimizer = list(self.discriminator.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # Train
        label_counter = Counter(labels)
        imbalance_weight = torch.FloatTensor([len(labels) / label_counter[0], len(labels) / label_counter[1]])
        
        if self.fp16:
            imbalance_weight = imbalance_weight.half()

        #if torch.cuda.is_available(): imbalance_weight.cuda()

        print("Disc Imbalance Weights:", imbalance_weight.tolist())

        crit = torch.nn.CrossEntropyLoss(weight=imbalance_weight).cuda() if torch.cuda.is_available() else torch.nn.CrossEntropyLoss(weight=imbalance_weight)
        
        best_state_dict = None
        best_f1 = 0.0
        for _ in range(n_epochs):

            print("New training epoch")
            self.discriminator.train()
            losses = []
            for i, batch in enumerate(tqdm.tqdm(train_dataloader)):
                batch_inputs, batch_labels = tuple(t.to(self.device) for t in batch)
                with autocast(self.fp16):
                    model_outputs = self.discriminator(batch_inputs) # , labels=batch_labels
                
                logits = model_outputs["logits"]
                
                loss = crit(logits, batch_labels)
                loss.backward()

                if i % self.optim_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                losses.append(loss.item())

            self.discriminator.eval()
            print("Train loss: %.3f" % (np.mean(losses)))
            with torch.no_grad():
                total_preds, total_labels = [], []
                for batch in dev_dataloader:
                    batch_inputs, batch_labels = tuple(t.to(self.device) for t in batch)
                    model_outputs = self.discriminator(batch_inputs)
                    preds = torch.argmax(model_outputs["logits"], axis=1).tolist()
                    total_labels += [l.item() for l in batch_labels]
                    total_preds += preds
                val_accuracy = np.mean(np.array(total_preds) == np.array(total_labels))
                val_f1 = f1_score(total_labels, total_preds, average="micro")

                if val_f1 >= best_f1:
                    best_state_dict = self.discriminator.state_dict()
                    best_f1 = val_f1
                print("Discriminator Validation. [Acc: %.3f] [F-1: %.3f]" % (val_accuracy, val_f1))
            
            if self.log_wandb:
                wandb.log({"train_loss": np.mean(losses), "eval_acc": val_accuracy})
                wandb.watch(self.discriminator)

        self.discriminator.load_state_dict(best_state_dict)
        self.discriminator.eval()
        self.optimizer = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        total_preds, total_labels, total_pred_1s = [], [], []
        with torch.no_grad():
            for batch in dev_dataloader:
                batch_inputs, batch_labels = tuple(t.to(self.device) for t in batch)
                model_outputs = self.discriminator(batch_inputs)
                preds_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=1)
                preds = torch.argmax(preds_probs, dim=1).tolist()
                total_labels += [l.item() for l in batch_labels]
                total_preds += preds
                prob_1s = preds_probs[:, 1]
                total_pred_1s += prob_1s.tolist()

            val_accuracy = np.mean(np.array(total_preds) == np.array(total_labels))
            val_f1 = f1_score(total_labels, total_preds, average="micro")

        print("[Final Discriminator] [Accuracy: %.3f] [F1: %.3f] [Average prediction: %.3f]" % (val_accuracy, val_f1, np.mean(total_pred_1s)))
        self.last_val_f1 = val_f1
        print("================")


    def retrain_auto(self):
        self.trained = True

        texts0 = list(set(self.cache_generateds))
        texts1 = list(set(self.cache_sources))
        print("[Disc] Number of negative samples: %d" % (len(texts0)))
        print("[Disc] Number of positive samples: %d" % (len(texts1)))

        texts = texts0 + texts1
        labels = ([0] * len(texts0)) + ([1] * len(texts1))

        self.train_from_dataset(texts, labels, n_epochs=5)

    def score(self, sources, generateds, printing=False, **kwargs):
        self.cache_sources += sources
        self.cache_generateds += generateds

        if len(set(self.cache_generateds) | set(self.cache_sources)) >= self.retrain_every:
            self.retrain_auto()
            self.cache_generateds = []
            self.cache_sources = []

        # If the model has not been trained yet
        if not self.trained:
            # Make it small but non-zero arbitrarily so that the multiplied score isn't nulled
            scores = torch.FloatTensor([0.2] * len(generateds)).to(self.device)
        else:
            # Do the actual scoring
            generateds = [text if len(text) > 0 else "empty text" for text in generateds] # Trying to fix the empty sequence problem
            toks = [torch.LongTensor(self.tokenizer.encode(text))[:200] for text in generateds]
            toks = [tok if len(tok) > 0 else [1] for tok in toks] # Make sure the sequence length is not zero, otherwise it crashes
            toks = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=0).cuda() if torch.cuda.is_available() else torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=0)

            with torch.no_grad():
                model_outputs = self.discriminator(toks)
                probs = torch.nn.functional.softmax(model_outputs["logits"], dim=1)
                scores = torch.clamp(probs[:, 1], 0.0001, 1.0)

        scores = scores.tolist()
        if printing:
            print("[discriminator]", scores)

        #return {"scores": scores, "val_f1": [self.last_val_f1] * len(scores)}
        return scores