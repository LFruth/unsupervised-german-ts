from transformers import AutoTokenizer, AutoModelForMaskedLM
from bert_score import plot_example
from torch.nn.modules.loss import CrossEntropyLoss

import torch
import numpy as np
import nltk

from misc.utils_text import TextPreprocessing, shift_score
from misc.utils_gpu import get_device
from sentence_transformers import SentenceTransformer
from misc import utils_masking


"""
The code is from Keep it Simple https://github.com/tingofurro/keep_it_simple/
"""

class BScoreSimilarity:
    # Measures the Meaning Preservation only with the BERTScore similarity

    def __init__(self, bert_scorer):# model="dbmdz/bert-base-german-cased", num_layers=5):
        #self.model_name = model
        #self.num_layers = num_layers
        self.preproc = TextPreprocessing()

        self.bscorer = bert_scorer#BERTScorer(model_type=model, num_layers=num_layers)


    def score(self, c_texts, s_texts, min=0.4, max=0.85, printing=False):
        scores = []
        
        #P, R, F1 = score(c_texts, s_texts, model_type=self.model_name, num_layers=self.num_layers)
        P, R, F1 = self.bscorer.score(c_texts, s_texts)

        scores = F1
        
        scores = shift_score(scores, min, max)
        return scores.tolist()

    # Only considers the keywords
    def score_nosw(self, c_texts, s_texts, min=0.55, max=0.9, printing=False):
        scores = []

        result = self.bscorer.raw_scores(c_texts, s_texts)
        
        for (ref, cand) in zip(result['refs'], result['cands']):
            ref_bscore = []
            cand_bscore = []

            for word in ref:
                if self.preproc.is_good_word(word['word']):
                    ref_bscore.append(word['max'])
            for word in cand:
                if self.preproc.is_good_word(word['word']):
                    cand_bscore.append(word['max'])
            
            recall = sum(ref_bscore) / len(ref_bscore)
            precision = sum(cand_bscore) / len(cand_bscore)
            f1 = 2 * ( (recall*precision) / (precision + recall) )
            scores.append(f1)
        scores = shift_score(scores, min, max)
        return scores


    def plot_example(self, t1, t2):
        plot_example(t1, t2, model_type=self.model_name, num_layers=self.num_layers, fname="tmp")
        #plot_example(t1, t2, lang='de')



class TextSimilarity:
    # Measures the Meaning Preservation with a combination of sentence alignment and BERTScore similarity

    def __init__(self, bert_scorer, model_card="sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned", max_seq_len=250, gpu_idx=1):
        self.model = SentenceTransformer(model_card)
        self.device = get_device(gpu_idx=gpu_idx)
        self.model.to(self.device)
        self.model.max_seq_len = max_seq_len
        self.bert_scorer = bert_scorer

    def get_sents(self, orig, simpl):
        return nltk.sent_tokenize(orig, language='german'), nltk.sent_tokenize(simpl, language='german')

    def encode_sents(self, sents_o, sents_s):
        return self.model.encode(sents_o, convert_to_tensor=True), self.model.encode(sents_s, convert_to_tensor=True)

    def get_sentence_alignments(self, embs_o, embs_s, sents_o, sents_s, printing=False):
        cos = torch.nn.CosineSimilarity(dim=0)
        alignments = {}
        alignments[-1] = 0

        for i in range(len(embs_s)):
            sims = [] 
            for j in range(len(embs_o)):
                sims.append(cos(embs_s[i], embs_o[j]))
            max_val = max(sims)
            orig_idx = sims.index(max_val) # index of the most similar sentence in the original text
            
            if printing:
                print("\n Original: ", sents_o[orig_idx])
                print("Simplification: ", sents_s[i])
                print("Value: ", max_val)
                
            if orig_idx not in alignments.keys():
                alignments[orig_idx] = [i]
            else:
                alignments[orig_idx].append(i)

            if max_val < 0.25: # penalty if simpl sentence has a very low similarity to a specific sentence
                alignments[-1] += 1

        return alignments

    def get_sentence_scores(self, embs_o, embs_s):
        cos = torch.nn.CosineSimilarity(dim=0)
        s_scores = {}

        for i in range(len(embs_s)):
            sims = [] 
            for j in range(len(embs_o)):
                sims.append(cos(embs_s[i], embs_o[j]))
            max_val = max(sims)
            orig_idx = sims.index(max_val) # index of the most similar sentence in the original text
            
            if orig_idx not in s_scores.keys():
                s_scores[orig_idx] = [max_val]
            else:
                s_scores[orig_idx].append(max_val)
        return s_scores


    def score(self, c_texts, s_texts, min=0.1, max=0.8, printing=False):
        scores = []

        for c, s in zip(c_texts, s_texts):
            sents_o, sents_s = self.get_sents(c, s)
            if len(sents_s) > 0:
                embs_o, embs_s = self.encode_sents(sents_o, sents_s)
                
                aligns = self.get_sentence_alignments(embs_o, embs_s, sents_o, sents_s, printing=printing)
                score = []

                aligned_sents = []
                for idx in range(len(sents_o)):
                    if idx in aligns.keys():
                        #o_emb = embs_o[idx]
                        s_sent = ''
                        for s in aligns[idx]:
                            s_sent += sents_s[s] + " "

                        aligned_sents.append([sents_o[idx], s_sent])

                        # sentence-transformers scoring
                        #s_emb = self.model.encode(s_sent, convert_to_tensor=True)
                        #score.append(torch.cosine_similarity(o_emb, s_emb, dim=0).tolist())
                        
                    else:
                        score.append(0) # original sentence has no aligned simplification sentence
                
                # bertscore of aligned sentences
                aligned_sents = np.array(aligned_sents)
                P, R, F1 = self.bert_scorer.score(aligned_sents[:, 0].tolist(), aligned_sents[:, 1].tolist())
                if printing:
                    print(F1.tolist())
                score.extend(F1.tolist())

                score.extend([0] * aligns[-1]) # add a penalty for each low similarity sentence
                
                scores.append(np.mean(score))
            else:
                scores.append(0)
        scores = shift_score(scores, min, max)
        return scores


def unfold(sent_toks, make_tensor=True):
    unfolded = [w for sent in sent_toks for w in sent]
    if make_tensor:
        unfolded = torch.LongTensor(unfolded)
    return unfolded

class CoverageModel:
    # The Coverage model from Keep it Simple https://github.com/tingofurro/keep_it_simple/ and
    # The Summary Loop https://github.com/CannyLab/summary_loop

    def __init__(self, masking_model, model_card="bert-base-german-cased", device="cuda", model_file=None, is_soft=False, normalize=False):
        self.model_card = model_card

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_card)

        self.eos_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.eos_token_id is None:
            self.eos_token_id = 0

        if type(masking_model) == str:
            self.masking_model = utils_masking.string2mask(masking_model)
        else:
            self.masking_model = masking_model
        self.masking_model.register_tokenizer(self.tokenizer)

        self.vocab_size = self.tokenizer.vocab_size
        self.device = device
        self.mask_id = self.tokenizer.mask_token_id
        self.masking_model.mask_id = self.mask_id

        self.normalize = normalize
        self.is_soft = is_soft
        if is_soft:
            print("Coverage will be soft.")

        self.model.to(self.device)
        if model_file is not None:
            self.reload_model(model_file)

    def reload_model(self, model_file, strict=True):
        print(self.model.load_state_dict(torch.load(model_file, map_location=self.device), strict=strict))

    def save_model(self, model_file):
        torch.save(self.model.state_dict(), model_file)

    def process_text(self, document):
        sentences = [" "+sent for sent in nltk.tokenize.sent_tokenize(document, language='german') if len(sent) > 0]
        unmasked, masked, is_masked, mr_eff = self.masking_model.mask(sentences)
        return unfold(unmasked), unfold(masked), unfold(is_masked), mr_eff

    def build_io(self, targets, generateds):
        N = len(targets)

        input_ids, labels, is_masked, mr_effs = [], [], [], []
        gen_toks = []

        for target, generated in zip(targets, generateds):
            unmasked, masked, is_ms, mr_eff = self.process_text(target)
            input_ids.append(masked)
            labels.append(unmasked)
            is_masked.append(is_ms)
            gen_toks.append(torch.LongTensor(self.tokenizer.encode(generated, add_special_tokens=False)))
            mr_effs.append(mr_eff)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        is_masked = torch.nn.utils.rnn.pad_sequence(is_masked, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
        input_ids = input_ids[:, :250]
        is_masked = is_masked[:, :250]
        labels = labels[:, :250]

        gen_toks = torch.nn.utils.rnn.pad_sequence(gen_toks, batch_first=True, padding_value=0)
        gen_toks = gen_toks[:, :250]
        gen_targets = torch.LongTensor([-1]).repeat(gen_toks.shape)

        seps = torch.LongTensor([self.eos_token_id]).repeat(N, 1)
        seps_targets = torch.LongTensor([-1]).repeat(seps.shape)

        input_ids = torch.cat((gen_toks, seps, input_ids), dim=1)
        labels = torch.cat((gen_targets, seps_targets, labels), dim=1)
        is_masked = torch.cat((torch.zeros_like(gen_toks), torch.zeros_like(seps), is_masked), dim=1)

        labels = labels.to(self.device)
        input_ids = input_ids.to(self.device)
        is_masked = is_masked.to(self.device)

        return input_ids, is_masked, labels, mr_effs

    def train_batch(self, contents, summaries):
        input_ids, is_masked, labels, mr_effs = self.build_io(contents, summaries)

        outputs = self.model(input_ids)
        logits = outputs["logits"]
        cross_ent = CrossEntropyLoss(ignore_index=-1)
        loss = cross_ent(logits.view(-1, self.vocab_size), labels.view(-1))

        num_masks = torch.sum(is_masked, dim=1).float() + 0.1
        with torch.no_grad():
            preds = torch.argmax(logits, dim=2)
            accs = torch.sum(preds.eq(labels).long() * is_masked, dim=1).float() / num_masks

        return loss, accs.mean().item()

    def score(self, bodies, decodeds, **kwargs):
        score_func = self.score_soft if self.is_soft else self.score_hard
        unnorm_scores = score_func(bodies, decodeds, **kwargs)

        if self.normalize:
            empty_scores = score_func(bodies, [""] * len(bodies), **kwargs)
            zero_scores = np.array(empty_scores["scores"])

            norm_scores = {k: v for k, v in unnorm_scores.items()}
            norm_scores["scores"] = ((np.array(unnorm_scores["scores"]) - zero_scores) / (1.0 - zero_scores))
            norm_scores["scores"] = norm_scores["scores"].tolist()
            return norm_scores
        else:
            return unnorm_scores

    def score_hard(self, bodies, decodeds, **kwargs):
        self.model.eval()
        with torch.no_grad():
            input_ids_w, is_masked_w, labels_w, mr_effs = self.build_io(bodies, decodeds)
            
            outputs_w = self.model(input_ids_w)
            preds_w = torch.argmax(outputs_w["logits"], dim=2)
            num_masks_w = torch.sum(is_masked_w, dim=1).float() + 0.1
            accs_w = torch.sum(preds_w.eq(labels_w).long() * is_masked_w, dim=1).float() / num_masks_w

            #     input_ids_wo, is_masked_wo, labels_wo = self.build_io(bodies, [""] * len(bodies))
            #     outputs_wo, = self.model(input_ids_wo)
            #     preds_wo = torch.argmax(outputs_wo, dim=2)
            #     num_masks_wo = torch.sum(is_masked_wo, dim=1).float() + 0.1
            #     accs_wo = torch.sum(preds_wo.eq(labels_wo).long() * is_masked_wo, dim=1).float() / num_masks_wo
        scores = accs_w # - accs_wo
        scores = scores.tolist()
        return {"scores": scores, "mr_eff": mr_effs}

    def score_soft(self, bodies, decodeds, printing=False, **kwargs):
        input_ids_w, is_masked_w, labels_w, mr_effs = self.build_io(bodies, decodeds)
        scores = self.score_soft_tokenized(input_ids_w, is_masked_w, labels_w)

        if printing:
            print("[coverage]", scores)

        return {"scores": scores, "mr_eff": mr_effs}

    def score_soft_tokenized(self, input_ids_w, is_masked_w, labels_w):
        self.model.eval()
        with torch.no_grad():
            outputs_w = self.model(input_ids_w)
            outputs_probs_w = torch.softmax(outputs_w["logits"], dim=2)
            max_probs, _ = outputs_probs_w.max(dim=2)

            relative_probs_w = (outputs_probs_w.permute(2, 0, 1) / max_probs).permute(1, 2, 0)

            batch_size, seq_len = is_masked_w.shape
            t_range = torch.arange(seq_len)

            scores = []
            for seq_rel_probs, seq_labels, seq_is_masked in zip(relative_probs_w, labels_w, is_masked_w):
                selected_probs = (seq_rel_probs[t_range, seq_labels])*seq_is_masked
                soft_score = torch.sum(selected_probs) / (torch.sum(seq_is_masked)+0.1)
                scores.append(soft_score.item())

        return scores

    def print_predictions(self, text, simpl):
        inp, im, labels, mr_effs = self.build_io([text], [simpl])
        
        outputs = self.model(inp)
        logits = outputs["logits"]

        with torch.no_grad():
            preds = torch.argmax(logits, dim=2)
            
        print("Input: ", self.tokenizer.decode(inp[0]))
        print("Output: ", self.tokenizer.decode(preds[0]))




if __name__ == '__main__':
    s = Saliency()

    t1 = "Nik sitzt in der Küche. Er raucht eine Kippe."
    t2 = "Nik hängt nur rum und raucht Zigaretten."

    print(s.score_no_sw([t1], [t2]))
    print(s.score([t1], [t2]))