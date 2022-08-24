import numpy as np
import regex as re
import torch
import nltk

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from bert_score import get_bert_embedding, sent_encode
from misc.utils_text import TextPreprocessing
from collections import defaultdict
from misc.utils_gpu import get_device
from difflib import get_close_matches
from collections import Counter

"""
The code is from Keep it Simple https://github.com/tingofurro/keep_it_simple/
"""

class Brevity:
    def __init__(self, min_ratio=0.6, max_ratio=1.3):#, min_target=0.8, max_target=1.2):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        #self.min_target = min_target
        #self.max_target = max_target

    def compute_compression_ratio(self, source, generated):
        return float(len(generated) / len(source))

    def shift_to_score(self, c_ratio):
        if c_ratio >= self.min_target and c_ratio <= self.max_target:
            return 1
        elif c_ratio > self.min_ratio and c_ratio < self.min_target:
            return (c_ratio - self.min_ratio) / (self.min_target - self.min_ratio)
        elif c_ratio < self.max_ratio and c_ratio > self.max_target:
            return 1 - (c_ratio - self.max_target) / (self.max_ratio - self.max_target)
        else: 
            return 0

    def score(self, sources, generateds, printing=False):
        brevity_scores = []
        for source, generated in zip(sources, generateds):
            compression_ratio = self.compute_compression_ratio(source, generated)
            #score = self.shift_to_score(compression_ratio)
            if compression_ratio > self.max_ratio or compression_ratio < self.min_ratio:
                score = 0
            else:
                score = 1
            brevity_scores.append(score)
        return brevity_scores 


class RepeatNGramPenalty:
    def __init__(self, gram=3, keep_stop_ws=False):
        self.gram = gram
        self.stop_words = set(nltk.corpus.stopwords.words("german"))
        self.keep_stop_ws = keep_stop_ws

    def score(self, sources, generateds, printing=False):
        scores = []
        for generated in generateds:
            words = nltk.tokenize.word_tokenize(generated.lower(), language='german')
            n_grams = [tuple(words[i:(i+self.gram)]) for i in range(len(words)-self.gram+1)]

            if not self.keep_stop_ws:
                n_grams = [ngram for ngram in n_grams if any(w not in self.stop_words for w in ngram)]

            n_repeated_three_grams = len([ngram for ngram, count in Counter(n_grams).most_common() if count > 1])
            repeat_penalty = 0 if n_repeated_three_grams > 0 else 1.0
            scores.append(repeat_penalty)
        return scores


class RepeatArticles:
    def __init__(self, gram=4):
        self.gram = gram
        self.stop_words = ["der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines"]
        
    def is_repeating_articles(self, gram):
        last_word = ''
        sws = 0
        for word in gram:
            if word in self.stop_words:
                sws += 1
            if last_word == word:
                return True
            last_word = word
        if sws >= 3:
            return True
        return False

    def score(self, sources, generateds, printing=False):
        scores = []
        for generated in generateds:
            penalty = 1

            words = nltk.tokenize.word_tokenize(generated.lower(), language='german')
            n_grams = [tuple(words[i:(i+self.gram)]) for i in range(len(words)-self.gram+1)]
            for gram in n_grams:
                if self.is_repeating_articles(gram):
                    penalty = 0
                    break
            scores.append(penalty)
        return scores


class HallucinationModel:

    def __init__(self, lemma_model, bert_scorer, tagger_model_card="Davlan/xlm-roberta-base-ner-hrl", bscore_threshold=0.74):
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = get_device(gpu_idx=1)
        print("Hallucination model on %s" % self.device)

        self.lemma_model = lemma_model

        tokenizer = AutoTokenizer.from_pretrained(tagger_model_card, truncation=True)
        model = AutoModelForTokenClassification.from_pretrained(tagger_model_card, max_length=500).to(self.device)
        if self.device != 'cpu':
            d = int(str(self.device)[-1])
        else:
            d = -1
        self.ner = pipeline("ner", model=model, tokenizer=tokenizer, device=d)

        self.tokenizer = bert_scorer._tokenizer#get_tokenizer(model_name, use_fast=True)
        self.model = bert_scorer._model#get_model(model_name, num_layers)
        #self.model.to(self.device)

        self.preproc = TextPreprocessing()

        self.bscore_threshold = bscore_threshold

    def tag_entities(self, text, accuracy=0.75, printing=False):
        entities = self.ner(text, aggregation_strategy='simple')
        
        tag_results = []
        # stitch together:
        for idx in range(len(entities)):
            if entities[idx]['score'] >= accuracy:
                tag_results.append(entities[idx])
        return tag_results

    def bscore(self, refs, cands, printing=False):
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[self.tokenizer.sep_token_id] = 0
        idf_dict[self.tokenizer.cls_token_id] = 0
        
        results = []

        for reference, candidate in zip(refs, cands):
            cand_embedding, masks, padded_idf = get_bert_embedding(
                [candidate], self.model, self.tokenizer, idf_dict, device=self.device, all_layers=False
            )
            ref_embedding, masks, padded_idf = get_bert_embedding(
                [reference], self.model, self.tokenizer, idf_dict, device=self.device, all_layers=False
            )
            

            ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
            cand_embedding.div_(torch.norm(cand_embedding, dim=-1).unsqueeze(-1))
            sim = torch.bmm(cand_embedding, ref_embedding.transpose(1, 2))
            sim = sim.squeeze(0).cuda() if torch.cuda.is_available() else sim.squeeze(0).cpu()
            # remove [CLS] and [SEP] tokens
            sim = sim[1:-1, 1:-1]
            

            encoded = sent_encode(self.tokenizer, candidate)

            tokens = self.tokenizer.convert_ids_to_tokens(encoded)[1:-1]
            
            # map where word begins and ends
            token_word_mapping = []
            idx = 0
            while idx < len(tokens):
                off = 1 
                if (idx+off) < len(tokens):

                    while tokens[idx+off].startswith("##"):
                        off += 1
                        if (idx+off) >= len(tokens):
                            break

                    token_word_mapping.append((idx, idx+off))
                idx = idx+off

            cand_tokens = [self.tokenizer.decode([i]) for i in encoded][1:-1]

            
            # merge tokens to words
            result = {}
            for word_maps in token_word_mapping:
                
                s, e = word_maps
                word = "".join(cand_tokens[s:e]).replace("##", "")
                
                max_sim = torch.max(sim[:][s:e], dim=1)
                max_sim = torch.mean(max_sim.values)
                result[word] = max_sim

            results.append(result)
            

        return results
        
    
    def lemmatize_word(self, word):
        lemma = self.lemma_model.lemmatize_words([word])[0]
        return lemma

    def get_word_bscore(self, word, bscores, printing=False):
        word = ''.join(w for w in word if w.isalnum())
        matched_word = get_close_matches(word, bscores.keys(), n=1)
        
        # if nothing that matches is found... using threshold
        if matched_word == None or len(matched_word) <= 0 or len(matched_word[0]) <= 1 or matched_word[0] not in bscores.keys():
            return torch.tensor(self.bscore_threshold)

        score = bscores[matched_word[0]]
        if printing:
            print(score)
        #if score <= self.bscore_threshold:
            #print("BScore for '%s': %.3f" % (word, score))
        return score


    def get_entity_bscores(self, entity, bscores, printing=False):
        if isinstance(entity, str):
            entity = [entity]
        
        score = []
            
        for word in entity:
            score.append(self.get_word_bscore(word, bscores, printing).to('cpu')) 

        return np.mean(score)


    def check_tag_score(self, tags, bscore, printing=False):
        for ent in tags:
            if printing:
                print("\nComparing ", ent['word'])

            words = re.split("[\s▁-]", ent['word'])
                    
            ent_bscores = self.get_entity_bscores(words, bscore, printing=printing)
                    
            if np.mean(ent_bscores) < self.bscore_threshold:
                if printing:
                    print("%s is invalid (%.2f < %.2f)" % (ent['word'], np.mean(ent_bscores), self.bscore_threshold))
                return 0
        return 1


    def score(self, sources, generateds, printing=False):
        
        tag_results = []
        for text in generateds:
            tag_results.append(self.tag_entities(text))

        bscores = self.bscore(sources, generateds, printing=printing)

        score = []
        for doc_idx in range(len(tag_results)):
            if printing:
                print("\nDocument ", doc_idx)

            result = self.check_tag_score(tag_results[doc_idx], bscores[doc_idx], printing=printing)

            score.append(result)

        return score

