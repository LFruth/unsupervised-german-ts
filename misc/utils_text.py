import nltk
import numpy as np
import torch
from HanTa import HanoverTagger as ht

def shift_score(score, min, max, sigmoid=False):
    for idx in range(len(score)):
        v = score[idx]
        v = (v - min) / (max - min)
        score[idx] = np.clip(v, 0.001, 1)
    return score

# TODO: refactor
def select_logprobs(logits, decoded_tokens, eos_id, device):
    logprobs = torch.nn.functional.log_softmax(logits, dim=2)

    selected_logprobs = []
    for i, generated_tokenized in enumerate(decoded_tokens):
        generated_tokenized.append(eos_id)
        generated_tokenized = generated_tokenized[:generated_tokenized.index(eos_id)] # Remove probs of stuff after end tokens
        selected_logprob = logprobs[i, torch.arange(len(generated_tokenized)), generated_tokenized]
        summed_logprob = torch.sum(selected_logprob)
        selected_logprobs.append(summed_logprob)
    selected_logprobs = torch.stack(selected_logprobs, dim=0)
    selected_logprobs.to(device)
    return selected_logprobs  

class TextPreprocessing:

    def __init__(self):
        self.stopws = set(nltk.corpus.stopwords.words("german") + ["``", "--"])

    def is_good_word(self, w):
        if "'" in w:
            return False
        if len(w) > 30 or len(w) == 1:
            return False
        if w.lower() in self.stopws:
            return False
        if all(c.isdigit() for c in w):
            return False
        return True

    def preprocess_text(self, txt):
        ## Tokenizes text and filters stopwords, chars etc.
        words = nltk.tokenize.word_tokenize(txt)
        words = [w.lower() for w in words if self.is_good_word(w)]
        return words

class LemmaModel:

    def __init__(self, model_path="./models/morphmodel_ger.pgz"):
        self.model = ht.HanoverTagger(model_path)

    def lemmatize_words(self, words):
        lemmas = [lemma for (word, lemma, pos) in self.model.tag_sent(words)]
        return lemmas
    
