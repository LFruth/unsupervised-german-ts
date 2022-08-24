from wordfreq import zipf_frequency
from torch import clamp, Tensor
import numpy as np, nltk, textstat

"""
The code is from Keep it Simple https://github.com/tingofurro/keep_it_simple/
"""

# we have a target shift. If you go beyond that, you should get penalized, but at a slower rate (right_slope).
def shift_to_score(shift, target_shift, right_slope=0.25):
    if shift <= target_shift:
        score = shift / (target_shift+0.001)
    else:
        score = 1.0 - right_slope * (shift - target_shift) / (target_shift+0.001)
    return score


# Vocab (V) and Readability (R) Shift models
class LexicalSimplicity:

    def __init__(self, lemma_model, target=0.8, word_change_ratio=0.15):        
        self.lemma = lemma_model
        self.target = target
        self.word_change_ratio = word_change_ratio # Number of words that we expect to be swapped

        self.stopws = set(nltk.corpus.stopwords.words("german") + ["der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "``", "--", "--"])

    def word_score_func(self, w): 
        return zipf_frequency(w, 'de', wordlist="large")

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

    def zipf_score(self, txt):
        words = nltk.tokenize.word_tokenize(txt, language='german')
        words = self.lemma.lemmatize_words(words)
        words = set([w.lower() for w in words if self.is_good_word(w)])

        zipfs = [self.word_score_func(w) for w in words]
        return zipfs

    def lexical_shift(self, c_txt, s_txt, printing=False):
        words1 = nltk.tokenize.word_tokenize(c_txt)
        words2 = nltk.tokenize.word_tokenize(s_txt)
        # Lemmatizing
        words1 = self.lemma.lemmatize_words(words1)
        words2 = self.lemma.lemmatize_words(words2)
        # Filtering
        words1 = set([w.lower() for w in words1 if self.is_good_word(w)])
        words2 = set([w.lower() for w in words2 if self.is_good_word(w)])

        removed_words = words1 - words2
        added_words = words2 - words1

        target_n_words = int(self.word_change_ratio * c_txt.count(" "))
        if printing:
            print("# of word swaps: ", target_n_words)

        l_shift = 0.0
        if len(removed_words) > 0 and len(added_words) > 0:

            added_words_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in added_words]
            removed_words_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in removed_words]
            
            removed_words_zipfs = sorted(removed_words_zipfs, key=lambda x: x['zipf'])[:target_n_words]
            added_words_zipfs = sorted(added_words_zipfs, key=lambda x: x['zipf'])[:min(target_n_words, len(removed_words_zipfs))]

            removed_avg_zipfs = np.mean([x['zipf'] for x in removed_words_zipfs])
            added_avg_zipfs = np.mean([x['zipf'] for x in added_words_zipfs])
            l_shift = added_avg_zipfs - removed_avg_zipfs

            if printing:
                print("[Avg Zipf: %.3f] Added words:" % (added_avg_zipfs), added_words_zipfs)
                print("[Avg Zipf: %.3f] Removed words:" % (removed_avg_zipfs), removed_words_zipfs)
                print("lexical shift: %.3f" % (l_shift))

        return l_shift
    

    def score(self, c_texts, s_texts, printing=False):
        self.c_texts = c_texts
        self.s_texts = s_texts
        self.scores = [0] * len(c_texts)
        
        scores = []
        for idx in range(len(c_texts)):
            lex_shift = self.lexical_shift(c_texts[idx], s_texts[idx], printing=printing)   
            score = shift_to_score(lex_shift, self.target, 0.25)
            scores.append(score)
        
        scores = clamp(Tensor(scores), 0.001, 1.0).tolist()
        return scores 


        
FRE = "FRE"
WSTF = "WSTF"

class SyntacticSimplicity:

    def __init__(self, scoring_func=FRE, target=60):
        textstat.set_lang('de')
        self.target = target
        self.scoring_func = scoring_func

    def readability_score(self, txt):
        if self.scoring_func == FRE:
            score = textstat.flesch_reading_ease(txt)
        elif self.scoring_func == WSTF:
            score = textstat.wiener_sachtextformel(txt, 4)
        return score

    def score(self, c_texts, s_texts, printing=False):
        scores = []
        
        for source, generated in zip(c_texts, s_texts):
            if len(generated) <= 2:
                scores.append(0)
            else:
                rsource = self.readability_score(source)
                rtarget = self.readability_score(generated)
                t_shift = self.target - rsource
                if t_shift <= 0:
                    t_shift = 0.01
                rshift = rtarget - rsource
                
                score = shift_to_score(rshift, t_shift, 0.3)
                scores.append(score)

        scores = clamp(Tensor(scores), 0.001, 1.0).tolist()
        return scores


if __name__ == '__main__':
    None