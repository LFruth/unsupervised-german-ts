import numpy as np
import torch

"""
The code is adapted from Keep it Simple https://github.com/tingofurro/keep_it_simple/
"""

class Scorer:

    def __init__(self, model, name="", weight=1, binary=False):
        self.model = model
        self.weight = weight
        self.binary = binary
        self.name = name


class RewardWrapper:
 
    def __init__(self, scorers):
        self.scorers = scorers

    def sum_score(self, sources, generateds, raw_score=False):
        scores = np.zeros(len(sources))
        raw_scores = []
        
        for scorer in self.scorers:
            weight = scorer.weight
            score = scorer.model.score(sources, generateds)
            weighted_scores = [s*weight for s in score]
            if not raw_score:
                scores = np.add(scores, weighted_scores)
            else:
                raw_scores.append(weighted_scores)
        if raw_score:
            return raw_scores
        return scores

    def logsum_score(self, sources, generateds, printing=False, print_candidates=False):
        scores = np.zeros((len(sources)))
        raw_scores = {}

        for scorer in self.scorers:
            score = scorer.model.score(sources, generateds, printing=printing)
            weight = scorer.weight
            raw_scores[scorer.name] = score
            score = np.clip(score, 0.0001, 0.9999)
            if scorer.binary == False:
                scores += weight*np.log(np.array(score))
            else: # It's a binary penalty
                scores += weight*np.log(np.array(score))
        
        scores = torch.tensor(scores)

        return (scores, raw_scores) 

    def print_candidates(self, s, g, scores, raw_scores):
        for idx in range(len(g)):
            print("Source: \n\t%s\n" % s[idx])
            print("Generated: \n\t%s\n" % g[idx])
            print("Scores: \n")
            for key, value in raw_scores.items():
                print("\t -%s: %.3f" % (key, value[idx]))
            print("Score: %.3f \n\n\n" % scores[idx])

    def product_score(self, sources, generateds, printing=False, print_candidates=False):
        scores = np.ones(len(sources))
        raw_scores = {}

        for scorer in self.scorers:
            s = scorer.model.score(sources, generateds, printing=printing)
            scores = np.multiply(scores, s)
            raw_scores[scorer.name] = s

        scores = torch.tensor(scores)

        if print_candidates:
            self.print_candidates(sources, generateds, scores, raw_scores)
        
        return (scores, raw_scores)




