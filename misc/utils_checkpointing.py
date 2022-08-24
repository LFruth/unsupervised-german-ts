import time
import numpy as np
import torch
from datetime import datetime

"""
The code is from Keep it Simple https://github.com/tingofurro/keep_it_simple/
"""

def save_candidates(name, source, candidates, raw_scores):
    file = open("./outputs/" + name + ".txt", "w", encoding='utf-8')

    file.write("SOURCE: \n")
    file.write(source)
    file.write("\n####################################################\n")
    for idx in range(len(candidates)):
        file.write(candidates[idx])
        file.write("\n")

        for k, v in raw_scores.items():
            print(k + ": %.5f" % v[idx])
        file.write("\n-------------------------------------------------------\n")

    file.close()

def print_candidates(source, candidates, raw_scores, scores):
    print("SOURCE: \n")
    print(source[0])
    print("\n####################################################\n")
    for idx in range(len(candidates)):
        print(candidates[idx])
        print("\n")

        for k, v in raw_scores.items():
            print(k + ": %.5f" % v[idx])
        print("= Score: ", scores[idx])
        print("\n-------------------------------------------------------\n")



class ModelCheckpoint:
    def __init__(self, model, ckpt_every, ckpt_lookback, ckpt_file):
        self.model = model
        self.ckpt_every = ckpt_every
        self.ckpt_lookback = ckpt_lookback
        self.best_ckpt_score = None
        self.score_history = []
        self.ckpt_file = ckpt_file
        self.time_start = time.time()
        self.time_ckpt = time.time()
        

    def tick(self, latest_score):
        self.score_history.append(latest_score)
        is_this_best = False
        if time.time() - self.time_start > 30*60 and len(self.score_history) > self.ckpt_lookback:
            # Don't do anything for the first 30 minutes
            current_score = np.mean(self.score_history[-self.ckpt_lookback:])

            if time.time()-self.time_ckpt > self.ckpt_every:
                revert_ckpt = self.best_ckpt_score is not None and current_score < min(1.15*self.best_ckpt_score, 0.85*self.best_ckpt_score) # Could be negative or positive
                print("================================== CKPT "+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" =================================")
                print("[CKPT] Previous best: %.4f vs. current: %.4f" % ((0.0 if self.best_ckpt_score is None else self.best_ckpt_score), current_score))
                print("[CKPT] Am I reverting? %s" % ("yes" if revert_ckpt else "no! BEST CKPT"))

                if revert_ckpt:
                    self.model.model.load_state_dict(torch.load(self.ckpt_file))
                self.time_ckpt = time.time()
                print("============================== END OF CKPT TIME ==============================")

            is_this_best = self.best_ckpt_score is None or current_score > self.best_ckpt_score
            if is_this_best:
                print("[CKPT] Saved new best at: %.4f" % (current_score))
                self.best_ckpt_score = current_score
                torch.save(self.model.model.state_dict(), self.ckpt_file)
        return is_this_best


class TickTimer:
    def __init__(self, start_key=None):
        self.T = time.time()
        self.time_map = {}
        self.current_key = start_key

    def tick(self, new_key):
        lapse = time.time() - self.T
        if self.current_key is not None:
            if self.current_key not in self.time_map:
                self.time_map[self.current_key] = 0
            self.time_map[self.current_key] += lapse
        
        self.current_key = new_key
        self.T = time.time()

    def reset(self):
        self.time_map = {}
        self.T = time.time()

    def report(self):
        if self.current_key is not None:
            self.tick(None)
        print("[TIMING REPORT] %s" % (" ".join("[%s: %.5f sec]" % (k, v) for k, v in self.time_map.items())))
