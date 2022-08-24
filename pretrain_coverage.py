from transformers.optimization import AdamW

import torch, time, numpy as np, argparse, os, wandb
import pandas as pd
from reward.meaning_preservation import CoverageModel
from misc.utils_masking import string2mask
from misc.utils_gpu import get_device

"""
The code is from The Summary Loop https://github.com/CannyLab/summary_loop
"""

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")
parser.add_argument("--kw_ratio", type=int, default=10, help="Top n words (tf-idf wise) will be masked in the coverage model.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch Size.")

args = parser.parse_args()

device = get_device()

#masking_model = string2mask(masker_name="kw%i"%args.kw_ratio)
masking_model = string2mask(masker_name="nostop")
cov = CoverageModel(masking_model, device=get_device())
cov.model.train()
print("Loaded model")

param_optimizer = list(cov.model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
scaler = torch.cuda.amp.GradScaler()

mlsum_df = pd.read_csv("./data/proxy_mlsumde.csv")
mlsum_df = mlsum_df.sample(frac = 1) # shuffle dataset
N = len(mlsum_df)
d_train = mlsum_df[0:int(N*0.9)]
d_dev = mlsum_df[int(N*0.9):]

split_size = len(d_train) / args.batch_size

time_save = time.time()
optim_every = 5
eval_every = 50

wandb.init(project="Coverage_Training")
wandb.run.name = args.experiment
wandb.config.update(args)
wandb.run.save()

eval_loss_min = 10000000

for n_epoch in range(5):
    b_idx = 0
    for batch in np.array_split(d_train, split_size):
        b_idx += 1
        print("Training Batch #", b_idx)

        texts = batch['text'].to_list()
        summaries = batch['summary'].to_list()

        with torch.autocast("cuda"):
            loss, acc = cov.train_batch(texts, summaries)
        
        scaler.scale(loss).backward()
        
        if b_idx%optim_every == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        wandb.log({'train_loss': loss, 'train_acc': acc})
        
        torch.cuda.empty_cache()

        if b_idx%eval_every == 0:
            eval_split_size = 10
            d_dev = d_dev.sample(frac = 1)
            with torch.no_grad():
                e_idx = 0
                losses = 0
                for i in range(5):
                    e_idx += 1
                    
                    eval_texts = d_dev['text'][0:eval_split_size].to_list()
                    eval_summaries = d_dev['summary'][0:eval_split_size].to_list()
                    
                    with torch.autocast("cuda"):
                        loss, acc = cov.train_batch(eval_texts, eval_summaries)
                        
                    losses += loss
                losses /= e_idx
                print("Eval Loss: %.3f" % (losses))
                scores = cov.score(eval_texts, eval_summaries)['scores']
                print("Scores: ", scores)

            if losses < eval_loss_min:
                eval_loss_min = losses
                model_output_file = os.path.join("E:/models/", "cov_model_%s.bin" % args.experiment)
                cov.save_model(model_output_file)

            wandb.log({'eval_loss': loss, 'eval_acc': acc, 'score(hard)': np.mean(scores)})
            
        
        