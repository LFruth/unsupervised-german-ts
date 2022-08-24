import argparse
import torch
import pandas as pd
import numpy as np
import os
import wandb

from transformers.optimization import AdamW

from generator import Generator

"""
The code is from The Summary Loop https://github.com/CannyLab/summary_loop
"""

# Pre-Trains the Generator on a copy-text task, for a good simplification baseline training start.

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="strong", required=True, help="Experiment name. Will be used to save a model file and a log file.")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
parser.add_argument("--batch_size", type=float, default=8)
parser.add_argument("--max_seq_length", type=int, default=265, help="Maximum output length. Saves time if the sequences are short.")
parser.add_argument("--stop_at", type=float, default=0.01)

args = parser.parse_args()
learning_rate = args.learning_rate
max_seq_length = args.max_seq_length
batch_size = args.batch_size

# load generator
generator = Generator(max_input_length=180, max_output_length=300)

# init optimizer
param_optimizer = list(generator.model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

# load data
wiki_paragraphs = pd.read_csv("./data/small_wiki_paragraphs.csv")
# shuffle
wiki_paragraphs = wiki_paragraphs.sample(frac = 1)

dataset = pd.DataFrame(columns=['source', 'target'])
dataset['source'] = wiki_paragraphs['paragraph']
dataset['target'] = wiki_paragraphs['paragraph']

N = len(dataset)
d_train = dataset[0:int(N*0.999)]
d_dev = dataset[int(N*0.999):]

split_size = len(d_train) / batch_size

wandb.init(project="Copy_Task")
wandb.run.name = args.experiment
wandb.config.update(args)
wandb.run.save()

for _ in range(5):
    b_idx = 0
    for batch in np.array_split(d_train, split_size):
        b_idx += 1
        print("Training Batch #", b_idx)

        generator.train()

        b_src = batch['source'].to_list()
        b_target = batch['target'].to_list()

        with torch.autocast("cuda"):
            loss = generator.train_batch(b_src, decoded=b_target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        wandb.log({'train_loss': loss})

        if b_idx % 10 == 0:
            eval_split_size = len(d_dev) / 20
            with torch.no_grad():
                e_idx = 0
                losses = 0
                for e_batch in np.array_split(d_dev, eval_split_size):
                    e_idx += 1
                    
                    e_src = e_batch['source'].to_list()
                    e_target = e_batch['target'].to_list()
                    
                    with torch.autocast("cuda"):
                        loss = generator.train_batch(e_src, decoded=e_target)
                    #print("Eval #%i: %.3f" % (e_idx, loss))
                    losses += loss
                losses /= e_idx
                print("Eval Loss: %.3f" % (losses))
                print("SOURCE: ", e_batch['source'].to_list()[0])
                print("GENERATED: ", generator.run(e_batch['source'].to_list()[0])[0])
            # save model
            model_output_file = os.path.join("./models/", "gpt2_copy_%s.bin" % args.experiment)
            generator.save(model_output_file)

            wandb.log({'eval_loss': loss})

            if loss <= args.stop_at:
                break

        
            
        


