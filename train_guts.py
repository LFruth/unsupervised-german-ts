import argparse
import torch
import pandas as pd
import wandb
import numpy as np
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to ignore a warning...

from transformers.optimization import AdamW

from generator import Generator
from misc.utils_text import LemmaModel, select_logprobs
from bert_score import BERTScorer 
from reward.simplicity import LexicalSimplicity, SyntacticSimplicity
from reward.fluency import Fluency, TextDiscriminator
from reward.meaning_preservation import TextSimilarity
from reward.guardrails import Brevity, HallucinationModel, RepeatNGramPenalty, RepeatArticles
from reward.reward import RewardWrapper, Scorer
from misc.utils_gpu import get_device, MemoryLogger
from misc.utils_checkpointing import print_candidates, ModelCheckpoint, TickTimer

"""
The code is from Keep it Simple https://github.com/tingofurro/keep_it_simple/
"""

parser = argparse.ArgumentParser()

parser.add_argument("--experiment", type=str, required=True, help="Experiment name. Will be used to save a model file and a log file.")

parser.add_argument("--model_path", type=str, default="./models/gpt2_copy_finetune.bin", help="Path to a pretrained model.")
parser.add_argument("--k", type=int, default=8, help="The k of k-SCST. Number of samples generated from a source text.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
parser.add_argument("--max_seq_length", type=int, default=265, help="Maximum output length. Saves time if the sequences are short.")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--memlog", type=int, default=0, help="Show some basic GPU memory allocation logs.")
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--top_k", type=int, default=5)
# CKPT Related
parser.add_argument("--ckpt_every", type=int, default=600, help="If 0, checkpointing is not used. Otherwise, checkpointing is done very x seconds.")
parser.add_argument("--ckpt_lookback", type=int, default=100, help="When checkpointing, will consider the avg total score of the last x samples.")
parser.add_argument("--print_every", type=int, default=150, help="Save the model and print out an example every x seconds.")
parser.add_argument("--timings", action='store_true', help="Whether to print out timings for each pass.")

args = parser.parse_args()

learning_rate = args.learning_rate
batch_size = 1
k = args.k
max_seq_length = args.max_seq_length
print_mlog = bool(args.memlog)
model_path = args.model_path
temperature = args.temperature
top_p = args.top_p
top_k = args.top_k

wandb.init(project="Train GUTS")
wandb.run.name = args.experiment
wandb.config.update(args)
wandb.run.save()

mlog = MemoryLogger(printing=print_mlog)

# init shared models
lemma_model = LemmaModel(model_path='./models/morphmodel_ger.pgz')
bert_scorer = BERTScorer("dbmdz/bert-base-german-cased", num_layers=5, use_fast_tokenizer=True, device=get_device(gpu_idx=1))

# load scorers
scorers = [Scorer(LexicalSimplicity(lemma_model), name="Lexical Simplicity", weight=0.5),
    Scorer(SyntacticSimplicity(), name="Syntactic Simplicity", weight=2.0),
    Scorer(TextSimilarity(bert_scorer), name="Meaning Preservation", weight=3.0),
    Scorer(Fluency(model_file="./models/wiki_finetune.bin", device_idx=0), name="Fluency", weight=0.5), 
    Scorer(TextDiscriminator(fp16=True), name="Discriminator Fluency", weight=0.5),
    Scorer(Brevity(min_ratio=0.6, max_ratio=1.3), name="Brevity", binary=True, weight=1.0), 
    Scorer(HallucinationModel(lemma_model, bert_scorer, tagger_model_card="Davlan/distilbert-base-multilingual-cased-ner-hrl", bscore_threshold=0.74), name="Hallucinate Facts", binary=True, weight=1.0),
    Scorer(RepeatNGramPenalty(), name='N-Gram Penalty', weight=1.0),
    Scorer(RepeatArticles(), name="Repeat Article N-Gram", weight=1.0)
    ]

reward = RewardWrapper(scorers)         

# load generator
generator = Generator(max_input_length=180, max_output_length=250)
if len(model_path) > 0:
    generator.reload(model_path)

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
wiki_paragraphs = pd.read_csv("./data/all_wiki_paragraphs.csv")
wiki_paragraphs = wiki_paragraphs.sample(frac = 1) # shuffle

checkpointer = ModelCheckpoint(generator, args.ckpt_every, args.ckpt_lookback, "./models/%s.bin" % args.experiment)
timer = TickTimer()

device = get_device()

T_start, T_last_best = time.time(), time.time()

#Training Loop
for epoch_i in range(10):
    mlog.start_id("loop")
    
    batch_idx = 0

    for idx, row in wiki_paragraphs.iterrows():
        print("...............................................")
        
        mlog.end_id("loop") 
        mlog.start_id("loop") 
        text = row['paragraph']
        
        batch_idx += 1

        # Sample, Score and compute loss
        with torch.autocast("cuda"):
            mlog.start_id("sample")
            
            outputs = generator.generate_torch_samples(text, max_output_len=230, sample_size=k, top_k=top_k, top_p=top_p, no_repeat_ngram=5, temperature=temperature)
            #outputs = generator.typical_sampling(text, sample_size=k, max_output_len=230, p=top_p, no_repeat_ngram=5, temperature=temperature)
            mlog.end_id("sample")

            mlog.start_id("Scoring")
            source = [text] * k
            generateds = [o['output_text'] for o in outputs]
            generateds_tokenized = [o['output_tokens'] for o in outputs]

            score_returns = reward.logsum_score(source, generateds, print_candidates=False)
            scores = score_returns[0]

            batch_scores = scores.reshape(1, k)
            mean_scores = batch_scores.mean(dim=1)
            max_scores = torch.max(batch_scores, dim=1).values # For logging
            unlooped_mean_scores = torch.repeat_interleave(mean_scores, k)

            normalized_rewards = (unlooped_mean_scores - scores).to(device)
            n_diff_pos, n_diff_neg = (normalized_rewards<-0.02).long().sum().item(), (normalized_rewards>0.02).long().sum().item()
            mlog.end_id("Scoring")
            #print("[%d samples] %d above avg and %d below avg" % (1*k, n_diff_pos, n_diff_neg))

            mlog.start_id("loss")
            logits = generator.train_batch(source, decoded_tokenized=generateds_tokenized, return_logits=True)
            selected_logprobs = select_logprobs(logits, generateds_tokenized, generator.end_id, generator.device)

            loss = torch.mean(normalized_rewards * selected_logprobs)
            mlog.end_id("loss")

        mlog.start_id("optim")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        mlog.end_id("optim")
        print("Batch %i (Epoch %i)" % (batch_idx, epoch_i))
        print("Scores: ", scores)
        print("Loss: ", loss)

        # log weights and biases
        log_obj = {"loss": loss, "max_scores": max_scores, "mean_scores": mean_scores}
        log_obj.update({k: np.mean(v) for k, v in score_returns[1].items()})
        
        wandb.log(log_obj)

        if args.timings:
            timer.report()

        # Run the Checkpoint engine
        current_score = mean_scores
        is_best = checkpointer.tick(current_score)
        if is_best: # Run the inspection dataset through
            T_last_best = time.time()

        if batch_idx % args.print_every == 0:
            print_candidates(source, generateds, score_returns[1], score_returns[0])

        torch.cuda.empty_cache()


