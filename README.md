
# GUTS

=**G**erman **U**nsupervised **T**ext **S**implification

Code from the Master Thesis and Paper: 

*"An Approach Towards Unsupervised Text Simplification on Paragraph-Level for German Texts"*  
Author: Leon Fruth

This approach is an adaption from the paper [Keep it Simple](https://arxiv.org/abs/2107.03444) to the German language.
Large parts of the code are copied and/or adapted from the Keep it Simple repository: https://github.com/tingofurro/keep_it_simple/ 

The reward scores and training progression from the training runs of GUTS are visualized [here](https://wandb.ai/lfruth/Train%20GUTS/reports/Training-runs-of-GUTS-1-and-GUTS-2--VmlldzoxOTE1Mjc1?accessToken=9cgcoshbah9z8brqk51pwj0ovx4o284e6nd3s98jmto9jiqc7iecc9gsjm3l2sfw). The model used in this repository is named GUTS-2 in the report.


## Run GUTS 
To test the GUTS models use the script [run_guts.py](run_guts.py). 

You can use arguments to test different models, decoding methods, and input texts

Without any arguments it generates a single simplification using greedy decoding.

## Training
To train GUTS use the script [train_guts.py](train_guts.py)

Before training first pre-train a GerPT-2 model on the [*copy task*](/train_copy_task.py). This way the generator model learns to copy the original paragraph, which is a good starting point.

## Reward
All parts for the parts are in the [reward](/reward) folder:
- [reward.py](reward/reward.py): Wraps the scores and utilizes them to calculate the overall reward. Different scoring functions and weights can be used with this.
- The scores can be variable added to the reward
  - [simplicity.py](reward/simplicity.py): Contains the score for the **lexical and syntactic simplicity**.
  - [meaning_preservation.py](reward/meaning_preservation.py): contains different methods to score the meaning preservation. The **TextSimilarity** score was used for this work. The file further contains the **CoverageModel**, used in *Keep it Simple* and the *Summary Loop*, and **BScoreSimilarity** a similarity scoring method only based on BERTScore.
  - [fluency.py](reward/fluency.py): The **LM-fluency** score and the **TextDiscriminator** score are contained in this file.
  - [guardrails.py](reward/guardrails.py): Different Guardrails for Hallucination Detection, Brevity, ArticleRepetition, and NGRamRepetition can be found in this file.

Some of these scores are analysed on the reference datasets TextComplexityDE and GWW_leichtesprache and visualized using some jupyter notebooks in [notebooks](/notebooks/).
- [lexical_analysis](/notebooks/lexical_analysis.ipynb)
- [syntactic_analysis](/notebooks/syntactic_analysis.ipynb)
- [fluency_analysis](/notebooks/fluency_analysis.ipynb)
- [meaning_preservation_analysis](/notebooks/meaning_preservation_analysis.ipynb)

## Data
The [data folder](data) contains the following datasets:
- [TextComplexityDE19](data/TextComplexityDE19/) contains the raw TextComplexityDE dataset
- [textcomplexityde.csv](data/textcomplexityde.csv) is the processed TextComplexity dataset. Here all sentences from a Wikipedia article are concatenated to form a Complex-Simple aligned dataset. This dataset was used for the analysis of the reward scores.
- [leichtesprache2.csv](data/leichtesprache2.csv) are parallel articles from [GWW](https://www.gww-netz.de/de/).
- [tc_eval.csv](data/tc_eval.csv) contains the manually composed paragraphs from the TextComplexityDE dataset, and the generated simplifications used for automatic evaluation of the thesis.
- [wiki_eval.csv](data/wiki_eval.csv) contains paragraphs from Wikipedia and the generated simplifications used for automatic evaluation of the thesis.
- [all_wiki_paragraphs.csv](data/all_wiki_paragraphs.csv) contains the extracted paragraphs from Wikipedia articles used for training. The file is contained in the latest [release](https://github.com/LFruth/unsupervised-german-ts/releases/tag/1.0)

## Automatic Evaluation
The jupyter notebook, where the automatic evaluation can be reproduced is located in [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb). 

This script uses the files [tc_eval.csv](data/tc_eval.csv) and [wiki_eval.csv](data/wiki_eval.csv) to generate the automatic results.

## Models
This repository contains the following saved [models](models/):
- One trained GUTS model: GUTS.bin (contained in the [release](https://github.com/LFruth/unsupervised-german-ts/releases/tag/1.0))
- morphmodel_ger.pgz: A model used for lemmatization of German words
- wiki_finetune.bin: A saved BERT model trained on wikipedia paragraphs, for the LM-Fluency score. (contained in the [release](https://github.com/LFruth/unsupervised-german-ts/releases/tag/1.0))
- All other models used in the reward scores are retrieved from the [huggingface library](https://huggingface.co/)

## Other scripts
- [notebooks/generate_pivot_simplifications.ipynb](notebooks/generate_pivot_simplifications.ipynb) is the script to generate the simplifications using the Pivot model for the automatic evaluation. To run this the generator model from [*Keep it Simple*](https://github.com/tingofurro/keep_it_simple/) is required, as well as the [trained model](https://github.com/tingofurro/keep_it_simple/releases/tag/0.1).
- [train_copy_task.py](train_copy_task.py): The training script to pre-train a GPT-2 model to copy wikipedia articles. Copying the original text is a good starting point for simplification.
- [pretrain_coverage.py](pretrain_coverage.py): The training script to train the coverage model from https://github.com/CannyLab/summary_loop

