#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import evaluate
from evaluate import load
import numpy as np
import sys
import json

# refernece_path = sys.argv[1]
output_path = sys.argv[1]

# datasets/webnlg/webnlg-test_both-eval.csv
dataset_path = sys.argv[2]

dataset =sys.argv[3]

print(dataset)

result_dict={}

if dataset == "webnlg":
    references = [d.strip().split('|') for d in pd.read_csv(dataset_path, sep="\t", encoding="utf-8")['target'].values]
elif dataset == "eventnarrative":
    references = [[d.strip()] for d in pd.read_csv(dataset_path, sep="\t")['target'].values]
elif dataset == "tekgen":
    references = [[d.strip()] for d in pd.read_csv(dataset_path, sep="\t")['target'].values]
elif dataset == "genwiki":
    references = [[d.strip()] for d in pd.read_csv(dataset_path, sep="\t")['target'].values]

with open(output_path+'/generated_predictions.txt', "r", encoding="utf-8") as f:
    predictions = [d.strip() for d in f.readlines()]

print(len(references) , len(predictions))

assert len(references) == len(predictions)

print(references[:5])
print(predictions[:5])

bleu = evaluate.load("sacrebleu")
results = bleu.compute(predictions=predictions, references=references)
print(results)
result_dict['sacrebleu'] = results['score']

meteor = evaluate.load("meteor")
if dataset == "webnlg":
    results = meteor.compute(predictions=predictions, references=[r[0] for r in references])
else:
    results = meteor.compute(predictions=predictions, references=[r for r in references])
result_dict['METEOR'] = results['meteor']

meteor = evaluate.load("chrf")
if dataset == "webnlg":
    results = meteor.compute(predictions=predictions, references=[r[0] for r in references])
else:
    results = meteor.compute(predictions=predictions, references=[r for r in references])
result_dict['chrF++'] = results['score']

bertscore = load("bertscore")
if dataset == "webnlg":
    results = bertscore.compute(predictions=predictions, references=[r[0] for r in references], lang="en", model_type="roberta-large")
else:
    results = bertscore.compute(predictions=predictions, references=[r for r in references], lang="en", model_type="roberta-large")
result_dict['BertScore'] = np.mean(results['f1'])

print(result_dict)

with open(output_path+'/our_testing_results.json', 'w+') as fp:
    json.dump(result_dict, fp, indent=4)

# ### Evaluation for unseen and seen BLEU

# references = [d.strip().split('|') for d in pd.read_csv("datasets/webnlg/webnlg-test_seen-eval.csv", sep="\t")['target'].values]
# with open(output_path+"/seen_decoding_output/generated_predictions.txt", "r") as f:
#     predictions = [d.strip() for d in f.readlines()]
# assert len(references) == len(predictions)
# bleu = evaluate.load("bleu")
# results = bleu.compute(predictions=predictions, references=references)
# result_dict['SEEN_BLEU'] = results['bleu']

# references = [d.strip().split('|') for d in pd.read_csv("datasets/webnlg/webnlg-test_unseen-eval.csv", sep="\t")['target'].values]
# with open(output_path+"/unseen_decoding_output/generated_predictions.txt", "r") as f:
#     predictions = [d.strip() for d in f.readlines()]
# assert len(references) == len(predictions)
# bleu = evaluate.load("bleu")
# results = bleu.compute(predictions=predictions, references=references)
# result_dict['UNSEEN_BLEU'] = results['bleu']
