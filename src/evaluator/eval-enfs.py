import spacy
import numpy as np
from spacy import displacy
from collections import Counter
import en_core_web_sm
import sys
from tqdm.contrib import tzip
import pandas as pd
import re
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# IMPORTANT: run it at the dir containing 'datasets' subdir!

nlp = en_core_web_sm.load()
output_path = sys.argv[1]
# datasets/webnlg/webnlg-test_both-eval.csv
dataset = sys.argv[2]

if dataset == "webnlg":
    dataset_path = "datasets/webnlg/webnlg-test_both-eval.csv"
elif dataset == "webnlg-examples":
    dataset_path = "datasets/webnlg/webnlg-test_both_examples-eval.csv"
elif dataset == "eventnarrative":
    dataset_path = "datasets/eventnarrative/eventnarrative-test.csv"
elif dataset == "tekgen":
    dataset_path = "datasets/tekgen/tekgen-test.csv"
elif dataset == "genwiki":
    dataset_path = "datasets/genwiki/genwiki-test.csv"
    
src_lines = [d+'<END>' for d in pd.read_csv(dataset_path, sep="\t")['source'].values]

# h_expression = r'<H>(.*?)<R>'
# t_expression = r'<T>(.*?)<END>'
# t2_expression = r'<T>(.*?)<H>'

h_expression = r'<H>(.*?)<R>'
r_expression = r'<R>(.*?)<T>'
t_expression = r'<T>(.*?)<H>'
tEND_expression = r'(?s:.*)<T>(.*?)<END>'
        
# a = re.findall(h_expression, src_lines[0])
# b = re.findall(t_expression, src_lines[0])
# b = re.findall(t2_expression, src_lines[0])

tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
factkb = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels = 2).to(device)


with open(output_path+'/generated_predictions.txt', "r") as f:
    tgt_lines = [d.strip() for d in f.readlines()]
    
def calculate_enfs(documents, summaries):
    assert len(documents) == len(summaries)
    res = []
    doc_no_ent_cnt = 0
    summ_no_ent_cnt = 0
    score_lines = []
    factkb_lines = []
    cnt = 0
    print("---------- Evaluating ENFS ----------")
    for document, summary in tzip(documents, summaries):
        # doc_ents = re.findall(h_expression, document) + re.findall(t_expression, document) + re.findall(t2_expression, document)
        head_ents = [d.strip() for d in re.findall(h_expression, document)]
        tail_ents = [d.strip() for d in re.findall(t_expression, document) + re.findall(tEND_expression, document)]
        doc_ents = head_ents + tail_ents

        doc_ents = [d.strip().lower() for d in doc_ents]
        # doc_ents = [(X.text, X.label_) for X in nlp(document).ents]
        summ_ents = [X.text.strip().lower() for X in nlp(summary).ents]
        if len(doc_ents) == 0:
            doc_no_ent_cnt += 1
            score_lines.append("NAN\n")
            continue
        if len(summ_ents) == 0:
            summ_no_ent_cnt += 1
            score_lines.append("NAN\n")
            continue
        cnt+=1
        hallu_ent = 0

        for summ_ent in summ_ents:
            hallucinated = True
            for doc_ent in doc_ents:
                if summ_ent in doc_ent:
                    hallucinated = False
                    break
            if hallucinated == True:
                hallu_ent+=1
        
        # print(summ_ents, doc_ents, hallu_ent)

        enfs = hallu_ent / len(summ_ents)
        res.append(enfs)
        score_lines.append(str(enfs)+'\n')

    print("---------- Evaluating FactKB ----------")
    eval_batch_size = 32    
    for i in tqdm(range(0, len(documents), eval_batch_size)):

        b_documents = documents[i:i+eval_batch_size]
        b_summareis = summaries[i:i+eval_batch_size]

        b_documents = [d.replace('<H>', '').replace('<R>', '').replace('<T>', '') for d in b_documents]

        input = [[s, d] for s,d in zip(b_summareis, b_documents)]

        tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True).to(device)
        result = torch.softmax(factkb(**tokens).logits, dim = 1).tolist()
        
        factkb_scores = [float(d[1]) for d in result]
        factkb_lines += factkb_scores

    # write scorelines
    assert len(score_lines) == len(documents)
    with open(output_path+"/enfs.score.lines", "w+") as f:
        f.writelines(score_lines)
    with open(output_path+"/factkb.score.lines", "w+") as f:
        f.writelines([str(d)+'\n' for d in factkb_lines])
    
    print("Document has no entity count = "+str(doc_no_ent_cnt)+"\n", 
                      "Summary has no entity count = "+str(summ_no_ent_cnt)+"\n", 
                      "ENFS score = "+str(np.mean(res))+'\n')

    print("FactKB score = ", str(np.mean(factkb_lines)))

    # with open(output_path+'/our_testing_results.json', 'r') as file:
    #     result_dict = json.load(file)
    
    # result_dict['ENFS%'] = np.mean(res)
    # result_dict['FactKB'] = np.mean(factkb_lines)

    # with open(output_path+'/our_testing_results.json', 'w') as fp:
    #     json.dump(result_dict, fp, indent=4)

calculate_enfs(src_lines, tgt_lines)




