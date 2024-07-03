import torch
import torch.nn.functional as F
import json
import ast

import re

import itertools

from collections import defaultdict

def get_hypo_label_mask(triples, max_num_classes=3):

    triples = ast.literal_eval(triples)

    # hypo_label = torch.tensor([int(d) for d in ])

    # Generating the mask for the one_hot label
    mask = torch.zeros(max_num_classes, max_num_classes).tolist()

    num_hypo = 2
    for i in range(max_num_classes):
        for j in range(max_num_classes):
            if i < num_hypo and j < len(triples):
                mask[i][j] = [1, 1] # one hot enc for faithful [0, 1]
            else:
                mask[i][j] = [0, 0] # one hot enc for faithful [0, 1]

    return mask

def get_hypo_label_mask_inference(str_triples, max_num_classes=3):

    
    len_triples = len([m.start() for m in re.finditer('<H>', str_triples)])

    # Generating the mask for the one_hot label
    mask = torch.zeros(max_num_classes, max_num_classes).tolist()

    num_hypo = 2
    for i in range(max_num_classes):
        for j in range(max_num_classes):
            if i < num_hypo and j < len_triples:
                mask[i][j] = [1, 1] # one hot enc for faithful [0, 1]
            else:
                mask[i][j] = [0, 0] # one hot enc for faithful [0, 1]

    return mask, len_triples

def label_parsing(hypo_label):
    hypo_label = ast.literal_eval(hypo_label)
    hypo_label = torch.tensor(hypo_label)
    return hypo_label

def get_hypo_label(hypo_label, max_num_classes):
    # Generating one_hot label for Y
    # Input: 
    # hypo_label: list of idx for contradicting triples: [1]
    # Output:
    # hypo_label: binary form of label in max_num_label: [1, 0, 1]

    hypo_label = F.one_hot(hypo_label, num_classes=max_num_classes) # automatic padding with 0
    hypo_label = torch.sum(hypo_label, dim=0)
    hypo_label = (hypo_label == 0).int().tolist() # convert hallu to 0 and faithful to 1
    
    return hypo_label

def hypo_label_encoding(triples, b_hypo_label, f_hypo_label, max_num_classes=3):

    triples = ast.literal_eval(triples)
    b_hypo_label = label_parsing(b_hypo_label)
    f_hypo_label = label_parsing(f_hypo_label)

    hypo_label = []
    if b_hypo_label == torch.tensor([-1]):
        b_hypo_label = torch.ones(max_num_classes).tolist()
    else:
        b_hypo_label = get_hypo_label(b_hypo_label, max_num_classes)
    
    if f_hypo_label == torch.tensor([-1]):
        f_hypo_label = torch.ones(max_num_classes).tolist()
    else:
        f_hypo_label = get_hypo_label(f_hypo_label, max_num_classes)

    padding_label = [[1 for i in range(max_num_classes)] for j in range(max_num_classes - 2)] # Padding labels vector into matrix form (N_max_label, N_max_label) 

    hypo_label = [b_hypo_label, f_hypo_label] + padding_label

    for i, d in enumerate(hypo_label):
        for j, z in enumerate(d):
            if int(hypo_label[i][j]) == 1:
                hypo_label[i][j] = [0, 1]
            elif int(hypo_label[i][j]) == 0:
                hypo_label[i][j] = [1, 0]

    return triples, hypo_label

# # Test case
# print(hypo_label_encoding("[['Aarhus Airport', 'city Served', 'Aarhus Denmark'], ['Aarhus Airport', 'operating Organisation', 'Aarhus Lufthavn A/S']]", "[-1]", 3))
# print(hypo_label_encoding("[['Aarhus Airport', 'city Served', 'Aarhus Denmark'], ['Aarhus Airport', 'operating Organisation', 'Aarhus Lufthavn A/S']]", "[0]", 3))
# print(hypo_label_encoding("[['Aarhus Airport', 'city Served', 'Aarhus Denmark'], ['Aarhus Airport', 'operating Organisation', 'Aarhus Lufthavn A/S']]", "[0,1]", 3))


def prepro_hvm_batch_inputs_old(batch_inputs, args, device):
    sources = []
    for d in batch_inputs:
        sources.append([d for i in range(args.num_beams * 2)])
    sources = [j for sub in sources for j in sub]

    score_matrix_mask = []
    num_triples = []
    for triples in sources:
        mask, len_triples = get_hypo_label_mask_inference(triples, max_num_classes=args.max_num_triples)
        score_matrix_mask.append(mask)
        num_triples.append([len_triples])
    num_triples = torch.tensor(num_triples).to(device)

    return sources, score_matrix_mask, num_triples

def prepro_hvm_batch_inputs(batch_inputs, args, device):
    sources = []
    for d in batch_inputs:
        sources.append([d for i in range(args.num_beams * 2)])
    sources = [j for sub in sources for j in sub]

    score_matrix_mask = []
    num_triples = []
    
    for triples in sources:
        len_triples = len([m.start() for m in re.finditer('<H>', triples)])
        num_triples.append(len_triples)
    
    max_num_triples = max(num_triples)

    for triples in sources:
        mask, len_triples = get_hypo_label_mask_inference(triples, max_num_classes=max_num_triples)
        score_matrix_mask.append(mask)
        
    num_triples = torch.tensor(num_triples).to(device)

    return sources, score_matrix_mask, num_triples

