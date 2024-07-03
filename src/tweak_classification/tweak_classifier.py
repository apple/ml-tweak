import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset
import torch
import pandas as pd

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AutoModel,
)

from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from collections import defaultdict

class BilinearMatrixAttention(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function.  This function has
    a matrix of weights ``W`` and a bias ``b``, and the similarity between the two matrices ``X``
    and ``Y`` is computed as ``X W Y^T + b``.
    Parameters
    ----------
    matrix_1_dim : ``int``
        The dimension of the matrix ``X``, described above.  This is ``X.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : ``int``
        The dimension of the matrix ``Y``, described above.  This is ``Y.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``X W Y^T + b`` calculation.  Default is no
        activation.
    use_input_biases : ``bool``, optional (default = False)
        If True, we add biases to the inputs such that the final computation
        is equivalent to the original bilinear matrix multiplication plus a
        projection of both inputs.
    label_dim : ``int``, optional (default = 1)
        The number of output classes. Typically in an attention setting this will be one,
        but this parameter allows this class to function as an equivalent to ``torch.nn.Bilinear``
        for matrices, rather than vectors.
    """
    def __init__(self,
                 matrix_1_dim: int,
                 matrix_2_dim: int,
                 use_input_biases: bool = False,
                 label_dim: int = 1) -> None:
        super(BilinearMatrixAttention, self).__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if label_dim == 1:
            self._weight_matrix = Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = Parameter(torch.Tensor(label_dim, matrix_1_dim, matrix_2_dim))

        self._bias = Parameter(torch.Tensor(1))
        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:

        if self._use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], -1)
            matrix_2 = torch.cat([matrix_2, bias2], -1)

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        return final.squeeze(1)

class TweakClassifier(nn.Module):
  def __init__(self, base_model, single_special_tok_representation, num_labels, model_config, train_with_sequence_loss=False): 
    super(TweakClassifier, self).__init__() 
    self.num_labels = num_labels

    #Load Model with given base_model and extract its body
    self.base_model = AutoModel.from_pretrained(
                                base_model,
                                config=model_config
                        )

    self.train_with_sequence_loss = train_with_sequence_loss

    if self.train_with_sequence_loss:
        self.sequence_classifier = RobertaClassificationHead(model_config)
        self.seq_loss_weight = 0.9

    self.mlp_dropout = 0.3
    self.mlp_hidden_size = 1024

    self.single_special_tok_representation = single_special_tok_representation
    
    # self.U = torch.nn.Bilinear(self.mlp_hidden_size, self.mlp_hidden_size, self.mlp_hidden_size, bias=True)
    if self.train_with_sequence_loss:
        self.element_classifier = BilinearMatrixAttention(self.mlp_hidden_size, self.mlp_hidden_size, use_input_biases=False, label_dim=2)
    else:
        self.U = BilinearMatrixAttention(self.mlp_hidden_size, self.mlp_hidden_size, use_input_biases=False, label_dim=2)

    logit_dropout = 0.2
    if logit_dropout > 0:
            self.logit_dropout = nn.Dropout(p=logit_dropout)
    else:
        self.logit_dropout = lambda x: x

    self.element_loss = nn.CrossEntropyLoss(reduction='mean')
    if self.train_with_sequence_loss:
        self.sequence_loss = nn.CrossEntropyLoss(reduction='mean')


  def forward(self, **inputs):
    
    hypo_label_matrix = inputs['hypo_label_matrix']
    score_matrix_mask = inputs['score_matrix_mask']
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if self.train_with_sequence_loss:
        if 'labels' not in inputs.keys():
            inputs['labels'] = torch.ones(input_ids.shape[0]).long().cuda()
        seq_lables = inputs['labels']

    max_num_classes = score_matrix_mask.shape[1]

    # Extract outputs from the base model
    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract embedding for all special tokens

    results = {}
    last_layer_outputs = outputs.last_hidden_state

    # Sequence-level prediction
    if self.train_with_sequence_loss:
        sequence_logits = self.sequence_classifier(last_layer_outputs)
        seq_loss = self.sequence_loss(sequence_logits, seq_lables)

    # Get the index for each special ids
    special_ids_batch = []

    if self.single_special_tok_representation:
        special_ids_list = [50267, 50265, 50266] # 65, 66: B, F; 67: <H>, specific for roberta-based hvm, may need to update later
    else:
        special_ids_list = [50267, 50265, 50266, 2] # 65, 66: B, F; 67: <H>

    for i, ids in enumerate(input_ids):
      each_ids_idx = []
      for s_id in special_ids_list:
        each_ids_idx = each_ids_idx + (ids == s_id).nonzero(as_tuple=True)[0].tolist()
      special_ids_batch.append(each_ids_idx)

    actual_num_triples = []
    # Get the embedding for each speical ids block
    triple_embs = []
    hypo_embs = []
    
    if self.single_special_tok_representation:
        for b_idx, special_ids in enumerate(special_ids_batch):
            buffer_b = []
            
            special_ids = torch.tensor(special_ids).to(self.base_model.device) # without pooling we do not need to consider </s>
            target_embs = torch.index_select(last_layer_outputs[b_idx], 0, special_ids)
            triple_reps = target_embs[:-2] # last two embeddings are b_hypo and f_hypo
            hypo_reps = target_embs[-2:] # Others are triples embedding

            actual_num_triples.append(len(triple_reps))

            # padding buffer for all triple vecs            
            triple_reps = F.pad(triple_reps, (0, 0, 0, max_num_classes - triple_reps.size(0)))
            hypo_reps = F.pad(hypo_reps, (0, 0, 0, max_num_classes - hypo_reps.size(0)))

            triple_embs.append(triple_reps)
            hypo_embs.append(hypo_reps)
    else:
        for b_idx, special_ids in enumerate(special_ids_batch):
            buffer_b = []
            for i in range(len(special_ids) - 1):
                indices = torch.arange(special_ids[i], special_ids[i+1]).to(self.base_model.device)
                target_embs = torch.index_select(last_layer_outputs[b_idx], 0, indices)
                
                # target_emb = torch.max(target_embs, dim=0).values
                target_emb = torch.mean(target_embs, dim=0)
                buffer_b.append(target_emb)

            triple_reps = buffer_b[:-2] # last two embeddings are b_hypo and f_hypo
            hypo_reps = buffer_b[-2:] # Others are triples embedding
            
            actual_num_triples.append(len(triple_reps))

            # padding buffer for all triple vecs
            for i in range(max_num_classes - len(triple_reps)):
                triple_reps.append(torch.zeros(triple_reps[0].shape).to(self.base_model.device))
            for i in range(max_num_classes - len(hypo_reps)):
                hypo_reps.append(torch.zeros(hypo_reps[0].shape).to(self.base_model.device))
            
            triple_embs.append(torch.stack(triple_reps, dim=0))
            hypo_embs.append(torch.stack(hypo_reps, dim=0))
      
    triple_embs = torch.stack(triple_embs, dim=0) # bs, max_triples, h
    hypo_embs = torch.stack(hypo_embs, dim=0) # bs, max_triples, h

    if self.train_with_sequence_loss:
        matrix = self.element_classifier(hypo_embs, triple_embs).permute(0, 2, 3, 1) # (bs, num_labels, seq_length, seq_length) -> (bs, seq_length, seq_length, num_labels)
    else:
        matrix = self.U(hypo_embs, triple_embs).permute(0, 2, 3, 1) # (bs, num_labels, seq_length, seq_length) -> (bs, seq_length, seq_length, num_labels)

    pred_probs_matrix = torch.softmax(matrix, dim=-1) * score_matrix_mask

    score_matrix_mask = score_matrix_mask.long()
    
    if hypo_label_matrix != None:
        scores = torch.masked_select(matrix, score_matrix_mask.bool()).view(-1,2)
        labels = torch.masked_select(hypo_label_matrix.float(), score_matrix_mask.bool()).view(-1,2)
        if self.train_with_sequence_loss:
            results['loss'] = (1 - self.seq_loss_weight) * self.element_loss(scores, labels) + self.seq_loss_weight * seq_loss
        else:
            results['loss'] = self.element_loss(scores, labels)

    if not self.training:

        results['predictions'] = torch.argmax(pred_probs_matrix, dim=-1)
        results['predicted_2d_scores'] = matrix * score_matrix_mask

        if self.train_with_sequence_loss:
            results['sequence_predictions'] = torch.argmax(sequence_logits, dim=-1)
            results['sequence_predicted_scores'] = sequence_logits

        predictions_results = []
        for i, predictions in enumerate(results['predictions']):
            f_prediction = torch.sum(results['predictions'][i, 0, :actual_num_triples[i]] == 0)
            b_prediction = torch.sum(results['predictions'][i, 1, :actual_num_triples[i]] == 0)

            if f_prediction == torch.tensor(0) and b_prediction == torch.tensor(0):
                predictions_results.append(torch.tensor(1))
            else:
                predictions_results.append(torch.tensor(0))
        
        results['predictions'] = torch.stack(predictions_results, dim=0)

        return results

    return results
