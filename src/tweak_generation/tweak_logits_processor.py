# coding=utf-8

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import LogitsProcessor
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

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

from tweak_generation.tweak_lookahead import _generate_greedy
import numpy as np

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)

from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from tweak_classification.tweak_classifier import TweakClassifier
from tweak_classification.tweak_utils import get_hypo_label_mask_inference

class TweakLogits(LogitsProcessor):

    def __init__(self, generator, gen_tokenizer, alpha=1.0, look_ahead_step=5, forward_tweak=False, backward_tweak=False, scoring_model="nli", lookahead_generation="greedy", bi_nli_scoring=False, dynamic_addition=False, hvm_path=None, max_num_triples=7, train_with_sequence_loss=False):
        
        self.forward_tweak = forward_tweak
        self.backward_tweak = backward_tweak

        self.model = generator.eval()
        self.look_ahead_step = look_ahead_step
        self.gen_tokenizer = gen_tokenizer

        self.lookback_len_threshold = 3 # only do lookback decoding after cur_len > 3, otherwise might be meaningless

        self.lookahead_generation = lookahead_generation
        self.bi_nli_scoring = bi_nli_scoring
        self.dynamic_addition = dynamic_addition
        self.alpha = alpha
        self.scoring_model = scoring_model
        self.max_num_triples = max_num_triples

        self.train_with_sequence_loss = train_with_sequence_loss

        if self.forward_tweak == False and self.backward_tweak == False:
            print("Tweaking decoding is not enable, beam search is used.")

        if self.scoring_model == "cosine":
            print("Initialize Cosine-based scoring model...")
            self.ranker = SentenceTransformer('sentence-transformers/nli-roberta-large')
        if self.scoring_model == "nli":
            print("Initialize NLI scoring model...")
            hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

            self.rank_tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
            self.ranker = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

        if self.scoring_model == "hvm" and hvm_path:
            
            print("Initialize TWEAK-HVM scoring model...")
            print("Loading from", hvm_path)

            if "onlyH" in hvm_path:
                self.only_head_special_token = True
                print("Only use Head Speical token as Tripe Representation!")
            else:
                print("Use Head/Relation/Tail Speical token as Tripe Representation!")
                self.only_head_special_token = False

            if train_with_sequence_loss:
                print("Use the sequence prediction head. If you want to use table score prediction head, turn the --sequence_classifier off.")
            else:
                print("Use the table prediction head as HVM scoring function.")
            
            if "singleTokRep" in hvm_path:
                print("Only use Special Token Embedding!")
                self.single_special_tok_representation = True
            else:
                print("Use Pooling Representation!")
                self.single_special_tok_representation = False

            config = AutoConfig.from_pretrained(
                hvm_path
            )
            self.rank_tokenizer = tokenizer = AutoTokenizer.from_pretrained(
                hvm_path
            )

            if self.only_head_special_token:
                new_tokens = ['<B>', '<F>', '<H>'] # 50265, 50266, 50267, 50268, 50269
            else:
                new_tokens = ['<B>', '<F>', '<H>', '<R>', '<T>'] # 50265, 50266, 50267, 50268, 50269

            new_tokens_vocab = {}
            new_tokens_vocab['additional_special_tokens'] = []
            for idx, t in enumerate(new_tokens):
                new_tokens_vocab['additional_special_tokens'].append(t)
            num_added_toks = self.rank_tokenizer.add_special_tokens(new_tokens_vocab)
            print('We have added ', num_added_toks, ' tokens')

            if self.single_special_tok_representation:
                hvm_model = TweakClassifier(base_model="roberta-large", num_labels=2, model_config=config, single_special_tok_representation=True, train_with_sequence_loss=self.train_with_sequence_loss)
            else:
                hvm_model = TweakClassifier(base_model="roberta-large", num_labels=2, model_config=config, single_special_tok_representation=False, train_with_sequence_loss=self.train_with_sequence_loss)

            # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
            # on a small vocab and want a smaller embedding size, remove this test.
            embedding_size = hvm_model.base_model.get_input_embeddings().weight.shape[0]
            if len(self.rank_tokenizer) > embedding_size:
                hvm_model.base_model.resize_token_embeddings(len(self.rank_tokenizer))
            hvm_model.load_state_dict(torch.load(hvm_path+'/pytorch_model.bin'))

            self.ranker = hvm_model
            

        # setting evaluation mode for ranker
        self.ranker = self.ranker.to(self.model.device).eval()
        
    def __call__(self, source_inputs, input_ids, next_token_scores, next_beam_indices, next_tokens):

        """
        Function Description:
        
        This function accepts generated topK next_tokens as candidates and their current scores. We do following things:
        
        1) Generate lookforward and backward hypothesis for each candidate 
        2) Scoring hypothesis according to source
        3) Return the updated score for future beam search use
        
        Function Usage:
        
        next_token_scores = tweak_processor(source_inputs, input_ids, next_token_scores, next_indices, next_tokens)

        Inputs:

        source_inputs: 

        next_token_scores: tensor([[ 0.0000e+00, -1.0000e+09, -1.0000e+09, -1.0000e+09,        -inf, -inf,        -inf,        -inf]], device='cuda:0') shape: (bs, 2*num_beam)
        next_tokens: tensor([[     0, 150804,  50268, 100536,      4,      3,      1,      2]], device='cuda:0') shape: (bs, 2*num_beam)
        next_indices: tensor([[0, 3, 1, 2, 0, 0, 0, 0]], device='cuda:0'), shape: (bs, 2*num_beam)
        next_tokens: tensor([[0, 0, 0, 0, 4, 3, 1, 2]], device='cuda:0'), shape: (bs, 2*num_beam)
        
        """

        if self.forward_tweak == False and self.backward_tweak == False:
            return next_token_scores

        # input_ids shape: 12 * cur_len
        num_batch_hypotheses = input_ids.shape[0] # batch_size * num_beam
        batch_size = len(next_tokens) # 3
        assert num_batch_hypotheses % batch_size == 0
        num_beam = int(num_batch_hypotheses / batch_size) # 4
        cur_len = input_ids.shape[-1]

        # preparing source input for ranker
        encoder_inputs = source_inputs.repeat_interleave(next_tokens.shape[-1], dim=0) # 3，35 -> 24, 35, each example's source is copid 2*num_beam=8 times
        if self.scoring_model == "nli":
            sources = self.gen_tokenizer.batch_decode(encoder_inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elif self.scoring_model == "hvm":
            sources = [s.replace('<s>', '').replace('</s>', '').replace('<pad>', '').replace('<mask>', '').replace('<unk>', '') for s in self.gen_tokenizer.batch_decode(encoder_inputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)]

        # Prepare hypothesis need to be verify (hypo: generated tokens at <t ++ candidate token at t)
        hypo_input_ids = []
        for b_id in range(batch_size): # loop each batch to get the heuristics, and flat them to be single block as greedy search input
            b_beam_indices, b_next_tokens, b_next_token_scores = next_beam_indices[b_id], next_tokens[b_id], next_token_scores[b_id]
            for i, beam_id in enumerate(b_beam_indices): # 2*4
                input_id_index = b_id * num_beam + beam_id # set offset=num_beam to index correct history
                hypo_input_ids.append(torch.cat((input_ids[input_id_index], b_next_tokens[i].unsqueeze(0))))
        hypo_input_ids = torch.cat(hypo_input_ids, dim=-1).view(batch_size * num_beam * 2, cur_len+1) 
        
        ### Producing lookahead using greedy/beam/sample method
        forward_scores = torch.zeros(next_token_scores.shape,dtype=torch.float).to(self.model.device)
        if self.forward_tweak:
            ## lookahead decoding preparation
            model_kwargs = {
                "encoder_outputs": self.model.get_encoder()(
                    encoder_inputs, return_dict=True
                )
            }
            lookahead_logits_processor = LogitsProcessorList(
                    [
                        MinLengthLogitsProcessor(3, eos_token_id=self.model.generation_config.eos_token_id),
                        NoRepeatNGramLogitsProcessor(ngram_size=3)
                    ]
                )
            lookahead_stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=self.look_ahead_step)])
            
            with torch.no_grad():
                if self.lookahead_generation == 'greedy':
                    outputs = self.model.greedy_search(input_ids=hypo_input_ids, logits_processor=lookahead_logits_processor, stopping_criteria=lookahead_stopping_criteria, max_length=self.look_ahead_step, **model_kwargs)
                elif self.lookahead_generation == 'sample':
                    lookahead_logits_warper = LogitsProcessorList(
                        [
                            TopKLogitsWarper(50),
                            TemperatureLogitsWarper(0.7),
                        ]
                    )
                    outputs = self.model.sample(input_ids=hypo_input_ids, logits_processor=lookahead_logits_processor, logits_warper=lookahead_logits_warper, stopping_criteria=lookahead_stopping_criteria, **model_kwargs)
                elif self.lookahead_generation == 'beam':
                    lookahead_beam_size = 4
                    model_kwargs['encoder_outputs'].last_hidden_state = model_kwargs['encoder_outputs'].last_hidden_state.repeat_interleave(lookahead_beam_size, dim=0) # 3，35 -> 24, 35, each example's source is copid 2*num_beam=8 times 
                    beam_scorer = BeamSearchScorer(
                        batch_size=hypo_input_ids.shape[0],
                        num_beams=lookahead_beam_size,
                        device=self.model.device,
                        max_length=self.look_ahead_step,
                        do_early_stopping=True
                    )
                    hypo_input_ids = hypo_input_ids.repeat_interleave(lookahead_beam_size, dim=0)
                    outputs = self.model.beam_search(input_ids=hypo_input_ids, beam_scorer=beam_scorer, logits_processor=lookahead_logits_processor,stopping_criteria=lookahead_stopping_criteria, **model_kwargs)
            
            # we only keep forward hypo, but we might consider use whole sequence in the future
            if self.scoring_model == "hvm":
                outputs = outputs[:,cur_len:] # only for hvm
            forward_hypos = self.gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            if self.scoring_model == 'nli':
                forward_scores = self.nli_scoring(sources, forward_hypos, num_beam, batch_size) # beam_size,
            
            # dynamic addition for f and b
            forward_hypo_len = [len(self.gen_tokenizer(d).input_ids[1:-1]) for d in forward_hypos]

        # ### Producing lookback heuristics and verifying with scorer
        back_scores = torch.zeros(next_token_scores.shape,dtype=torch.float).to(self.model.device)
        if self.backward_tweak:
            back_hypo_len = torch.ones(2 * batch_size * num_beam,dtype=torch.long) * cur_len
            tokens_hypo = hypo_input_ids
            back_hypos = self.gen_tokenizer.batch_decode(tokens_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            if cur_len > self.lookback_len_threshold:
                # usage for scorer
                if self.scoring_model == 'nli':
                    back_scores = self.nli_scoring(sources, back_hypos, num_beam, batch_size) # beam_size, 


        if self.forward_tweak and self.backward_tweak and self.scoring_model == 'hvm':
            if self.train_with_sequence_loss:
                faithful_scores = self.hvm_scoring(sources, back_hypos, forward_hypos, num_beam, batch_size) # beam_size, 
                return next_token_scores + self.alpha * faithful_scores # normalise all scores into the same scale, aggregate with hyerpara alpha (alpha should be >1)
            else:
                back_scores, forward_scores = self.hvm_scoring(sources, back_hypos, forward_hypos, num_beam, batch_size) # beam_size, 
        
        if self.forward_tweak and self.backward_tweak and self.dynamic_addition:
            forward_weights = torch.tensor([s_f / (s_f + s_b) for s_f, s_b in zip(forward_hypo_len, back_hypo_len)]).view(next_token_scores.shape).to(self.model.device)
            backward_weights = torch.tensor([s_b / (s_f + s_b) for s_f, s_b in zip(forward_hypo_len, back_hypo_len)]).view(next_token_scores.shape).to(self.model.device)

            faithful_scores = forward_scores * forward_weights + backward_weights * back_scores
        else:
            faithful_scores = forward_scores + back_scores

        next_token_scores = next_token_scores + self.alpha * faithful_scores # normalise all scores into the same scale, aggregate with hyerpara alpha (alpha should be >1)

        return next_token_scores # higher is better

    def nli_scoring(self, sources, hypos, num_beam, batch_size):

        max_length = 384

        if self.bi_nli_scoring == False:

            source_hypo_pairs = [[s,h] for s,h in zip(sources, hypos)]

            tokenized_input_seq_pair = self.rank_tokenizer.batch_encode_plus(source_hypo_pairs,
                                                            max_length=max_length,
                                                            return_token_type_ids=True, truncation=True, padding=True)

            # print(tokenized_input_seq_pair)

            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().to(self.model.device)

            # # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().to(self.model.device)
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().to(self.model.device)

            with torch.no_grad():
                outputs = self.ranker(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)

            outputs = nn.functional.log_softmax(
               outputs.logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            # return (outputs[:,0] - outputs[:,-1]).view(batch_size, num_beam * 2) # our best NLI
            return outputs[:,0].view(batch_size, num_beam * 2) # positive scoring only

        elif self.bi_nli_scoring:

            source_hypo_pairs = [[s,h] for s,h in zip(sources, hypos)]
            tokenized_input_seq_pair = self.rank_tokenizer.batch_encode_plus(source_hypo_pairs,
                                                            max_length=max_length,
                                                            return_token_type_ids=True, truncation=True, padding=True)
            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().to(self.model.device)

            # # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().to(self.model.device)
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().to(self.model.device)

            with torch.no_grad():
                source_hypo_outputs = self.ranker(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)
            
            source_hypo_scores = nn.functional.log_softmax(
               source_hypo_outputs, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            hypo_source_pairs = [[h,s] for s,h in zip(sources, hypos)]
            tokenized_input_seq_pair = self.rank_tokenizer.batch_encode_plus(hypo_source_pairs,
                                                            max_length=max_length,
                                                            return_token_type_ids=True, truncation=True, padding=True)
            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().to(self.model.device)

            # # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().to(self.model.device)
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().to(self.model.device)

            with torch.no_grad():
                hypo_source_outputs = self.ranker(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)

            # Note:
            # "id2label": {
            #     "0": "entailment",
            #     "1": "neutral",
            #     "2": "contradiction"
            # },

            hypo_source_scores = nn.functional.log_softmax(
               hypo_source_outputs, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            return ((source_hypo_scores[:,0] - source_hypo_scores[:,-1]).view(batch_size, num_beam * 2) + (hypo_source_scores[:,0] - hypo_source_scores[:,-1]).view(batch_size, num_beam * 2))/2

    def hvm_scoring(self, sources, b_hypos, f_hypos, num_beam, batch_size):
        
        score_matrix_mask = self.score_matrix_mask
        num_triples = self.num_triples
        sources = self.sources

        if self.only_head_special_token:
            sources = [s.replace('<R>', '').replace('<T>', '') for s in sources]

        if self.single_special_tok_representation:
            hvm_hypos = [s+' </s> <B> '+b+' <F> '+f for (s,b,f) in zip(sources, b_hypos, f_hypos)]
        else:
            hvm_hypos = [s+' <B> '+b+' <F> '+f for (s,b,f) in zip(sources, b_hypos, f_hypos)]

        tokenized_input_seq_pair = self.rank_tokenizer.batch_encode_plus(hvm_hypos,
                                                            max_length=512,
                                                            return_token_type_ids=True, truncation=True, padding=True)
        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().to(self.model.device)

        # # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().to(self.model.device)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().to(self.model.device)
        score_matrix_mask = torch.Tensor(score_matrix_mask).long().to(self.model.device)

        inputs = {}
        inputs['hypo_label_matrix'] = None
        inputs['score_matrix_mask'] = score_matrix_mask
        inputs['input_ids'] = input_ids
        inputs['attention_mask'] = attention_mask

        with torch.no_grad():
            outputs = self.ranker(**inputs)
        
        if self.train_with_sequence_loss:
            log_logits = outputs["sequence_predicted_scores"]
            outputs = nn.functional.log_softmax(
                log_logits, dim=-1
                )
            return outputs[:,1].view(batch_size, num_beam * 2) # positive scoring only, 1 is entailment
        else:
            log_logits = outputs["predicted_2d_scores"] * score_matrix_mask.float()  # (batch_size * num_beams, vocab_size)
            log_logits = nn.functional.log_softmax(
                log_logits, dim=-1
                ) * score_matrix_mask.float()  # (batch_size * num_beams, vocab_size)

            # predicted_scores = log_logits[:,:2,:,1] - log_logits[:,:2,:,0] # taking difference, makes log_softmax useless
            predicted_scores = log_logits[:,:2,:,1] # only using 1 scores
            
            b_scores = torch.sum(predicted_scores[:,0,:], dim=-1).view(batch_size, num_beam * 2) # bs, num_beam*2
            f_scores = torch.sum(predicted_scores[:,1,:], dim=-1).view(batch_size, num_beam * 2)
            
            num_triples = num_triples.view(batch_size, num_beam * 2)

            # normalize digit with num_pred
            b_scores = torch.div(b_scores, num_triples)
            f_scores = torch.div(f_scores, num_triples)

            return b_scores, f_scores

    def prepare_inputs(self, inputs):
        self.score_matrix_mask = inputs['hvm_hypo_label_mask']
        self.num_triples = inputs['hvm_num_triples']
        self.sources = inputs['sources']
