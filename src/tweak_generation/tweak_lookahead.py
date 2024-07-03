# coding=utf-8

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

def _generate_greedy(
        model,
        input_ids,
        cur_len,
        look_ahead_step,
        attention_mask,
        position_ids,
        model_kwargs,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """

    for t in range(look_ahead_step):
        
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = model(**model_inputs)

        next_token_logits = outputs[0][:, -1, :]

        log_prob = F.log_softmax(next_token_logits, dim=-1)

        scores = postprocess_next_token_scores(
            model=model,
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
            pad_token_id=pad_token_id,
        )

        # if model has past, then set the past variable to speed up decoding
        if model._use_cache(outputs, use_cache):
            past = outputs[1]

        # Greedy decoding
        next_token = torch.argmax(scores, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        scores_to_add = torch.gather(log_prob, 1, tokens_to_add[:, None]).squeeze(1)
        scores_to_add *= (tokens_to_add != pad_token_id).float()
        sent_scores += scores_to_add

        # # compute the score for phrases to appear in the future
        # for j, (phrase, phrase_mask) in enumerate(phrases_idx_mask):
        #     if torch.sum(phrase_mask[:, t]).item():
        #         phrase_score = log_prob[:, phrase[0]]

        #         if len(phrase) > 1:
        #             phrase_input_ids = input_ids.new(phrase)[None, :].expand(batch_size, -1)
        #             phrase_position_ids = torch.cat([position_ids[:, -1] + 1 + i for i in range(len(phrase))], dim=-1)
        #             phrase_attention_mask = torch.cat(
        #                 [attention_mask] + [attention_mask.new_ones((attention_mask.shape[0], 1))
        #                                     for _ in range(len(phrase))], dim=-1
        #             )

        #             follow_logits = model(input_ids=phrase_input_ids, past=past,
        #                                  attention_mask=phrase_attention_mask, position_ids=phrase_position_ids,
        #                                  labels=phrase_input_ids, use_cache=use_cache)[1]
        #             follow_log_prob = F.log_softmax(follow_logits, dim=-1)

        #             phrase_score = phrase_score[:, None]
        #             for i in range(len(phrase[:-1])):
        #                 phrase_score = torch.cat([phrase_score, follow_log_prob[:, i, phrase[i + 1]][:, None]], dim=-1)
        #             phrase_score = torch.mean(phrase_score, dim=-1)

        #         phrase_score.masked_fill_(phrase_mask[:, t] == 0, -float("inf"))
        #         phrase_score.masked_fill_(unfinished_sents == 0, -float("inf"))
        #         # look_ahead_scores[j, :, t] = phrase_score

        # # update word embedding for current tokens_to_add
        # if fusion_t is not None:
        #     scores = scores / fusion_t
        #     probs = F.softmax(scores, dim=-1)
        #     word_embeds = torch.matmul(probs, model.get_input_embeddings().weight)

        #     if eos_token_id is not None:
        #         pad_one_hot = F.one_hot(next_token.new(next_token.shape).fill_(pad_token_id), num_classes=vocab_size)
        #         pad_embed = torch.matmul(pad_one_hot.float(), model.get_input_embeddings().weight)
        #         word_embeds = word_embeds * unfinished_sents[:, None] + pad_embed * (1 - unfinished_sents[:, None])

        # # add token and increase length by one
        # input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        # position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
        # cur_len = cur_len + 1

        # if eos_token_id is not None:
        #     eos_in_sents = tokens_to_add == eos_token_id
        #     # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
        #     is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
        #     sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
        #     # unfinished_sents is set to zero if eos in sentence
        #     unfinished_sents.mul_((~eos_in_sents).long())

        # # stop when there is a </s> in each sentence, or if we exceed the maximul length
        # if unfinished_sents.max() == 0:
        #     break

        # # extend attention_mask for new generated input if only decoder
        # if model.config.is_encoder_decoder is False:
        #     attention_mask = torch.cat(
        #         [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        #     )

    # return states
    # look_ahead_scores = torch.max(torch.max(look_ahead_scores, dim=2)[0], dim=0)[0]
    # return look_ahead_scores.tolist()
