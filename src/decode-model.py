# coding=utf-8

import argparse
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
import torch
import copy
import pathlib

from tweak_generation.tweak_generate import generate
from tweak_generation.tweak_logits_processor import TweakLogits
from tweak_classification.tweak_utils import prepro_hvm_batch_inputs

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='HF Decoding Program')

# Files
parser.add_argument('--test_file', default="./datasets/webnlg/webnlg-test_both.csv")
parser.add_argument('--model_path', default="bart-eventnarrative-outputs")
parser.add_argument('--hvm_path', default=None)
parser.add_argument('--model_name', default="bart")
parser.add_argument('--output_path', default="./")
parser.add_argument('--gpu_id', default=-1, type=int, help="gpu_id used for parallel, if do not use parallel it is set to be -1")

# Decoding strategy
parser.add_argument('--forward_tweak', default=False, action='store_true')
parser.add_argument('--backward_tweak', default=False, action='store_true')

### Tweak config
# Scoring model
parser.add_argument('--scoring_model', default="nli")
# Lookahead simulation strategy
parser.add_argument('--lookahead_generation', default="greedy")
# Scoring direction
parser.add_argument('--bi_nli_scoring', default=False, action='store_true')
parser.add_argument('--dynamic_addition', default=False, action='store_true')
parser.add_argument('--alpha', default=0.9, type=float)

# Decoding parameters
parser.add_argument('--eval_batch_size', default=1, type=int)
parser.add_argument('--num_beams', default=5, type=int)
parser.add_argument('--max_generate_length', default=384, type=int)

parser.add_argument('--sequence_classifier', default=False, action='store_true')

args = parser.parse_args()

print("Decoding settings:",args)

# Logging
print("Loading model from:", args.model_path)
print("Loading testing file from", args.test_file)
print("Outputing to", args.output_path)

path = pathlib.Path(args.output_path)
path.mkdir(parents=True, exist_ok=True)

# Prepare decoding inputs
tokenizer = AutoTokenizer.from_pretrained(args.model_path) # load from model's saving path so special tokens are added
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
model.eval()

test_inputs = [d.strip() for d in pd.read_csv(args.test_file, sep="\t")['source'].values]

# For t5
if args.model_name == "t5":
    prefix = "translate from Graph to Text:"
    test_inputs = [prefix + inp for inp in test_inputs]

decode_output_lines = []
tweak_processor = TweakLogits(
            generator=copy.deepcopy(model), 
            gen_tokenizer=tokenizer, 
            look_ahead_step=30,
            forward_tweak=args.forward_tweak, 
            backward_tweak=args.backward_tweak, 
            scoring_model=args.scoring_model, 
            lookahead_generation=args.lookahead_generation,
            bi_nli_scoring=args.bi_nli_scoring,
            dynamic_addition=args.dynamic_addition,
            alpha=args.alpha,
            hvm_path=args.hvm_path,
            train_with_sequence_loss=args.sequence_classifier
        )

if args.scoring_model == "nli":
    for i in tqdm(range(0, len(test_inputs), args.eval_batch_size)):
        with torch.no_grad():
            batch_inputs = test_inputs[i:i+args.eval_batch_size]
            inputs = tokenizer(batch_inputs, return_tensors='pt', padding=True).to(device)
            outputs = generate(model, tokenizer, **inputs, max_new_tokens=args.max_generate_length, num_beams=args.num_beams, using_tweak_decoding=True, tweak_processor=tweak_processor)
            decode_output_lines+=tokenizer.batch_decode(outputs, skip_special_tokens=True)

elif args.scoring_model == "hvm":
    prepared_input_batches = []
    # HVM preprocessing
    for i in tqdm(range(0, len(test_inputs), args.eval_batch_size)):
        sources = test_inputs[i:i+args.eval_batch_size]    
        generated_inputs = tokenizer(sources, return_tensors='pt', padding=True).to(device)
        sources_str, mask, num_triples = prepro_hvm_batch_inputs(sources, args, device)

        inputs = {}
        inputs['hvm_hypo_label_mask'] = mask
        inputs['hvm_num_triples'] = num_triples
        inputs['sources'] = sources_str
        inputs['inputs'] = generated_inputs

        prepared_input_batches.append(inputs)
    
    # TWEAK-HVM Decoding
    for batch_inputs in tqdm(prepared_input_batches):
        generate_inputs = batch_inputs['inputs']
        with torch.no_grad():
            tweak_processor.prepare_inputs(batch_inputs)
            outputs = generate(model, tokenizer, **generate_inputs, max_new_tokens=args.max_generate_length, num_beams=args.num_beams, using_tweak_decoding=True, tweak_processor=tweak_processor)
            decode_output_lines+=tokenizer.batch_decode(outputs, skip_special_tokens=True)


if args.gpu_id != -1:
    with open(args.output_path+'/generated_predictions_'+str(args.gpu_id)+'.txt', "w+", encoding="utf-8") as f:
        f.writelines([d+'\n' for d in decode_output_lines])
else:
    with open(args.output_path+'/generated_predictions.txt', "w+", encoding="utf-8") as f:
        f.writelines([d+'\n' for d in decode_output_lines])