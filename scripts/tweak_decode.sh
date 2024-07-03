#!/bin/bash
export PYTHONIOENCODING=utf8

### Config HVM
hvm_path="webnlg-hvm"

############### Webnlg ###############
model_output_path=bart-webnlg-outputs

### BART's decoding with TWEAK-HVM
output_dir=tweak_hvm_decoding_outputs
python3 decode-model.py \
    --model_path ${model_output_path} \
    --hvm_path ${hvm_path} \
    --model_name bart \
    --output_path ${model_output_path}/${output_dir} \
    --eval_batch_size 8 \
    --num_beams 4 \
    --backward_tweak --forward_tweak \
    --scoring_model "hvm" \
    --lookahead_generation "greedy" \
    --dynamic_addition \
    --alpha 8 \

### BART's decoding with TWEAK-NLI
output_dir=tweak_nli_decoding_outputs
python3 decode-model.py \
    --model_path ${model_output_path} \
    --model_name bart \
    --output_path ${model_output_path}/${output_dir} \
    --eval_batch_size 8 \
    --num_beams 4 \
    --backward_tweak --forward_tweak \
    --scoring_model "nli" \
    --lookahead_generation "greedy" \
    --dynamic_addition \
    --alpha 8 \

### BART's decoding with TWEAK-NLI-ForwardOnly
output_dir=tweak_F_nli_decoding_outputs
python3 decode-model.py \
    --model_path ${model_output_path} \
    --model_name bart \
    --output_path ${model_output_path}/${output_dir} \
    --eval_batch_size 8 \
    --num_beams 4 \
    --forward_tweak \
    --scoring_model "nli" \
    --lookahead_generation "greedy" \
    --alpha 8 \

### BART's decoding with TWEAK-NLI-BackwardOnly
output_dir=tweak_B_nli_decoding_outputs
python3 decode-model.py \
    --model_path ${model_output_path} \
    --model_name bart \
    --output_path ${model_output_path}/${output_dir} \
    --eval_batch_size 8 \
    --num_beams 4 \
    --backward_tweak \
    --scoring_model "nli" \
    --alpha 8 \

### BART's decoding with BeamSearch
output_dir=beam_decoding_outputs
python3 decode-model.py \
    --model_path ${model_output_path} \
    --model_name bart \
    --output_path ${model_output_path}/${output_dir} \
    --eval_batch_size 8 \
    --num_beams 4 \

### BART's decoding with GreedySearch
output_dir=greedy_decoding_outputs
python3 decode-model.py \
    --model_path ${model_output_path} \
    --model_name bart \
    --output_path ${model_output_path}/${output_dir} \
    --eval_batch_size 8 \
    --num_beams 1 \

## WebNLG Evaluation Code
python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} webnlg
python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} webnlg
bash evaluator/eval-factcc.sh datasets/webnlg/webnlg-test_both.csv ${model_output_path}/$output_dir
bash evaluator/eval-mfact.sh datasets/webnlg/webnlg-test_both.csv ${model_output_path}/$output_dir

### T5's decoding with TWEAK-HVM
model_output_path=t5-webnlg-outputs
output_dir=tweak_hvm_decoding_outputs

python3 decode-model.py \
    --model_path ${model_output_path} \
    --hvm_path ${hvm_path} \
    --model_name t5 \
    --output_path ${model_output_path}/${output_dir}  \
    --eval_batch_size 8 \
    --num_beams 3 \
    --forward_tweak --backward_tweak \
    --scoring_model "hvm" \
    --lookahead_generation "greedy" \
    --dynamic_addition \
    --alpha 8 \

## WebNLG Evaluation Code
python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} webnlg
python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} webnlg
bash evaluator/eval-factcc.sh datasets/webnlg/webnlg-test_both.csv ${model_output_path}/$output_dir
bash evaluator/eval-mfact.sh datasets/webnlg/webnlg-test_both.csv ${model_output_path}/$output_dir

############### TekGEN ###############

### BART TWEAK-HVM
model_output_path=bart-tekgen-outputs
output_dir=tweak_hvm_decoding_outputs

python3 decode-model.py \
    --test_file "./datasets/tekgen/tekgen-test.csv" \
    --model_path ${model_output_path} \
    --hvm_path ${hvm_path} \
    --model_name bart \
    --eval_batch_size 8 \
    --output_path ${model_output_path}/${output_dir}  \
    --num_beams 5 \
    --forward_tweak --backward_tweak \
    --scoring_model "hvm" \
    --lookahead_generation "greedy" \
    --dynamic_addition \
    --alpha 8 \

## TekGEN Evaluation Code
python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} tekgen
python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} tekgen
bash evaluator/eval-factcc.sh datasets/tekgen/tekgen-test.csv ${model_output_path}/$output_dir
bash evaluator/eval-mfact.sh datasets/tekgen/tekgen-test.csv ${model_output_path}/$output_dir

### T5 TWEAK-HVM
model_output_path=t5-tekgen-outputs
output_dir=tweak_hvm_decoding_outputs

python3 decode-model.py \
    --test_file "./datasets/tekgen/tekgen-test.csv" \
    --model_path ${model_output_path} \
    --hvm_path ${hvm_path} \
    --model_name t5 \
    --output_path ${model_output_path}/${output_dir}  \
    --eval_batch_size 8 \
    --num_beams 3 \
    --forward_tweak --backward_tweak \
    --scoring_model "hvm" \
    --lookahead_generation "greedy" \
    --dynamic_addition \
    --alpha 1 \

## TekGEN Evaluation Code
python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} tekgen
python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} tekgen
bash evaluator/eval-factcc.sh datasets/tekgen/tekgen-test.csv ${model_output_path}/$output_dir
bash evaluator/eval-mfact.sh datasets/tekgen/tekgen-test.csv ${model_output_path}/$output_dir


############### GenWiki ###############

### BART TWEAK-HVM
model_output_path=bart-genwiki-outputs
output_dir=tweak_hvm_decoding_outputs

python3 decode-model.py \
    --test_file "./datasets/genwiki/genwiki-test.csv" \
    --model_path ${model_output_path} \
    --hvm_path ${hvm_path} \
    --model_name bart \
    --eval_batch_size 8 \
    --output_path ${model_output_path}/${output_dir}  \
    --num_beams 5 \
    --forward_tweak --backward_tweak \
    --scoring_model "hvm" \
    --lookahead_generation "greedy" \
    --dynamic_addition \
    --alpha 8 \

python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} genwiki
python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} genwiki
bash evaluator/eval-factcc.sh datasets/genwiki/genwiki-test.csv ${model_output_path}/$output_dir
bash evaluator/eval-mfact.sh datasets/genwiki/genwiki-test.csv ${model_output_path}/$output_dir

### T5 TWEAK-HVM
model_output_path=t5-genwiki-outputs
output_dir=tweak_hvm_decoding_outputs

python3 decode-model.py \
    --test_file "./datasets/genwiki/genwiki-test.csv" \
    --model_path ${model_output_path} \
    --hvm_path ${hvm_path} \
    --model_name t5 \
    --output_path ${model_output_path}/${output_dir}  \
    --eval_batch_size 8 \
    --num_beams 3 \
    --forward_tweak --backward_tweak \
    --scoring_model "hvm" \
    --lookahead_generation "greedy" \
    --dynamic_addition \
    --alpha 2 \

python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} genwiki
python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} genwiki
bash evaluator/eval-factcc.sh datasets/genwiki/genwiki-test.csv ${model_output_path}/$output_dir
bash evaluator/eval-mfact.sh datasets/genwiki/genwiki-test.csv ${model_output_path}/$output_dir