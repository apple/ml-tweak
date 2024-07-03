#!/bin/bash
# Evaluate mFACT model

echo "Evaluating mFACT metrics!"

source /miniconda/etc/profile.d/conda.sh

# conda activate ~/miniconda3/envs/mfact

# prepro
SRC_PATH=$1
TGT_PATH=$2

echo "Source path for mFACT:"${SRC_PATH}
echo "Target path for mFACT:"${TGT_PATH}

python3 evaluator/prepro_mfact.py ${SRC_PATH} ${TGT_PATH}

python3 evaluator/run_mfact.py \
      --model_name_or_path "evaluator/mfact-en" \
      --do_predict \
      --test_file "evaluator/mfact_inputs.csv" \
      --output_dir "evaluator/"  \
      --save_steps -1 \
      --overwrite_output_dir \
      --get_predict_scores