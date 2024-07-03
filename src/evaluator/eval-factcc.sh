#!/bin/bash
# Evaluate FactCC model

echo "Evaluating FactCC metrics!"

source /miniconda/etc/profile.d/conda.sh

conda activate factcc

# prepro
SRC_PATH=$1
TGT_PATH=$2

echo "Source path for FactCC:"${SRC_PATH}
echo "Target path for FactCC:"${TGT_PATH}

python3 evaluator/prepro_factcc.py ${SRC_PATH} ${TGT_PATH}

# FactCC
sh evaluator/factCC/modeling/scripts/factcc-eval.sh ${PWD}/evaluator
rm evaluator/cached_dev_bert-base-uncased_512_factcc_annotated

# FactCC X
sh evaluator/factCC/modeling/scripts/factccx-eval.sh ${PWD}/evaluator
rm evaluator/cached_dev_bert-base-uncased_512_factcc_annotated