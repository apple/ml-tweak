#!/bin/bash

export PYTHONIOENCODING=utf8
pretrained="facebook/bart-large"
model_output_path="./bart-tekgen-outputs/"
dataset="tekgen"

### Training
python3 src/transformers/run_kg2text.py \
    --model_name_or_path ${pretrained} \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file "datasets/"${dataset}"/"${dataset}"-train.csv" \
    --validation_file "datasets/"${dataset}"/"${dataset}"-validation.csv" \
    --test_file "datasets/"${dataset}"/"${dataset}"-test.csv" \
    --text_column "source" \
    --summary_column "target" \
    --num_train_epochs 2 \
    --output_dir $model_output_path \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --overwrite_output_dir \
    --predict_with_generate \
    --learning_rate 3e-05 \
    --metric_for_best_model sacrebleu \
    --save_strategy steps \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --generation_max_length 384 \
    --generation_num_beams 5