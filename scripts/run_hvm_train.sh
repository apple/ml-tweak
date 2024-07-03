export PYTHONIOENCODING=utf8

# max length
# webnlg: 7

dataset="webnlg"

python src/tweak_classification/run_tweak_classification.py \
    --model_name_or_path "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli" \
    --train_file "datasets/tweak-classification-datasets/dsample-tweak-classification-"${dataset}"-train-200k.csv" \
    --validation_file "datasets/tweak-classification-datasets/dsample-tweak-classification-"${dataset}"-val.csv" \
    --test_file "datasets/tweak-classification-datasets/dsample-tweak-classification-"${dataset}"-test.csv" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 512 \
    --num_max_triples 7 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --output_dir "tweak-classification/"${dataset}"-full" \
    --evaluation_strategy "steps" \
    --save_strategy steps \
    --eval_steps 2500 \
    --save_steps 2500 \
    --logging_steps 2500 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --train_with_sequence_loss
