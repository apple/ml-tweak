#!/bin/bash

export PYTHONIOENCODING=utf8

gpu_num=4

### HVM Config
hvm_path="webnlg-hvm"

############### Webnlg ###############
## Parallelizing the dataset
python3 dataset-parallel.py ./datasets/webnlg/webnlg-test_both.csv $gpu_num
echo "Finishing WebNLG dataset truncaiton!"

## Webnlg+BART
model_output_path=bart-webnlg-outputs
output_dir=tweak_hvm_decoding_outputs
rm ${model_output_path}/${output_dir}/generated_predictions.txt

for gpu_i in `seq 0 $(expr $gpu_num - 1)`
do
    {
    export CUDA_VISIBLE_DEVICES=$gpu_i
    python3 decode-model.py \
        --test_file "./datasets/webnlg/webnlg-test_both_"${gpu_i}".csv" \
        --model_path ${model_output_path} \
        --hvm_path ${hvm_path} \
        --model_name bart \
        --output_path ${model_output_path}/${output_dir} \
        --eval_batch_size 8 \
        --num_beams 4 \
        --forward_tweak --backward_tweak \
        --scoring_model "hvm" \
        --lookahead_generation "greedy" \
        --dynamic_addition \
        --alpha 8 \
        --sequence_classifier \
        --gpu_id $gpu_i
    }&
done
wait

# ## Webnlg+T5
# model_output_path=t5-webnlg-outputs
# output_dir=tweak_hvm_decoding_outputs
# rm ${model_output_path}/${output_dir}/generated_predictions.txt

# for gpu_i in `seq 0 $(expr $gpu_num - 1)`
# do
#     {
#     export CUDA_VISIBLE_DEVICES=$gpu_i
#     python3 decode-model.py \
#         --test_file "./datasets/webnlg/webnlg-test_both_"${gpu_i}".csv" \
#         --model_path ${model_output_path} \
#         --hvm_path ${hvm_path} \
#         --model_name t5 \
#         --output_path ${model_output_path}/${output_dir}  \
#         --eval_batch_size 8 \
#         --num_beams 3 \
#         --forward_tweak --backward_tweak \
#         --scoring_model "hvm" \
#         --lookahead_generation "greedy" \
#         --alpha 8 \
#         --gpu_id $gpu_i
#     }&
# done
# wait

# # combine outputs in chunks to one file
echo "Merging different decoding parts for WebNLG..."
cat ${model_output_path}/${output_dir}/generated_predictions_*.txt >> ${model_output_path}/${output_dir}/generated_predictions.txt

python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} webnlg
python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} webnlg
bash evaluator/eval-factcc.sh datasets/webnlg/webnlg-test_both.csv ${model_output_path}/$output_dir
bash evaluator/eval-mfact.sh datasets/webnlg/webnlg-test_both.csv ${model_output_path}/$output_dir

############### TekGEN ###############

# python3 dataset-parallel.py ./datasets/tekgen/tekgen-test.csv $gpu_num
# echo "Finishing TekGEN dataset truncaiton!"

# # ## TekGEN+BART
# model_output_path=bart-tekgen-outputs
# output_dir=tweak_hvm_decoding_outputs
# rm ${model_output_path}/${output_dir}/generated_predictions.txt

# for gpu_i in `seq 0 $(expr $gpu_num - 1)`
# do
#     {
#     export CUDA_VISIBLE_DEVICES=$gpu_i
#     python3 decode-model.py \
#         --test_file "./datasets/tekgen/tekgen-test_"${gpu_i}".csv" \
#         --model_path ${model_output_path} \
#         --hvm_path ${hvm_path} \
#         --model_name bart \
#         --output_path ${model_output_path}/${output_dir} \
#         --eval_batch_size 8 \
#         --num_beams 4 \
#         --forward_tweak --backward_tweak \
#         --scoring_model "hvm" \
#         --lookahead_generation "greedy" \
#         --dynamic_addition \
#         --alpha 8 \
#         --gpu_id $gpu_i
#     }&
# done
# wait

# ## TekGEN+T5
# model_output_path=t5-tekgen-outputs
# output_dir=tweak_hvm_decoding_outputs
# rm ${model_output_path}/${output_dir}/generated_predictions.txt

# for gpu_i in `seq 0 $(expr $gpu_num - 1)`
# do
#     {
#     export CUDA_VISIBLE_DEVICES=$gpu_i
#     python3 decode-model.py \
#         --test_file "./datasets/tekgen/tekgen-test_"${gpu_i}".csv" \
#         --model_path ${model_output_path} \
#         --hvm_path ${hvm_path} \
#         --model_name t5 \
#         --output_path ${model_output_path}/${output_dir}  \
#         --eval_batch_size 8 \
#         --num_beams 3 \
#         --forward_tweak --backward_tweak \
#         --scoring_model "hvm" \
#         --lookahead_generation "greedy" \
#         --dynamic_addition \
#         --alpha 1 \
#         --gpu_id $gpu_i
#     }&
# done
# wait

# # # combine outputs in chunks to one file
# echo "Merging different decoding parts for TekGEN..."
# cat ${model_output_path}/${output_dir}/generated_predictions_*.txt >> ${model_output_path}/${output_dir}/generated_predictions.txt

# python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} tekgen
# python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} tekgen
# bash evaluator/eval-factcc.sh datasets/tekgen/tekgen-test.csv ${model_output_path}/$output_dir
# bash evaluator/eval-mfact.sh datasets/tekgen/tekgen-test.csv ${model_output_path}/$output_dir

############### GenWiki ###############

# # ## GenWiki+BART
# model_output_path=bart-genwiki-outputs
# output_dir=tweak_hvm_decoding_outputs
# rm ${model_output_path}/${output_dir}/generated_predictions.txt

# for gpu_i in `seq 0 $(expr $gpu_num - 1)`
# do
#     {
#     export CUDA_VISIBLE_DEVICES=$gpu_i
#     python3 decode-model.py \
#         --test_file "./datasets/genwiki/genwiki-test_"${gpu_i}".csv" \
#         --model_path ${model_output_path} \
#         --hvm_path ${hvm_path} \
#         --model_name bart \
#         --output_path ${model_output_path}/${output_dir} \
#         --eval_batch_size 8 \
#         --num_beams 5 \
#         --forward_tweak --backward_tweak \
#         --scoring_model "hvm" \
#         --lookahead_generation "greedy" \
#         --dynamic_addition \
#         --alpha 8 \
#         --gpu_id $gpu_i
#     }&
# done
# wait

# ## GenWiki+T5
# model_output_path=t5-genwiki-outputs
# output_dir=tweak_hvm_decoding_outputs
# rm ${model_output_path}/${output_dir}/generated_predictions.txt

# for gpu_i in `seq 0 $(expr $gpu_num - 1)`
# do
#     {
#     export CUDA_VISIBLE_DEVICES=$gpu_i
#     python3 decode-model.py \
#         --test_file "./datasets/genwiki/genwiki-test_"${gpu_i}".csv" \
#         --model_path ${model_output_path} \
#         --hvm_path ${hvm_path} \
#         --model_name t5 \
#         --output_path ${model_output_path}/${output_dir}  \
#         --eval_batch_size 8 \
#         --num_beams 3 \
#         --forward_tweak --backward_tweak \
#         --scoring_model "hvm" \
#         --lookahead_generation "greedy" \
#         --dynamic_addition \
#         --alpha 2 \
#         --gpu_id $gpu_i
#     }&
# done
# wait

# # # combine outputs in chunks to one file
# echo "Merging different decoding parts for GenWiki..."
# cat ${model_output_path}/${output_dir}/generated_predictions_*.txt >> ${model_output_path}/${output_dir}/generated_predictions.txt

# python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} genwiki
# python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} genwiki
# bash evaluator/eval-factcc.sh datasets/genwiki/genwiki-test.csv ${model_output_path}/$output_dir
# bash evaluator/eval-mfact.sh datasets/genwiki/genwiki-test.csv ${model_output_path}/$output_dir