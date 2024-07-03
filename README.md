# TWEAK: Thinking While Effectively Articulating Knowledge
This software accompanies the paper [Thinking While Effectively Articulating Knowledge](https://arxiv.org/abs/2311.09467)

Yifu Qiu, Varun Embar, Shay B. Cohen, Benjamin Han

NAACL Findings, 2024

Please cite our paper if you find our work useful for your research:
```
@article{qiunaacl24,
  title={Think While You Write: Hypothesis Verification Promotes Faithful Knowledge-to-Text Generation},
  author={Qiu, Yifu and Embar, Varun and Cohen, Shay and Han, Benjamin},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2024},
  year={2024}
}
```

## Installation

Create a conda environment with python==3.8.16 and activate it
```
conda create -n tweak python==3.8.16
conda activate tweak
```

Install the dependencies
```
pip install -r requirements.txt
```

Install transformers 4.31
```
pip install transformers==4.31

```
Download en_core_web_sm for spacy 
```
python -m spacy download en_core_web_sm
```

Download datasets

WebNLG 
```
https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/
```

GenWiki
```
https://github.com/zhijing-jin/genwiki
```

TekGen
```
https://github.com/google-research-datasets/KELM-corpus#part-1-tekgen-training-corpus
```

To generate the splits used in this paper run the scripts in the scripts/data directory

```
python ./scripts/data/genwiki_gen_splits.py <genwiki_data_dir> <genwiki_output_dir> ./scripts/data/genwiki_validation_instances.csv
python ./scripts/data/tekgen_gen_splits.py <tekgen_data_dir> <tekgen_output_dir> ./scripts/data/tekgen_train_instances.csv ./scripts/data/tekgen_validation_instances.csv ./scripts/data/tekgen_test_instances.csv
python ./scripts/data/webnlg_gen_splits.py <webnlg_data_dir> <webnlg_output_dir> ./scripts/data/webnlg_validation_instances.csv ./scripts/data/webnlg_test_instances.csv
```

To run FactCC evaluation, install the requirements using the below command
```
pip install -r evaluator/factCC/requirements.txt
```
## Inference with TWEAK decoding

### Beam Search Decoding

The example script for running BART+Beam Search on WebNLG is, 

```
export PYTHONIOENCODING=utf8

### BART's decoding with BeamSearch
model_output_path=bart-webnlg-outputs
output_dir=beam_decoding_outputs

python3 decode-model.py \
    --model_path ${model_output_path} \
    --model_name bart \
    --output_path ${model_output_path}/${output_dir} \
    --eval_batch_size 8 \
    --num_beams 4 \
```

For `Greedy Search`, simply tune the `--num_beams` to be `1`.

### Tweak-NLI Decoding (backward-only)
The example script for running BART+Backward-only Tweak on WebNLG is, 

```
export PYTHONIOENCODING=utf8

### BART's decoding with TWEAK-NLI-BackwardOnly
model_output_path=bart-webnlg-outputs
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
```

### Tweak-NLI Decoding (forward-only)
The example script for running BART+Forward-only Tweak on WebNLG is, 

```
export PYTHONIOENCODING=utf8

### BART's decoding with TWEAK-NLI-ForwardOnly
model_output_path=bart-webnlg-outputs
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
```

### Full Tweak-NLI Decoding (forward+backward)
The example script for running BART+Full Tweak on WebNLG is, 

```
export PYTHONIOENCODING=utf8

### BART's decoding with TWEAK-NLI
model_output_path=bart-webnlg-outputs
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
```

### Full Tweak-HVM Decoding (forward+backward)

The example script for running BART+Full Tweak on WebNLG is, 

```
export PYTHONIOENCODING=utf8

### BART's decoding with TWEAK-HVM
model_output_path=bart-webnlg-outputs
hvm_path="webnlg-hvm"
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
```

### Other Experiments

For running all experiments (e.g., `T5` and `TekGEN` and `GenWiki`), please see `tweak_decode.sh`.

## Evaluation

Evaluating generated outputs using quality metrics,
```
## WebNLG Evaluation Code
python3 evaluator/eval-metrics.py ${model_output_path}/${output_dir} webnlg
python3 evaluator/eval-enfs.py ${model_output_path}/${output_dir} webnlg
bash evaluator/eval-mfact.sh datasets/webnlg/webnlg-test_both.csv ${model_output_path}/$output_dir
```

Explanation for each files,

1. `eval-metrics.py`: evaluating the quality metrics including `BLEU`, `METEOR` and `BERTScore`.  
2. `eval-enfs.py`: evaluating the faithfulness metrics including `FactKB`.  
4. `eval-mfact.py`: evaluating the faithfulness metrics `mFACT`.

## Parallelized Optimization

We also provide an optimized decoding script. We first split the testing set into multiple chunks (e.g., 4). Then we let each GPU run decoding job for one chunks in parallel. If you want to use this parallel decoding feature, use the script `tweak_decode_parallel.sh` to replace `tweak_decode.sh`.

## WebNLG Results for Reference

Using the scripts and checkpoints as above, you should be able to get the following results,

|WebNLG|Decoding|BLEU|METEOR|BERTScore|FactKB|mFACT|
|:----|:----|:----|:----|:----|:----|:----|
|BART|Greedy|51.3|66.79|94.2|27.74|89.37|
| |Beam|54.23|67.55|94.35|28.91|92.32|
| |TWEAK-NLI-F|52.02|67.17|94.2|30.46|92.05|
| |TWEAK-NLI-B|49.68|65.88|94.12|30.59|91.35|
| |TWEAK-NLI-B+F|51.62|66.84|94.19|30.47|91.68|
| |TWEAK-HVM|53.14|67.38|94.25|31.34|92.86|

|WebNLG|Decoding|BLEU|METEOR|BERTScore|FactKB|mFACT|
|:----|:----|:----|:----|:----|:----|:----|
|T5|Greedy|57.71|68.71|94.84|30.14|92.8|
| |Beam|58.93|69.38|94.86|31.29|94.47|
| |TWEAK-NLI-F|53.51|67.8|94.39|33.03|92.86|
| |TWEAK-NLI-B|44.96|65.02|93.93|31.49|89.8|
| |TWEAK-NLI-B+F|51.71|66.73|94.19|32.71|91.51|
| |TWEAK-HVM|57.31|69.02|94.68|33.34|94.58|


## Training the Generative Model from Scratch

You can use the following script for training the base K2T generator. Modify the data paths accordingly.

|Dataset|Model|Script|
|:----|:----|:----|
|WebNLG|BART-large|`run_bart_webnlg.sh`|
|WebNLG|T5-large|`run_t5_webnlg.sh`|
|TekGEN|BART-large|`run_bart_tekgen.sh`|
|TekGEN|T5-large|`run_t5_tekgen.sh`|
|GenWiki|BART-large|`run_bart_genwiki.sh`|
|GenWiki|T5-large|`run_t5_genwiki.sh`|

## Training the HVM from Scratch

You can use the following script for training from scratch,

`sh run_hvm_train.sh`

The best checkpoint is selected based on HVM's downstream performance (i.e., performing TWEAK decoding with K2T generator). It normally happens in the `2500th` update step.


## License
For licensing see accompanying LICENSE file.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
