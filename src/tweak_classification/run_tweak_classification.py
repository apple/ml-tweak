#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import itertools

import datasets
import numpy as np
from datasets import load_dataset
import torch
import pandas as pd

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from tweak_utils import hypo_label_encoding, get_hypo_label_mask
from datasets import Dataset, DatasetDict
from tweak_classifier import TweakClassifier
from tweak_trainer import TweakTrainer

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_info()


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    get_predict_scores: bool = field(
        default=False, metadata={"help": "Get predicted scores rather than labels."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    language: str = field(
        default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    )
    train_language: Optional[str] = field(
        default=None, metadata={"help": "Train language if it is different from the evaluation language."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    num_max_triples: int = field(
        default=3,
        metadata={"help": "The max number of triples that encoder will take."},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    only_head_special_token: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    single_special_tok_representation: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    train_with_sequence_loss: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_xnli", model_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    if (data_args.train_file is not None) or (data_args.validation_file is not None) or (data_args.test_file is not None):
        data_files = {}
        if training_args.do_train and data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if training_args.do_eval and data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if training_args.do_predict and data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

        # preprocessing for the training set
        if (data_args.train_file is not None) and extension == "csv":
            df_train = pd.read_csv(data_args.train_file, sep='\t')
            df_train.fillna('', inplace=True)

            df = []
            for index, d in df_train.iterrows():

                triples, hypo_label = hypo_label_encoding(d['triples'], d['b_label'], d['f_label'], max_num_classes=model_args.num_max_triples)
                mask = get_hypo_label_mask(d['triples'], max_num_classes=model_args.num_max_triples)

                df.append([triples, d['b_hypo'], d['f_hypo'], hypo_label, mask, d['label']])

            df = pd.DataFrame(df, columns = ['triples', 'b_hypo', 'f_hypo', 'hypo_label_matrix', 'score_matrix_mask', 'labels'])
            dataset_train = Dataset.from_pandas(df)

        if (data_args.validation_file is not None) and extension == "csv":
            df_validation = pd.read_csv(data_args.validation_file, sep='\t')
            df_validation.fillna('', inplace=True)

            df = []
            for index, d in df_validation.iterrows():
                triples, hypo_label = hypo_label_encoding(d['triples'], d['b_label'], d['f_label'], max_num_classes=model_args.num_max_triples)
                mask = get_hypo_label_mask(d['triples'], max_num_classes=model_args.num_max_triples)

                df.append([triples, d['b_hypo'], d['f_hypo'], hypo_label, mask, d['label']])

            df = pd.DataFrame(df, columns = ['triples', 'b_hypo', 'f_hypo', 'hypo_label_matrix', 'score_matrix_mask', "labels"])
            dataset_validation = Dataset.from_pandas(df)

        if (data_args.test_file is not None) and extension == "csv":
            df_predict = pd.read_csv(data_args.test_file, sep='\t')
            df_predict.fillna('', inplace=True)

            df = []
            for index, d in df_predict.iterrows():
                triples, hypo_label = hypo_label_encoding(d['triples'], d['b_label'], d['f_label'], max_num_classes=model_args.num_max_triples)
                mask = get_hypo_label_mask(d['triples'], max_num_classes=model_args.num_max_triples)

                df.append([triples, d['b_hypo'], d['f_hypo'], hypo_label, mask, d['label']])

            df = pd.DataFrame(df, columns = ['triples', 'b_hypo', 'f_hypo', 'hypo_label_matrix', "score_matrix_mask", "labels"])
            dataset_test = Dataset.from_pandas(df)

        raw_datasets = DatasetDict({})

        # raw_datasets = load_dataset(
        #     extension,
        #     data_files=data_files,
        #     delimiter="\t",
        #     cache_dir=model_args.cache_dir,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )

        # Preprocessing the raw dataset used for training and validation
        if training_args.do_train:
            raw_datasets['train'] = dataset_train
        if training_args.do_eval:
            raw_datasets['validation'] = dataset_validation
        if training_args.do_predict:
            raw_datasets['test'] = dataset_test


        # Convert label column to labelFeature in HF
        print(raw_datasets)
    #     raw_datasets = raw_datasets.class_encode_column("f_label") # automatically find all the unique string values in the column
        raw_datasets = raw_datasets.class_encode_column("labels") # automatically find all the unique string values in the column

        if training_args.do_train:
            train_dataset=raw_datasets['train']
            label_list = train_dataset.features["labels"].names
    #         label_list += train_dataset.features["b_label"].names
        if training_args.do_eval:
            eval_dataset=raw_datasets['validation']
            label_list = eval_dataset.features["labels"].names
    #         label_list += eval_dataset.features["b_label"].names
        if training_args.do_predict:
            predict_dataset=raw_datasets['test']
            label_list = predict_dataset.features["labels"].names
    #         label_list += predict_dataset.features["b_label"].names

    # # Labels
    print("The label list is", label_list, "; with number of label in", len(label_list))
    num_labels = len(label_list)
    # 1/0

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, # roberta-large
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # processing tokenizer to allow additional special tokens
    # We need <B> for backward hypo, <F> for forward hypo, <H1> <R1> <T1> for the first triple, etc.

    # for kg2t, append <H> special tokens
    # new_tokens = ['<B>', '<F>']
    # new_tokens += ['<H'+str(i)+'>' for i in range(model_args.num_max_triples)]
    # new_tokens += ['<R'+str(i)+'>' for i in range(model_args.num_max_triples)]
    # new_tokens += ['<T'+str(i)+'>' for i in range(model_args.num_max_triples)]
    
    if model_args.only_head_special_token:
        new_tokens = ['<B>', '<F>', '<H>'] # 50265, 50266, 50267, 50268, 50269
    else:
        new_tokens = ['<B>', '<F>', '<H>', '<R>', '<T>'] # 50265, 50266, 50267, 50268, 50269

    new_tokens_vocab = {}
    new_tokens_vocab['additional_special_tokens'] = []
    for idx, t in enumerate(new_tokens):
        new_tokens_vocab['additional_special_tokens'].append(t)
    num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)
    print('We have added ', num_added_toks, ' tokens')

    # Loading the main model from its base checkpoint
    if training_args.do_train:
        model = TweakClassifier(base_model=model_args.model_name_or_path, max_num_classes=model_args.num_max_triples, num_labels=num_labels, model_config=config, single_special_tok_representation=model_args.single_special_tok_representation, train_with_sequence_loss=model_args.train_with_sequence_loss)
    else:
        model = TweakClassifier(base_model="roberta-large", max_num_classes=model_args.num_max_triples, num_labels=num_labels, model_config=config, single_special_tok_representation=model_args.single_special_tok_representation, train_with_sequence_loss=model_args.train_with_sequence_loss)
        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.base_model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.base_model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(model_args.model_name_or_path+'/pytorch_model.bin'))

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.base_model.resize_token_embeddings(len(tokenizer))

    print(tokenizer)
    print(model)

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    # )
    
    

    # from transformers.utils import find_labels
    # print(find_labels(model.__class__))

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):

        # Convert triple to string form
        for i_instance, triples in enumerate(examples['triples']):
            if model_args.only_head_special_token:
                examples['triples'][i_instance] = ' '.join(["<H> "+t[0].strip()+" "+t[1].strip()+" "+t[2].strip() for t in triples])
            else:
                examples['triples'][i_instance] = ' '.join(["<H> "+t[0].strip()+" <R> "+t[1].strip()+" <T> "+t[2].strip() for t in triples])
        
        # Preprocessing the input text to ideal format
        processed_inputs = []
        for i, (t, b, f) in enumerate(zip(examples['triples'], examples["b_hypo"], examples["f_hypo"])):
            if model_args.single_special_tok_representation:
                input_str = t + ' </s> <B> ' + b + ' <F> ' + f # we append </s> to mimic nli format
            else:
                input_str = t + ' <B> ' + b + ' <F> ' + f
            processed_inputs.append(input_str)

        # Tokenize the texts
        text_toks = tokenizer(
            processed_inputs,
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        return text_toks

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Get the metric function
    # metric = evaluate.load("xnli")
    metric = evaluate.combine([
        evaluate.load("precision", average="macro"),
        evaluate.load("recall", average="macro"),
        evaluate.load("f1", average="macro")
    ])
    # metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # # predictions and label_ids field) and has to return a dictionary string to float.
    # def compute_metrics(p: EvalPrediction):
        
    #     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        
    #     # print(type(preds))
    #     # print(p.label_ids)
    #     # 1/0
        
    #     # preds = np.argmax(preds, axis=1)
        
    #     return metric.compute(predictions=preds, references=p.label_ids)

    # # Element-wise
    def compute_metrics(p: EvalPrediction):
        
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        matrix_logits = p.predictions[1] if isinstance(p.predictions, tuple) else p.predictions
        
        matrix_logits = torch.tensor(matrix_logits)
        matrix_preds = torch.argmax(matrix_logits[:,:2,:,:], dim=-1).permute(0, 2, 1).tolist()

        all_preds = []
        all_labels = []
        for preds, labels in zip(matrix_preds, p.label_ids):
            num_triples = len(labels)
            preds = preds[:num_triples]

            all_preds += list(itertools.chain.from_iterable(preds))
            all_labels += list(itertools.chain.from_iterable(labels))

        print("# Hallucinated: ", all_labels.count(0))
        print("# Faithful: ", all_labels.count(1))
        print("SKlearn Acc per class:", accuracy_score(all_labels, all_preds))
        print("SKlearn F1 score per class:", f1_score(all_labels, all_preds, average=None))
        print("SKlearn Precision per class:", precision_score(all_labels, all_preds, average=None))
        print("SKlearn Recall per class:", recall_score(all_labels, all_preds, average=None))

        # print(all_preds)
        # assert len(all_preds) == len(all_labels)
        # 1/0
        
        return metric.compute(predictions=all_preds, references=all_labels, average='macro')

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    training_args.label_names = ['labels']
    # Initialize our Trainer
    trainer = TweakTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        
        # scores = torch.softmax(torch.tensor(predictions), axis=1) # softmax to normalize
        # scores = scores[:,1] # 1 for entailment score

        # predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                if data_args.get_predict_scores:
                    for index, item in enumerate(scores):
                        writer.write(f"{index}\t{item}\n")
                else:
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()