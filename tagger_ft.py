import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["NCCL_P2P_DISABLE"] ="1"
# os.environ["NCCL_IB_DISABLE"] ="1"

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed
)
import numpy as np
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

from transformers.utils import send_example_telemetry

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

logger = logging.getLogger(__name__)

# 'DATE', 'CARDINAL', 'DECIMAL','MEASURE', 'MONEY', 'ORDINAL', 'TIME', 'DIGIT', 'FRACTION', 'TELEPHONE', 'ADDRESS'
label_list = ['DATE', 'CARDINAL', 'DECIMAL','MEASURE', 'MONEY', 'ORDINAL', 'TIME', 'DIGIT', 'FRACTION', 'TELEPHONE', 'ADDRESS']
bio_label_list = ["O"]
for l in label_list:
    bio_label_list.append(f"B-{l}")
    bio_label_list.append(f"I-{l}")
    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )

    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    num_proc: int = field(
        default=None,
        metadata={
            "help": (
                "Number of processes when downloading and generating the dataset locally."
                "Multiprocessing is disabled by default."
            )
        }
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

    def __post_init__(self):
        if self.train_file is None and self.eval_file is None:
            raise ValueError("Need either a dataset name or a training/eval file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.eval_file is not None:
                extension = self.eval_file.split(".")[-1]
                assert extension in ["csv", "json"], "`eval_file` should be a csv or a json file."


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    for prediction, label in zip(predictions, labels):
        true_pred = []
        for (p, l) in zip(prediction, label):
            if l != -100:
                true_pred.append(bio_label_list[p])
        true_predictions.append(true_pred)

    true_labels = []
    for prediction, label in zip(predictions, labels):
        true_label = []
        for (p, l) in zip(prediction, label):
            if l != -100:
                true_label.append(bio_label_list[l])
        true_labels.append(true_label)

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    overall_result = classification_report(true_labels, true_predictions)
    print(overall_result)
    
    result = {}
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1
    
    return result

def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("train_tagger", model_args, data_args)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
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

    label2id = {j:i for i,j in enumerate(bio_label_list)}
    id2label = {i:j for i,j in enumerate(bio_label_list)}
    num_labels = len(bio_label_list)

    model = AutoModelForTokenClassification.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.eval_file is not None:
        data_files["eval"] = data_args.eval_file
        extension = data_args.eval_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, num_proc = data_args.num_proc)

    def tokenize_and_align_labels(examples):

        tokenized_inputs = tokenizer.batch_encode_plus(examples["tokens"],
                                    padding="max_length",
                                    truncation=True,
                                    max_length=data_args.max_seq_length,
                                    is_split_into_words=True)
                
        labels = []
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    tag = label[word_idx]
                    
                    if word_idx == previous_word_idx and tag != "O":
                        label_ids.append(label2id[f"I-{tag}"])
                    else:
                        if tag != "O":
                            label_ids.append(label2id[f"B-{tag}"])
                        else:
                            label_ids.append(label2id[tag])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    if training_args.do_train:

        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.map(tokenize_and_align_labels, 
                                        batched=True, 
                                        remove_columns=["tokens","tags","labels"],
                                        load_from_cache_file=True,
                                        num_proc=data_args.num_proc,
                                        desc="Running tokenizer on train dataset")

    if training_args.do_eval:

        if "eval" not in raw_datasets:
            raise ValueError("--do_eval requires a eval dataset")
        
        eval_dataset = raw_datasets["eval"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    
        eval_dataset = eval_dataset.map(tokenize_and_align_labels, 
                                        batched=True, 
                                        remove_columns=["tokens","tags","labels"],
                                        load_from_cache_file=True,
                                        num_proc=data_args.num_proc,
                                        desc="Running tokenizer on eval dataset")
    if training_args.do_predict:

        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        
        test_dataset = raw_datasets["test"]

        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(test_dataset), data_args.max_predict_samples)
            test_dataset = test_dataset.select(range(max_predict_samples))
            
        test_dataset = test_dataset.map(tokenize_and_align_labels, 
                                        batched=True, 
                                        remove_columns=["tokens","tags","labels"],
                                        load_from_cache_file=True,
                                        num_proc=data_args.num_proc,
                                        desc="Running tokenizer on test dataset")
        
        
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset  if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if training_args.do_eval:

        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
         
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

if __name__ == '__main__':
    main() 