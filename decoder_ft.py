#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["NCCL_P2P_DISABLE"] ="1"
# os.environ["NCCL_IB_DISABLE"] ="1"

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import transformers
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

import datasets
import evaluate
import nltk
import numpy as np
import torch
from datasets import load_dataset

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__)

def main():

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")

    parser.add_argument('--data_dir', default='dataset', type=str, help=" A directory containing the training, validation and testing data.")
    parser.add_argument('--data_cache_dir', default='t5_cache_dir', type=str)
    
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. "
            "during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    # parser.add_argument(
    #     "--pad_to_max_length",
    #     action="store_true",
    #     help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    # )
    parser.add_argument(
        "--model_name_or_path",
        default="./models/t5-small",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")


    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    # parser.add_argument(
    #     "--resume_from_checkpoint",
    #     type=str,
    #     default=None,
    #     help="If the training should continue from a checkpoint folder.",
    # )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )

    args = parser.parse_args()


    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
    # Handle the repository creation
    if accelerator.is_main_process:
        # if args.push_to_hub:
        #     # Retrieve of infer repo_name
        #     repo_name = args.hub_model_id
        #     if repo_name is None:
        #         repo_name = Path(args.output_dir).absolute().name
        #     # Create repo and retrieve repo_id
        #     api = HfApi()
        #     repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

        #     with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
        #         if "step_*" not in gitignore:
        #             gitignore.write("step_*\n")
        #         if "epoch_*" not in gitignore:
        #             gitignore.write("epoch_*\n")
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    data_files = {}
    data_files["train"] = os.path.join(args.data_dir,'train_decoder.csv')
    data_files["dev"] = os.path.join(args.data_dir,'dev_decoder.csv')
    data_files["test"] = os.path.join(args.data_dir,'test_decoder.csv')
    
    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=args.data_cache_dir,
        num_proc = args.preprocessing_num_workers
    )

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, use_safetensors=False)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    
    
    def split_into_sequence(examples):
        model_inputs = {
            "src": [],
            "tgt": []
        }
        
        # for example in examples["text"]:
        #     line =  example.split("\t")
            
        #     if len(line) == 2:
        #         continue

        #     if line[-1] == "<self>":
        #         continue
            
        #     cls, s, t = line
            
        #     prefix = f"Normalize {cls}: \n"
        #     model_inputs["src"].append(prefix + s)
        #     model_inputs["tgt"].append(t)
        # for token, tag, decode in zip(examples["tokens"], examples["tags"], examples["decodes"]):
        #     if decode not in ["<self>","sil"]:
        #         prefix = f"normalize {tag}: "
        #         src = prefix + token
        #         tgt = decode
        #         if src not in model_inputs["src"]:
        #             model_inputs["src"].append(src)
        #             model_inputs["tgt"].append(tgt)
        tag = examples["Semiotic Class"].lower()
        prefix = f"normalize {tag}: "
        src = prefix + examples["Input Token"]
        tgt = examples["Output Token"]
        model_inputs["src"].append(src)
        model_inputs["tgt"].append(tgt)
        return model_inputs

    raw_datasets = raw_datasets.map(split_into_sequence,
                                    batch_size=None,
                                    # remove_columns=["sentence","tokens","tags","decodes","labels"],
                                    remove_columns=["Semiotic Class","Input Token","Output Token"],
                                    load_from_cache_file=not args.overwrite_cache,
                                    num_proc=args.preprocessing_num_workers,
                                   )
        
    # raw_datasets["train"] = raw_datasets["train"].select(range(1000))
    # raw_datasets["dev"] = raw_datasets["dev"].select(range(200))
    # raw_datasets["test"] = raw_datasets["test"].select(range(200))

    # Temporarily set max_target_length for training.
    # max_target_length = args.max_target_length
    # padding = "max_length" if args.pad_to_max_length else False
    
    def preprocess(examples):

        source_text = [ s for src in examples["src"] for s in src ]
        target_text = [ t for tgt in examples["tgt"] for t in tgt ]
        # model_inputs = tokenizer(examples["src"])
        # labels = tokenizer(examples["tgt"])
        
        model_inputs = tokenizer(source_text,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=args.max_source_length)
        labels = tokenizer(target_text,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=args.max_target_length)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        # if padding == "max_length":
        labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(preprocess, 
                                         batched=True, 
                                         remove_columns=["src","tgt"],
                                         load_from_cache_file=not args.overwrite_cache,
                                         num_proc=args.preprocessing_num_workers,
                                         desc="Running tokenizer on train dataset")

        eval_dataset = raw_datasets["dev"].map(
            preprocess,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=["src","tgt"],
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
                
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization_no_trainer", experiment_config)

    # Metric
    metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_acc = 0.0
    # Potentially load in the weights and states from a previous save
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
    #         checkpoint_path = args.resume_from_checkpoint
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
    #         dirs.sort(key=os.path.getctime)
    #         path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
    #         checkpoint_path = path
    #         path = os.path.basename(checkpoint_path)

    #     accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
    #     accelerator.load_state(checkpoint_path)
    #     # Extract `epoch_{i}` or `step_{i}`
    #     training_difference = os.path.splitext(path)[0]

    #     if "epoch" in training_difference:
    #         starting_epoch = int(training_difference.replace("epoch_", "")) + 1
    #         resume_step = None
    #         completed_steps = starting_epoch * num_update_steps_per_epoch
    #     else:
    #         # need to multiply `gradient_accumulation_steps` to reflect real steps
    #         resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
    #         starting_epoch = resume_step // len(train_dataloader)
    #         completed_steps = resume_step // args.gradient_accumulation_steps
    #         resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        # if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
        #     # We skip the first `n` batches in the dataloader when resuming from a checkpoint
        #     active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        # else:
        #     active_dataloader = train_dataloader

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        model.eval()

        gen_kwargs = {
            "max_length": args.max_target_length,
            "num_beams": args.num_beams,
        }
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                # if not args.pad_to_max_length:
                #     # If we did not pad to max length, we need to pad the labels too
                #     labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                true_labels = [1] * len(decoded_labels)
                predicted_labels = [0] * len(decoded_preds)
                for idx, (pred,label) in enumerate(zip(decoded_preds, decoded_labels)):
                    if pred == label:
                        predicted_labels[idx] = 1
                # print("decoded_preds:", decoded_preds[:10])
                # print("decoded_labels:", decoded_labels[:10])
                # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                # metric.add_batch(
                #     predictions=decoded_preds,
                #     references=decoded_labels,
                # )
                metric.add_batch(
                    predictions=predicted_labels,
                    references=true_labels,
                )
        # result = metric.compute(use_stemmer=True)
        result = metric.compute()
        result = {k: round(v * 100, 4) for k, v in result.items()}

        logger.info(result)
        if result["accuracy"] > best_acc:
            best_acc = result["accuracy"]
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
            accelerator.save_state(output_dir)
        if args.with_tracking:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)

        # if args.push_to_hub and epoch < args.num_train_epochs - 1:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(
        #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        #     )
        #     if accelerator.is_main_process:
        #         tokenizer.save_pretrained(args.output_dir)
                # api.upload_folder(
                #     commit_message=f"Training in progress epoch {epoch}",
                #     folder_path=args.output_dir,
                #     repo_id=repo_id,
                #     repo_type="model",
                #     token=args.hub_token,
                # )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            # if args.push_to_hub:
            #     api.upload_folder(
            #         commit_message="End of training",
            #         folder_path=args.output_dir,
            #         repo_id=repo_id,
            #         repo_type="model",
            #         token=args.hub_token,
            #     )

            all_results = {f"eval_{k}": v for k, v in result.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

if __name__ == '__main__':
    main() 