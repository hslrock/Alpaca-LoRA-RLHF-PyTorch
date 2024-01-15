#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2023-04-24 23:52:38
LastEditTime: 2023-04-24 23:52:41
LastEditors: Kun
Description: 
FilePath: /Alpaca-LoRA-RLHF-PyTorch/data_loader/rm_dataloader.py
'''


"""
data loader for reward modeling
"""

from datasets import load_dataset

import numpy as np
class CustomRewardDataLoader(object):
    def __init__(self, ds,dataset_name, tokenizer, num_proc =24) -> None:
        super(CustomRewardDataLoader, self).__init__()

        self.dataset_name = dataset_name
        self.ds = ds
        self.num_proc = num_proc
        self.tokenizer = tokenizer

    # Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
    # Then tokenize the dataset.

    def preprocess_function(self, examples):
        new_examples = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "score" : []
        }
        for instruction, question, response,score in zip(examples["instruction"], examples["input"], examples["output"],examples["return_min"]):
            tokenized_j = self.tokenizer(
                "Instruction: " + instruction + "\n\nQuestion: " + question+ "\n\Response: " + response)
            new_examples["input_ids_j"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_j"].append(
                tokenized_j["attention_mask"])
        
            
            new_examples['score'].append(np.clip(-np.log(score+1e-9),None,0))
        return new_examples

    def load_data(self):

        # Load the human stack-exchange-paired dataset for tuning the reward model.
        # train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train")
        train_dataset = self.ds['train']
        # eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
        eval_dataset = self.ds['test']

        original_columns = train_dataset.column_names
        print("train_dataset: ", len(train_dataset))
        train_dataset = train_dataset.map(
            self.preprocess_function, batched=True, num_proc=self.num_proc, remove_columns=original_columns
        )
        train_dataset = train_dataset.filter(lambda x: len(
            x["input_ids_j"]) <= 512)
        print("train_dataset: ", len(train_dataset))
        
        print("eval_dataset: ", len(eval_dataset))
        eval_dataset = eval_dataset.map(
            self.preprocess_function, batched=True, num_proc=self.num_proc, remove_columns=original_columns)
        eval_dataset = eval_dataset.filter(lambda x: len(
            x["input_ids_j"]) <= 512 )
        print("eval_dataset: ", len(eval_dataset))

        return train_dataset, eval_dataset
