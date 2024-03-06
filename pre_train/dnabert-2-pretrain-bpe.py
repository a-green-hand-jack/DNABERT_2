#!/usr/bin/env python
# coding: utf-8

import torch 
import transformers
from transformers import AutoTokenizer, EvalPrediction
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import BertForMaskedLM, Trainer, TrainingArguments
from torch.utils.data import random_split
import re
import os
import csv
import copy
import json
import logging
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import transformers
from transformers import BertForMaskedLM
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np
from torch.utils.data import Dataset
from scipy.special import softmax
from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    AdaLoraModel
)
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        sequence_list = content.split()

    return sequence_list

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")
    #model_name_or_path: Optional[str] = field(default="google-bert/bert-base-uncased")
    #model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    #model_name_or_path: Optional[str] = field(default="decapoda-research/llama-7b-hf")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="Wqkv,mlp.wo,dense", metadata={"help": "where to perform LoRA"})
        

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=16)
    num_train_epochs: int = field(default=1000)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    evaluation_strategy: str = field(default="steps")
    load_best_model_at_end: bool = field(default=True)     # load the best model when finished training (default metric is loss)
    metric_for_best_model: str = field(default="matthews_correlation") # the metric to use to compare models
    greater_is_better: bool = field(default=True)           # whether the `metric_for_best_model` should be maximized or not
    logging_strategy: str = field(default="steps")  # Log every "steps"
    logging_steps: int = field(default=100)  # Log every 100 steps
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=50)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="/common/zhanh/cardioNet/output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)

class DNADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
class MLMNetwork(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", cache_dir=None):
        super(MLMNetwork, self).__init__()
        self.base_model = BertForMaskedLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, 
                                  attention_mask=attention_mask, 
                                  labels=labels)
        return outputs
    @property
    def device(self):
        return next(self.parameters()).device
    
def evaluate_mlm(model, tokenizer, data_collator, eval_dataset):
    model.eval()
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)
    
    total_accuracy = 0
    total_loss = 0
    total_examples = 0

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
            predictions = torch.argmax(outputs.logits, dim=-1)
            labels = batch['labels'].to(model.device)
            mask = labels != -100  # Only consider masked tokens for accuracy
            
            # Accuracy
            correct_predictions = (predictions == labels) & mask
            total_accuracy += correct_predictions.sum().item()
            total_examples += mask.sum().item()

            # Loss for perplexity
            loss = outputs.loss
            total_loss += loss.item() * batch['input_ids'].shape[0]  # Multiply by batch size

    # Calculate metrics
    accuracy = total_accuracy / total_examples
    perplexity = torch.exp(torch.tensor(total_loss / len(eval_dataset)))

    return accuracy, perplexity.item()


def compute_mlm_metrics(eval_preds):
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = logits.argmax(dim=-1)
    
    # Mask positions where labels are -100 (i.e., padding tokens)
    masked_positions = (labels != -100)

    # Calculate accuracy only for the masked positions
    masked_accuracy = (predictions[masked_positions] == labels[masked_positions]).float().mean().item()

    # Calculate perplexity using cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).view(logits.size())

    # Mask out padding positions
    loss = loss * (masked_positions.float())

    # Calculate perplexity for the masked positions
    masked_perplexity = torch.exp(loss.sum() / masked_positions.float().sum()).item()

    return {
        'masked_accuracy': masked_accuracy,
        'masked_perplexity': masked_perplexity,
    }



if __name__ == "__main__":

    model_args = ModelArguments()

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=100,
        per_device_train_batch_size=8,
        logging_dir='./logs',
    )

    # tokenizer初始化移到训练循环内
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    input_file_path = "../../Dataset/Human_genome/huixin/24_chromosomes-002.txt"
    sequences = read_txt_file(input_file_path)

    dataset = DNADataset(sequences)  # 你可能需要根据你的 DNADataset 类的实现进行修改

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    model = MLMNetwork(model_name_or_path=model_args.model_name_or_path)

    # 打印模型结构
    # print(model)

    # 计算模型参数数量
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_parameters}")
    # 打印分词器信息
    print(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        compute_metrics=compute_mlm_metrics
    )

    # 在训练循环中逐步加载数据
    for epoch in range(training_args.num_train_epochs):
        for batch in DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True):
            tokenized_inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            trainer.train_step(model_inputs=tokenized_inputs)

    # 评估模型
    accuracy, perplexity = evaluate_mlm(model, tokenizer, data_collator, eval_dataset)

    print(f"Accuracy of predicting masked tokens: {accuracy:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


