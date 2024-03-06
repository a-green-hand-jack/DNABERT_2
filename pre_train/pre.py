#!/usr/bin/env python
# coding: utf-8




import torch 
import transformers
from transformers import AutoTokenizer
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





@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNA_bert_3")
    # model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")
    #model_name_or_path: Optional[str] = field(default="google-bert/bert-base-uncased")
    #model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    #model_name_or_path: Optional[str] = field(default="decapoda-research/llama-7b-hf")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="Wqkv,mlp.wo,dense", metadata={"help": "where to perform LoRA"})
        
model_args = ModelArguments()




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

training_args = TrainingArguments()     




#tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

# Simulate tokenization of DNA sequences
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        sequence_list = content.split()

    return sequence_list

input_file_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002-1.0.txt"
sequences = read_txt_file(input_file_path)

# For demonstration, let's treat each nucleotide as a separate 'word' (highly simplified)
tokenized_inputs = tokenizer(sequences, padding=True, truncation=True, max_length=512, return_tensors="pt")





class DNADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

dataset = DNADataset(tokenized_inputs)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])





data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)





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




def compute_mlm_metrics(eval_preds):
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = logits.argmax()
    
    # Mask positions where labels are -100 (i.e., padding tokens)
    masked_positions = (labels != -100)

    # Convert masked_positions to a PyTorch tensor
    masked_positions = torch.tensor(masked_positions, dtype=torch.bool)

    # Print shapes for debugging
    print("logits shape:", logits.shape)
    print("labels shape:", labels.shape)
    print("masked_positions shape:", masked_positions.shape)

    # Ensure masked_positions is a boolean tensor
    masked_positions = torch.nonzero(masked_positions, as_tuple=False).squeeze()

    # Print additional information for debugging
    print("Number of masked positions:", len(masked_positions))
    print("Sample masked positions:", masked_positions[:10])  # Print the first 10 masked positions

    if len(masked_positions) == 0:
        raise ValueError("No masked positions found. Check your data or masking logic.")

    # Calculate accuracy only for the masked positions
    try:
        masked_accuracy = (predictions[masked_positions] == labels[masked_positions]).float().mean().item()
    except Exception as e:
        print("Error:", e)
        print("predictions shape:", predictions.shape)
        print("labels shape:", labels.shape)
        print("masked_positions shape:", masked_positions.shape)
        raise e

    # Calculate other metrics if needed
    # ...

    return {
        'masked_accuracy': masked_accuracy,
        # Add other metrics as needed
    }
model = MLMNetwork(model_name_or_path= model_args.model_name_or_path)




training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset,
    compute_metrics=compute_mlm_metrics
    )




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




trainer.train()
accuracy, perplexity = evaluate_mlm(model, tokenizer, data_collator, eval_dataset)

print(f"Accuracy of predicting masked tokens: {accuracy:.4f}")
print(f"Perplexity: {perplexity:.4f}")