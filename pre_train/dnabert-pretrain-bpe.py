#!/usr/bin/env python
# coding: utf-8
import torch 
import transformers
from transformers import AutoTokenizer, EvalPrediction, RobertaTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments, PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from functools import partial
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
# from datasets import Dataset
from tokenizers import Tokenizer
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import random
import os


class DNADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}



# def tokenize_and_concat_dataset(dna_tokenizer, text_data, chunk_size=int(1e8), max_length=512):
#     # Initialize an empty list to store individual datasets
#     tokenized_datasets = []

#     # Calculate the number of chunks
#     num_chunks = (len(text_data) + chunk_size - 1) // chunk_size

#     # Use tqdm for a progress bar
#     for i in tqdm(range(num_chunks), desc="Tokenizing dataset"):
#         start_idx = i * chunk_size
#         end_idx = min((i + 1) * chunk_size, len(text_data))
#         chunk = text_data[start_idx:end_idx]

#         # Tokenize the chunk using dna_tokenizer
#         tokenized_chunk = dna_tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

#         # Create a DNADataset from the tokenized chunk
#         dna_dataset_chunk = DNADataset(tokenized_chunk)

#         # Append the dataset to the list
#         tokenized_datasets.append(dna_dataset_chunk)

#     # Concatenate all the datasets into a single large dataset
#     concatenated_dataset = ConcatDataset(tokenized_datasets)

#     return concatenated_dataset


def tokenize_and_concat_dataset(dna_tokenizer, text_data, num_chunks, max_length=512):
    # Create a temporary folder to store cached features
    cache_folder = "./cached_features_file"
    os.makedirs(cache_folder, exist_ok=True)

    # Initialize an empty list to store individual datasets
    
    text_data_len = len(text_data)
    # Tokenize and save each chunk to the temporary folder
    for i in tqdm(range(num_chunks), desc="Tokenizing dataset"):
        start_idx = i * text_data_len // num_chunks
        end_idx = (i + 1) * text_data_len // num_chunks
        chunk = text_data[start_idx:end_idx]

        # Tokenize the chunk using dna_tokenizer
        tokenized_chunk = dna_tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        # Save the tokenized chunk to a file
        filename = os.path.join(cache_folder, f"chunk_{i}.pt")
        torch.save(tokenized_chunk, filename)

    # Load and concatenate all the datasets from the temporary folder
    tokenized_datasets = []
    for i in tqdm(range(num_chunks), desc="Loading tokenized datasets"):
        filename = os.path.join(cache_folder, f"chunk_{i}.pt")
        tokenized_chunk = torch.load(filename)

        # Create a DNADataset from the tokenized chunk
        dna_dataset_chunk = DNADataset(tokenized_chunk)

        # Append the dataset to the list
        tokenized_datasets.append(dna_dataset_chunk)

    # Concatenate all the datasets into a single large dataset
    concatenated_dataset = ConcatDataset(tokenized_datasets)

    return concatenated_dataset

def split_txt_file(input_file_path, split_ratio=0.8, random_seed=42):
    # Set the random seed for reproducibility
    random.seed(random_seed)
    # Read lines from the input file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Calculate the split index
    split_index = int(len(lines) * split_ratio)

    # Split the lines into train and eval sets
    train_lines = lines[:split_index]
    eval_lines = lines[split_index:]
    print(f"训练集中含有的碱基的长度{len(train_lines) * 512}")
    print(f"验证集中含有的碱基的长度{len(eval_lines) * 512}")

    # Define output file paths
    train_file_path = os.path.join(os.path.dirname(input_file_path), 'train.txt')
    eval_file_path = os.path.join(os.path.dirname(input_file_path), 'eval.txt')

    # Write train lines to train file
    with open(train_file_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_lines)

    # Write eval lines to eval file
    with open(eval_file_path, 'w', encoding='utf-8') as eval_file:
        eval_file.writelines(eval_lines)

    return train_file_path, eval_file_path

class ChunkedDNADataset(IterableDataset):
    def __init__(self, tokenizer, text_data, chunk_size=1000, max_length=512):
        self.tokenizer = tokenizer
        self.text_data = text_data
        self.chunk_size = chunk_size
        self.max_length = max_length

        # Calculate the total number of chunks
        self.num_chunks = (len(self.text_data) + self.chunk_size - 1) // self.chunk_size

    def __iter__(self):
        for start_idx in tqdm(range(0, len(self.text_data), self.chunk_size), desc="Tokenizing dataset", total=self.num_chunks):
            end_idx = min(start_idx + self.chunk_size, len(self.text_data))
            chunk = self.text_data[start_idx:end_idx]

            # Tokenize the chunk using dna_tokenizer
            tokenized_chunk = self.tokenizer(chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

            # Create a DNADataset from the tokenized chunk
            dna_dataset_chunk = DNADataset(tokenized_chunk)

            yield dna_dataset_chunk



@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="dnabert")
    n_process: int = field(default=1, metadata={"help":"none"})
    overwrite_cache: bool = False
    model_name_or_path: Optional[str] = field(default="../zhihan1996/DNABERT-2-117M")
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
    # metric_for_best_model: str = field(default="matthews_correlation") # the metric to use to compare models
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
    
def evaluate_mlm(model, data_collator, eval_dataset):
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


def compute_mlm_metrics(p):
    predictions = p.predictions.argmax(axis=2)  # 选择预测的最高概率的标签
    labels = p.label_ids

    # 计算准确率
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())

    # 计算精确度、召回率、F1 分数
    precision, recall, f1, _ = precision_recall_fscore_support(labels.flatten(), predictions.flatten(), average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def load_and_split_dataset(txt_file_path, split_ratio=0.2, random_seed=42):
    # Example usage:
    train_file, eval_file = split_txt_file(txt_file_path, split_ratio, random_seed)
    print(f"Train file created: {train_file}")
    print(f"Eval file created: {eval_file}")
    # 加载数据集
    data_files = {"train": train_file, "test": eval_file}
    dataset = load_dataset("text", data_files=data_files)
    # 转化为PyTorch Dataset
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    return train_dataset, eval_dataset


def load_and_convert_tokenizer(load_path):
    new_tokenizer = Tokenizer.from_file(load_path)
    # print(new_tokenizer.mask_token)
    
    transformer_tokenizer = PreTrainedTokenizerFast(tokenizer_object=new_tokenizer, 
                                                    mask_token = "[MASK]", 
                                                    unk_token = '[UNK]', 
                                                    pad_token = '[PAD]', 
                                                    sep_token = '[SEP]', 
                                                    cls_token = '[CLS]', 
                                                    padding_site='right')
    return transformer_tokenizer

if __name__ == "__main__":

    model_args = ModelArguments()

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_dir='./logs',
    )

    # 替换成你的文件路径和其他参数
    txt_file_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002-small.txt"
    train_dataset, eval_dataset = load_and_split_dataset(txt_file_path, split_ratio=0.99, random_seed=42)

    # 加载自己的tokenizer
    model_name_or_path = "../tokenizer/save_json/config.json"
    dna_tokenizer = load_and_convert_tokenizer(model_name_or_path)

    # 这里是不使用迭代的方法
    # tokenized_train_dataset = dna_tokenizer(train_dataset['text'], padding=True, truncation=True, max_length=512, return_tensors="pt")
    # # print(tokenized_train_dataset)
    # dna_train_dataset = DNADataset(tokenized_train_dataset)
    # tokenized_eval_dataset = dna_tokenizer(eval_dataset['text'], padding=True, truncation=True, max_length=512, return_tensors="pt")
    # # print(tokenized_eval_dataset)
    # dna_eval_dataset = DNADataset(tokenized_eval_dataset)
    # 这里是不使用迭代的方法

    # 这里是函数分块
    dna_train_dataset = tokenize_and_concat_dataset(dna_tokenizer=dna_tokenizer, text_data=train_dataset['text'])
    dna_eval_dataset = tokenize_and_concat_dataset(dna_tokenizer=dna_tokenizer, text_data=eval_dataset['text'])
    # 这里是函数分块

    # 这里是class 迭代
    # dna_train_dataset = ChunkedDNADataset(tokenizer=dna_tokenizer, text_data=train_dataset['text'], chunk_size=1e8, max_length=512)
    # dna_eval_dataset = ChunkedDNADataset(tokenizer=dna_tokenizer, text_data=eval_dataset['text'], chunk_size=1e8, max_length=512)
    # 这里是class 迭代
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=dna_tokenizer, mlm=True, mlm_probability=0.15)

    # 加载model
    model = MLMNetwork(model_name_or_path=model_args.model_name_or_path)

    # 开始训练
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dna_train_dataset,
    data_collator=data_collator,
    eval_dataset=dna_eval_dataset,
    compute_metrics=compute_mlm_metrics
    )
    trainer.train()

    # # 评估模型
    accuracy, perplexity = evaluate_mlm(model, data_collator, dna_eval_dataset)
    print(f"Accuracy of predicting masked tokens: {accuracy:.4f}")
    print(f"Perplexity: {perplexity:.4f}")




