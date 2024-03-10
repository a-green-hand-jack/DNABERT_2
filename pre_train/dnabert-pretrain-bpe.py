#!/usr/bin/env python
# coding: utf-8
import torch 
import transformers
from transformers import AutoTokenizer, EvalPrediction, RobertaTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from functools import partial
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
# from datasets import Dataset

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

class DNADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}



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


def compute_mlm_metrics(p):
    predictions = p.predictions.argmax(axis=2)  # 选择预测的最高概率的标签
    labels = p.label_ids

    # 计算准确率
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())

    # 计算精确度、召回率、F1 分数
    precision, recall, f1, _ = precision_recall_fscore_support(labels.flatten(), predictions.flatten(), average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def load_and_split_dataset(txt_file_path, test_size=0.2, random_state=42):
    # 加载数据集
    data_files = {"train": f"{txt_file_path}/train.txt", "test": f"{txt_file_path}/eval.txt"}
    # pubmed_dataset_streamed = load_dataset("text", data_files=data_files, streaming=True)
    pubmed_dataset_streamed = load_dataset("text", data_files=data_files)

    # 转化为PyTorch Dataset
    train_dataset = pubmed_dataset_streamed['train']
    eval_dataset = pubmed_dataset_streamed['test']

    # ds = Dataset.from_generator(partial(gen_from_iterable_dataset, iterable_ds), features=iterable_ds.features)

    return train_dataset, eval_dataset

def process_example(example, tokenizer):
        # 假设输入数据是example["text"]
        tokenized_inputs = tokenizer.encode(example["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")
        print(tokenized_inputs)
        return tokenized_inputs


if __name__ == "__main__":

    model_args = ModelArguments()

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_dir='./logs',
    )

    # 替换成你的文件路径和其他参数
    txt_file_path = "../../Datasets/Human_genome/huixin"
    tokenizer_path = "../tokenizer/save_tokenizer_small"
    train_dataset, eval_dataset = load_and_split_dataset(txt_file_path, test_size=0.01)
    print(eval_dataset)

    # 加载作者的tokenizer
    # dna_tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     use_fast=True,
    #     trust_remote_code=True,
    # )
    # 加载自己的tokenizer
    model_name_or_paht = "../tokenizer/"
    dna_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # 这里是不使用迭代的方法
    tokenized_train_dataset = dna_tokenizer(train_dataset['text'], padding=True, truncation=True, max_length=512, return_tensors="pt")
    # print(tokenized_train_dataset)
    dna_train_dataset = DNADataset(tokenized_train_dataset)
    tokenized_eval_dataset = dna_tokenizer(eval_dataset['text'], padding=True, truncation=True, max_length=512, return_tensors="pt")
    # print(tokenized_eval_dataset)
    dna_eval_dataset = DNADataset(tokenized_eval_dataset)
    # 这里是不使用迭代的方法
    
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
    accuracy, perplexity = evaluate_mlm(model, dna_tokenizer, data_collator, dna_eval_dataset)
    print(f"Accuracy of predicting masked tokens: {accuracy:.4f}")
    print(f"Perplexity: {perplexity:.4f}")




