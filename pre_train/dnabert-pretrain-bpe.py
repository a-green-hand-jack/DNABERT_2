#!/usr/bin/env python
# coding: utf-8
import os
import random
from typing import Optional, Dict, Sequence, Tuple, List, Union, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, IterableDataset, ConcatDataset
from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

import transformers
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    RobertaTokenizerFast,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling
)

from datasets import load_dataset
from tokenizers import Tokenizer
from dataclasses import dataclass, field

from multiprocessing import Pool
import pickle


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str = "", block_size: int = 512, batch_size: int = 8):
        """
        Initializes the LineByLineTextDataset.

        Args:
            tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenization.
            file_path (str): The path to the dataset file.
            block_size (int): The block size for tokenization.
            batch_size (int): The batch size for processing.
        """
        assert os.path.isfile(file_path)
        print("Creating features from dataset file at %s" % file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.current_index = 0
        print(f"Length of the dataset is {len(self.lines)}")

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.lines)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            torch.Tensor: The encoded tensor of the input text.
        """
        try:
            lines = self.lines[index]
        except IndexError:
            print(f"IndexError: list indexout of range, and the max index should be {len(self.lines)} and the index is {index}")
            raise

        if not lines:
            raise IndexError(f"Index {self.current_index} is out of bounds.")

        self.current_index += self.block_size

        encoded = self.tokenizer(lines, add_special_tokens=True, truncation=True)['input_ids']
        tensor_encoded = torch.tensor(encoded, dtype=torch.long)

        return {'input_ids': tensor_encoded}

@dataclass
class DataCollatorForMLM(DataCollatorForLanguageModeling):
    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for masked language modeling.

        Args:
            instances (Sequence[Dict[str, torch.Tensor]]): List of instances containing input tensors.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing input_ids, labels, and attention_mask tensors.
        """

        input_ids, labels = self.mask_tokens(instances)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def mask_tokens(
        self, instances: Sequence[Dict[str, torch.Tensor]], mlm_probability: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens for masked language modeling.

        Args:
            instances (Sequence[Dict[str, torch.Tensor]]): List of instances containing input tensors.
            mlm_probability (float): Probability of masking tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing input_ids and labels tensors.
        """
        input_ids = pad_sequence(
            [instance['input_ids'] for instance in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = input_ids.clone()

        probability_matrix = torch.full(input_ids.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels


class MLMNetwork(nn.Module):
    def __init__(self, model_name_or_path: str = "bert-base-uncased", cache_dir: str = None):
        super(MLMNetwork, self).__init__()
        self.base_model = BertForMaskedLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass of the MLMNetwork model.

        Args:
            input_ids: Input tensor of token ids.
            attention_mask: Optional tensor for attention mask.
            labels: Optional tensor for masked language modeling labels.
            **kwargs: Additional keyword arguments.

        Returns:
            outputs: Model outputs from the base model.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    @property
    def device(self):
        """
        Property to get the device of the model parameters.

        Returns:
            device: Device of the model parameters.
        """
        return next(self.parameters()).device
    

def split_txt_file(input_file_path: str, split_ratio: float = 0.8, random_seed: int = 42) -> Tuple[str, str]:
    """
    Split a text file into train and eval sets based on the split ratio.

    Args:
        input_file_path (str): Path to the input text file.
        split_ratio (float): Ratio to split the data into train and eval sets.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[str, str]: Paths to the train and eval text files.
    """
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
    print(f"Length of bases in the training set: {len(train_lines) * 512}")
    print(f"Length of bases in the evaluation set: {len(eval_lines) * 512}")

    # Define output file paths
    train_file_path = os.path.join(os.path.dirname(input_file_path), 'train.txt')
    eval_file_path = os.path.join(os.path.dirname(input_file_path), 'eval.txt')

    # Write train lines to train file
    with open(train_file_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_lines)

    # Write eval lines to eval file
    with open(eval_file_path, 'w', encoding='utf-8') as eval_file:
        eval_file.writelines(eval_lines)
    print(f"Train set saved at: {train_file_path}")
    print(f"Evaluation set saved at: {eval_file_path}")

    return train_file_path, eval_file_path


def evaluate_mlm(model, data_collator, eval_dataset):
    """
    Evaluate the masked language model on the evaluation dataset.

    Args:
        model: The masked language model to evaluate.
        data_collator: Data collator for processing batches.
        eval_dataset: Evaluation dataset for evaluation.

    Returns:
        accuracy: Accuracy of the model on the evaluation dataset.
        perplexity: Perplexity of the model on the evaluation dataset.
    """
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
    """
    Compute metrics for masked language model predictions.

    Args:
        p: Prediction object containing predictions and labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    predictions = p.predictions.argmax(axis=2)  # Choose the label with the highest probability
    labels = p.label_ids

    # Calculate accuracy
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels.flatten(), predictions.flatten(), average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def load_and_convert_tokenizer(load_path: str) -> PreTrainedTokenizerFast:
    """
    Load a tokenizer from a file and convert it to a PreTrainedTokenizerFast object.

    Args:
        load_path (str): Path to the tokenizer file.

    Returns:
        PreTrainedTokenizerFast: Converted PreTrainedTokenizerFast object.
    """
    new_tokenizer = Tokenizer.from_file(load_path)
    # print(new_tokenizer.mask_token)

    transformer_tokenizer = PreTrainedTokenizerFast(tokenizer_object=new_tokenizer, 
                                                    mask_token="[MASK]", 
                                                    unk_token='[UNK]', 
                                                    pad_token='[PAD]', 
                                                    sep_token='[SEP]', 
                                                    cls_token='[CLS]', 
                                                    padding_site='right')
    return transformer_tokenizer

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



    
if __name__ == "__main__":

    model_args = ModelArguments()
    batch_size = 32
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        logging_dir='./logs',
    )

    # 加载自己的tokenizer
    model_name_or_path = "../tokenizer/save_json/config.json"
    dna_tokenizer = load_and_convert_tokenizer(model_name_or_path)
    data_collator = DataCollatorForMLM(tokenizer=dna_tokenizer)

    txt_file_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002.txt"
    train_file_path, eval_file_path = split_txt_file(txt_file_path, split_ratio=0.99, random_seed=42)
    dna_train_dataset = LineByLineTextDataset(tokenizer=dna_tokenizer, file_path=train_file_path, batch_size=batch_size)
    dna_eval_dataset = LineByLineTextDataset(tokenizer=dna_tokenizer, file_path=eval_file_path, batch_size=batch_size)

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




