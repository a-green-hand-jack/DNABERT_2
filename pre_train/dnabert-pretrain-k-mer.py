import transformers
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    RobertaTokenizerFast,
    BertForMaskedLM,
    Trainer,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    BertTokenizer,
    BertForPreTraining, 
    BertTokenizerFast
)
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
# from transformers.trainer_pt_utils import LabelSmoother
from tokenizers import Tokenizer, models, pre_tokenizers
from typing import Optional, Dict, Sequence, Tuple, List, Union, Any
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, IterableDataset, ConcatDataset
import dataclasses
from dataclasses import dataclass, field
import os
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import random

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str , high_tokenizer: PreTrainedTokenizer, low_tokenizer: PreTrainedTokenizer, high_len:int, low_len:int,):
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

        self.high_tokenizer = high_tokenizer
        self.low_tokenizer = low_tokenizer
        self.high_len = high_len
        self.low_len = low_len

        print(f"Length of the dataset is {len(self.lines)}")
        
    def insert_spaces(self, dna_sequence, interval):
        # 初始化结果字符串
        result = ""
        
        # 遍历 DNA 序列
        for i, base in enumerate(dna_sequence):
            # 每隔指定间隔插入一个空格
            if i % interval == 0 and i != 0:
                result += " "
            # 添加当前碱基
            result += base
        
        return result

    def combine_tensors(self, tensor1, tensor2):
        """将两个张量组合成一个张量，并在最后添加一个额外的值作为分隔符"""
        split_point = tensor1.numel()
        extra_value = torch.tensor([split_point])  # 分割点作为额外的值
        combined_tensor = torch.cat((tensor1.flatten(), tensor2.flatten(), extra_value), dim=0)
        return combined_tensor

    def subtract_value_for_values(self, tensor, value_to_subtract):
        """对张量中小于等于 value_to_subtract 的值进行减法操作"""
        subtracted_tensor = tensor.clone()  # 复制输入张量，以避免修改原始张量
        
        # 使用 torch.where() 函数将小于等于 value_to_subtract 的值替换为相应的减法结果
        subtracted_tensor = torch.where(subtracted_tensor <= 5, 
                                        subtracted_tensor - value_to_subtract, 
                                        subtracted_tensor)
        
        return subtracted_tensor

    def tokenize_sequence(self, tokenizer_high, tokenizer_low, sequence, high_token_len=6, low_token_len=3):
        # 计算两个tokenizer分词后的token数量
        tokens_high = tokenizer_high.tokenize(self.insert_spaces(sequence, high_token_len))
        tokens_low = tokenizer_low.tokenize(self.insert_spaces(sequence, low_token_len))

        # 将token转换为对应的token ID
        high_token_ids = tokenizer_high.convert_tokens_to_ids(tokens_high)
        low_token_ids = tokenizer_low.convert_tokens_to_ids(tokens_low)

        # 将token ID转换为tensor
        high_token_tensor = torch.tensor(high_token_ids, dtype=torch.long)
        low_token_tensor = self.subtract_value_for_values(torch.tensor(low_token_ids, dtype=torch.long), len(tokenizer_high.vocab)) + torch.tensor(len(tokenizer_high.vocab), dtype=torch.long)

        # 使用torch.cat()函数连接两个tensor
        combined_tensor = torch.cat((high_token_tensor, low_token_tensor), dim=0)


        return combined_tensor



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
            raise IndexError(f"Index {index} is out of bounds.")

        high_tokenized_tensor = self.tokenize_sequence(sequence=lines,tokenizer_high=self.high_tokenizer, tokenizer_low=self.low_tokenizer, high_token_len=self.high_len, low_token_len=self.low_len)

        # print("打印 high-level tokenized tensor:\n", high_tokenized_tensor)
        # print("打印 low-level tokenized tensor:\n", low_tokenized_tensor)

        # return {"high-level":{'input_ids': high_tokenized_tensor}, "low-level":{'inputs': low_tokenized_tensor}}
        # return {"high-level": {'input_ids': high_tokenized_tensor}, "low-level": {'input_ids': low_tokenized_tensor}}
        return {"input_ids":high_tokenized_tensor}

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
        instances = [instance['input_ids'] for instance in instances]
        # print(instances)
        # print(self.tokenizer.pad_token_id)
        # import torch
        inputs = pad_sequence(
            instances,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        input_ids, labels, attention_masks = self.mask_tokens(inputs)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_masks,
        )
    
    def mask_tokens(self, inputs: Any) -> Tuple[Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels/attention_mask for masked language modeling: 80% MASK, 10% random, 10%
        original. N-gram not applied yet.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # probability be `1` (masked), however in albert model attention mask `0` means masked, revert the value
        attention_mask = (~masked_indices).float()
        if self.tokenizer._pad_token is not None:
            attention_padding_mask = labels.eq(self.tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=1.0)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens, -100 is default for CE compute

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, attention_mask



# @dataclass
# class ModelArguments:
#     model_type: Optional[str] = field(default="dnabert")
#     n_process: int = field(default=1, metadata={"help":"none"})
#     overwrite_cache: bool = False
#     model_name_or_path: Optional[str] = field(default="../zhihan1996/DNABERT-2-117M")
#     use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
#     lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
#     lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
#     lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
#     lora_target_modules: str = field(default="Wqkv,mlp.wo,dense", metadata={"help": "where to perform LoRA"})
        

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # 模型缓存目录，默认为None
    # cache_dir: Optional[str] = field(default=None)
    # 训练过程的标识名称，默认为"run"
    run_name: str = field(default="run")
    # 优化器类型，默认为"adamw_torch"
    optim: str = field(default="adamw_torch")
    # 模型最大长度，默认为512
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    # 梯度积累的步数，默认为1
    gradient_accumulation_steps: int = field(default=1)
    # 每个设备的训练批次大小，默认为8
    per_device_train_batch_size: int = field(default=8)
    # 每个设备的评估批次大小，默认为16
    per_device_eval_batch_size: int = field(default=16)
    # 训练轮数，默认为1000
    num_train_epochs: int = field(default=1000)
    # 是否使用混合精度训练，默认为False
    fp16: bool = field(default=False)
    # 训练过程中记录日志的步数，默认为100
    logging_steps: int = field(default=1000)
    # 保存模型的步数，默认为500；注意保存模型的步数需要时评估模型的步数的整数倍
    save_steps: int = field(default=1000)
    # 评估模型的步数，默认为500
    eval_steps: int = field(default=100)
    # 评估策略，默认为"steps"
    '''
    No: 表示不进行评估。选择这个选项意味着在训练过程中不会自动对模型进行验证，通常用于当外部验证逻辑被使用或者仅仅想要快速完成训练时。
    Epoch: 每完成一次遍历训练数据集（即一个epoch）之后进行一次评估。这是一个比较常用的策略，因为它允许模型在看过所有训练数据一次后，对其性能进行一次整体评估。
    Steps (前面已提及): 模型会在每个固定步数之后进行评估，这个步数由另一个参数eval_steps定义。这种策略允许在训练过程中更频繁地监控模型的性能，有助于更早地识别和响应模型训练中可能出现的问题。
    在深度学习训练过程中，一个“step”通常指的是使用一个批次（batch）的数据进行一次前向传播和一次反向传播的过程。也就是说，每个step涉及到计算模型在一个批次数据上的损失，然后根据这个损失更新模型的权重。
    因此，当讨论到评估策略为“steps”时，比如设置`eval_steps=100`，意味着每进行100个训练步骤（即处理了100个批次的数据并更新了模型100次）后，模型会在验证集上进行一次评估。这种方式可以让你在训练过程中频繁地监控模型的性能，而不必等到完成一个完整的epoch。
    '''
    evaluation_strategy: str = field(default="steps")
    # 训练结束时是否加载最佳模型，默认为True
    load_best_model_at_end: bool = field(default=True)
    # 是否使用大于比较，默认为True
    greater_is_better: bool = field(default=True)
    # 日志记录策略，默认为"steps"
    logging_strategy: str = field(default="steps")
    # 梯度更新前的预热步数，默认为50
    warmup_steps: int = field(default=50)
    # 权重衰减，默认为0.01
    weight_decay: float = field(default=0.01)
    # 学习率，默认为1e-4
    learning_rate: float = field(default=1e-4)
    # 最大保存模型数，默认为50
    save_total_limit: int = field(default=50)
    # 训练时是否发现未使用的参数，默认为False
    find_unused_parameters: bool = field(default=False)
    # 是否进行模型检查点，默认为False
    checkpointing: bool = field(default=False)
    # 是否将数据加载到固定内存，默认为False
    dataloader_pin_memory: bool = field(default=False)
    # 评估并保存结果，默认为True
    eval_and_save_results: bool = field(default=True)
    # 是否保存模型，默认为False
    save_model: bool = field(default=False)
    # 随机数种子，默认为42
    seed: int = field(default=42)


    
class MLMNetwork(nn.Module):
    def __init__(self, model_name_or_path: str = "bert-base-uncased", cache_dir: str = None):
        super(MLMNetwork, self).__init__()
        print(f"load the pre-train model from {model_name_or_path}")
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
    
@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        logits = model_output["prediction_logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

class MyTrainer(Trainer):
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        # 检查是否存在labels键
        if "labels" in inputs:
            labels = inputs.pop("labels")  # 从inputs中移除labels并保存
        else:
            labels = None
        
        # 使用模型处理inputs并获取输出
        outputs = model(**inputs)

        # 如果存在labels，则根据任务类型调用损失平滑器来计算损失
        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        # 如果不存在labels，则直接从模型输出中获取损失
        else:
            # 从outputs中获取损失
            loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        # 返回损失和outputs（如果需要）
        return (loss, outputs) if return_outputs else loss


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

def print_tokenizer_features(tokenizer):
    # 打印特殊token及其对应的id
    print("Special tokens:")
    print(f"Pad token: {tokenizer.pad_token_id}, {tokenizer.pad_token}")
    print(f"CLS token: {tokenizer.cls_token_id}, {tokenizer.cls_token}")
    print(f"UNK token: {tokenizer.unk_token_id}, {tokenizer.unk_token}")
    print(f"Mask token: {tokenizer.mask_token_id}, {tokenizer.mask_token}")
    print(f"SEP token: {tokenizer.sep_token_id}, {tokenizer.sep_token}")
    vocab_size = len(tokenizer)
    # 打印词汇表大小
    print("词汇表大小:", vocab_size)

    # 计算字典中token对应的最大长度和最小长度、平均长度
    max_token_len = max(len(token) for token in tokenizer.get_vocab().keys())
    min_token_len = min(len(token) for token in tokenizer.get_vocab().keys())
    avg_token_len = sum(len(token) for token in tokenizer.get_vocab().keys()) / vocab_size
    print(f"Maximum token length: {max_token_len}")
    print(f"Minimum token length: {min_token_len}")
    print(f"Average token length: {avg_token_len}")
    
    

def print_processed_data_samples(dataset, data_collator, tokenizer, model,num_samples=3):
    

    # 随机选择num_samples个样本
    sample_indices = random.sample(range(len(dataset)), num_samples)
    print("Randomly selected samples:")
    # 打印模型结构
    print("Show the strucature of the model")
    print(model)
    # 打印嵌入层维度
    embedding_dimension = model.config.hidden_size
    print("model's embedding dimemsion:", embedding_dimension)
    # 计算模型的总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters in the model:", total_params)
    # 计算模型的可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", trainable_params)
    

    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]

        # 打印原始数据
        print(f"\nSample {idx + 1}:")
        print("Original data:", sample)

        # 使用data collator处理数据
        processed_data = data_collator([sample])

        # 打印处理后的数据
        print("Processed data:", processed_data)

        # 解码处理后的数据
        decoded_text = tokenizer.batch_decode(processed_data['input_ids'], skip_special_tokens=True)

        # 打印解码后的文本
        print("Decoded text:", decoded_text)
        
        # 打印model的输出
        output = model(**processed_data)
        print("Out put of the model", output)

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
    # Calculate the total number of characters in a single line
    # Assuming all lines have the same number of characters
    num_chars_per_line = len(lines[0].strip())

    # Calculate the total number of characters in the training and evaluation sets
    train_num_chars = sum(len(line.strip()) for line in train_lines)
    eval_num_chars = sum(len(line.strip()) for line in eval_lines)

    print(f"Total number of characters in a single line: {num_chars_per_line}")
    print(f"Total number of characters in the training set: {train_num_chars}")
    print(f"Total number of characters in the evaluation set: {eval_num_chars}")

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

def load_and_convert_tokenizer(load_path: str) -> PreTrainedTokenizerFast:
    """
    Load a tokenizer from a file and convert it to a PreTrainedTokenizerFast object.

    Args:
        load_path (str): Path to the tokenizer file.

    Returns:
        PreTrainedTokenizerFast: Converted PreTrainedTokenizerFast object.
    """
    # new_tokenizer = Tokenizer.from_file(load_path)
    print(f"load tokenize's vocab.txt from {load_path}")
    tokenizer = BertTokenizer(vocab_file=load_path, do_lower_case=False) # 注意，这里一定要规定`do_lower_case=False`!!!!!
    # print(new_tokenizer.mask_token)
    
    
    return tokenizer

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train DNA BERT model")

    parser.add_argument("--high_model_path", type=str, default="../tokenizer/dnabert-config/bert-config-6/vocab.txt",
                        help="Path to the high-level tokenizer vocabulary file")
    parser.add_argument("--low_model_path", type=str, default="../tokenizer/dnabert-config/bert-config-3/vocab.txt",
                        help="Path to the low-level tokenizer vocabulary file")
    parser.add_argument("--model_path", type=str, default="../tokenizer/dnabert-config/high-low-63-vocab.txt",
                        help="Path to the DNA tokenizer vocabulary file")
    parser.add_argument("--data_path", type=str, default="../../Datasets/Human_genome/huixin/24_chromosomes-002.txt",
                        help="Path to the DNA dataset text file")
    parser.add_argument("--output_dir", type=str, default="./dnabert_6/results",
                        help="Output directory for training results")
    parser.add_argument("--logging_dir", type=str, default="./dnabert_6/logs",
                        help="Logging directory for training logs")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # 加载自己的tokenizer
    high_dna_tokenizer = load_and_convert_tokenizer(args.high_model_path)
    low_dna_tokenizer = load_and_convert_tokenizer(args.low_model_path)
    dna_tokenizer = load_and_convert_tokenizer(args.model_path)
    print_tokenizer_features(dna_tokenizer)

    # 加载DNA 数据集
    train_file_path, eval_file_path = split_txt_file(args.data_path, split_ratio=0.8, random_seed=42)
    dna_train_dataset = LineByLineTextDataset(file_path=train_file_path, high_len=6, low_len=3, high_tokenizer=high_dna_tokenizer, low_tokenizer=low_dna_tokenizer)
    dna_eval_dataset = LineByLineTextDataset(file_path=eval_file_path, high_len=6, low_len=3, high_tokenizer=high_dna_tokenizer, low_tokenizer=low_dna_tokenizer)

    # 使用 data_collator 处理数据
    data_collator = DataCollatorForMLM(tokenizer=dna_tokenizer, mlm=True, mlm_probability=0.15)
    # 使用示例
    
    model = BertForPreTraining.from_pretrained("bert-base-uncased")
    print_processed_data_samples(dna_train_dataset, data_collator, dna_tokenizer, model)

    # 开始训练
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        logging_dir=args.logging_dir,
    )
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dna_train_dataset,
        data_collator=data_collator,
        eval_dataset=dna_eval_dataset,
        compute_metrics=compute_mlm_metrics
    )
    trainer.train()
    # 指定保存路径
    # model_save_path = os.path.join(args.output_dir, "final_model_save")
    # trainer.save_model(model_save_path)
    # 使用 save_pretrained 方法保存模型
    model_save_path = os.path.join(args.output_dir, "final_model_save")
    model.save_pretrained(model_save_path)





