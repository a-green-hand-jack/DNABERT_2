
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
    BertTokenizerFast,
    TrainingArguments,
    BertConfig, 
    # LocalRationalAttention,
    BertModel
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

# from my_bert.modeling_bert import BertForMaskedLM
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
        max_length = self.tokenizer.model_max_length
        if len(instances) > self.tokenizer.model_max_length:
            instances = instances[:, :self.tokenizer.model_max_length]
        instances_list = [instance['input_ids'][:max_length] for instance in instances]
        

        inputs = pad_sequence(
            instances_list,
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

def print_tokenizer_features(tokenizer, tokenizer_name:str='default tokenizer'):
    # 打印特殊token及其对应的id
    print(f"===========================这里打印显示{tokenizer_name}的一些基本情况============================")
    print("Special tokens:")
    print(f"Pad token: {tokenizer.pad_token_id}, {tokenizer.pad_token}")
    print(f"CLS token: {tokenizer.cls_token_id}, {tokenizer.cls_token}")
    print(f"UNK token: {tokenizer.unk_token_id}, {tokenizer.unk_token}")
    print(f"Mask token: {tokenizer.mask_token_id}, {tokenizer.mask_token}")
    print(f"SEP token: {tokenizer.sep_token_id}, {tokenizer.sep_token}")
    vocab_size = len(tokenizer)
    # 打印词汇表大小
    print("tokenizer 的词汇表大小是:", vocab_size)

    # 计算字典中token对应的最大长度和最小长度、平均长度
    max_token_len = max(len(token) for token in tokenizer.get_vocab().keys())
    min_token_len = min(len(token) for token in tokenizer.get_vocab().keys())
    avg_token_len = sum(len(token) for token in tokenizer.get_vocab().keys()) / vocab_size
    print(f"Maximum token length: {max_token_len}")
    print(f"Minimum token length: {min_token_len}")
    print(f"Average token length: {avg_token_len}")
    

def initialize_custom_bert_model(low_dna_tokenizer, high_dna_tokenizer, pre_model_config_path):
    # 创建一个新的 BertConfig 对象，并从预训练的模型路径加载配置
    dna_bert_6config = BertConfig.from_pretrained(pre_model_config_path)
    
    # 计算新的词汇表大小并将其分配给vocab_size属性
    new_vocab_size = len(low_dna_tokenizer) + len(high_dna_tokenizer)
    dna_bert_6config.vocab_size = new_vocab_size

    # 打印修改后的配置
    print("Modified config:", dna_bert_6config)
    
    # 根据修改后的配置创建 BertForMaskedLM 模型
    model = BertForMaskedLM(dna_bert_6config)

    # 将模型的权重参数初始化为随机值
    model.init_weights()

    return model


def print_processed_data_samples(dataset, data_collator, tokenizer, model,num_samples=3):
    

    # 随机选择num_samples个样本
    sample_indices = random.sample(range(len(dataset)), num_samples)
    # 打印model的全部设定
    print('=============打印model的全部设定，也就是config==========================')
    print(model.config)

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
    # 打印模型使用的字典数量
    num_embeddings = model.config.vocab_size
    print("Number of embeddings in the model's vocabulary:", num_embeddings)

    for i, idx in enumerate(sample_indices):
        print('*************打印第{}个例子的情况***************'.format(i))
        sample = dataset[idx]

        # 打印原始数据
        print(f"\nSample {idx + 1}:")
        print("\n Original data:\n", sample)

        # 使用data collator处理数据
        processed_data = data_collator([sample])

        # 打印处理后的数据
        print("\n Processed data:\n", processed_data)

        # 解码处理后的数据
        decoded_text = tokenizer.batch_decode(processed_data['input_ids'], skip_special_tokens=True)

        # 打印解码后的文本
        print("\n Decoded text:\n", decoded_text)
        
        # 打印embedding（嵌入后的文本）
        # Get embeddings
        with torch.no_grad():
            print("\n Embedding text shape:\n")
            # 方法0：失败，因为BertForMaskedLM对象没有名为embeddings的属性
            # embedding_token = model.embeddings(input_ids=processed_data['input_ids'])
            # 方法1：使用模型的bert属性
            bert_model = model.bert
            embedding_output = bert_model.embeddings(input_ids=processed_data['input_ids'])

            # 方法2：使用模型的get_input_embeddings()方法
            # embedding_layer = model.get_input_embeddings()
            # embedding_output = embedding_layer(processed_data['input_ids'])
            print(embedding_output.shape)
            
        # 打印model的输出
        output = model(**processed_data)
        print("\n Out put of the model:\n", output)

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
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Evaling batch size per device")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # 加载自己的tokenizer
    high_dna_tokenizer = load_and_convert_tokenizer(args.high_model_path)
    print_tokenizer_features(high_dna_tokenizer, "high dna tokenizer")
    low_dna_tokenizer = load_and_convert_tokenizer(args.low_model_path)
    print_tokenizer_features(low_dna_tokenizer, "low dna tokenizer")
    dna_tokenizer = load_and_convert_tokenizer(args.model_path)
    print_tokenizer_features(dna_tokenizer, "dna tokenizer")

    # 加载DNA 数据集
    train_file_path, eval_file_path = split_txt_file(args.data_path, split_ratio=0.8, random_seed=42)
    dna_train_dataset = LineByLineTextDataset(file_path=train_file_path, high_len=6, low_len=3, high_tokenizer=high_dna_tokenizer, low_tokenizer=low_dna_tokenizer)
    dna_eval_dataset = LineByLineTextDataset(file_path=eval_file_path, high_len=6, low_len=3, high_tokenizer=high_dna_tokenizer, low_tokenizer=low_dna_tokenizer)

    # 使用 data_collator 处理数据
    data_collator = DataCollatorForMLM(tokenizer=dna_tokenizer, mlm=True, mlm_probability=0.15)
    # 使用示例

    dna_bert_6_config_path = '../wjk2002/DNA-BERT-6'
    bert_base_config_path = '../wjk2002/bert-base-uncased'
    model = initialize_custom_bert_model(low_dna_tokenizer=low_dna_tokenizer, high_dna_tokenizer=high_dna_tokenizer, pre_model_config_path=dna_bert_6_config_path)

    print_processed_data_samples(dna_train_dataset, data_collator, dna_tokenizer, model,1)

    # 开始训练
    training_args = TrainingArguments(
        run_name="run",
        optim="adamw_torch",
        gradient_accumulation_steps=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        logging_strategy="steps",
        logging_steps=1000,
        logging_dir=args.logging_dir,
        evaluation_strategy="steps",    # please select one of ['no', 'steps', 'epoch']
        eval_steps=100000,
        load_best_model_at_end=True,
        greater_is_better=True,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=1e-4,
        save_total_limit=50,
        dataloader_pin_memory=False,
        seed=42,
        save_steps=100000
        
    )

    trainer = Trainer(
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





