from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import random
import os
from tqdm import tqdm

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        sequence_list = content.split()
    return sequence_list

def get_training_corpus(dataset, chunk_size=1000):
    for i in range(0, len(dataset), chunk_size):
        yield dataset[i : i + chunk_size]

def split_large_file(input_file, output_dir, chunk_size:int =100000000):
    print(f"开始分隔文件{input_file}，并且保存临时文件夹{output_dir}")
    with open(input_file, 'r') as file:
        content = file.read()

    total_chunks = int(len(content) / chunk_size) + 1
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in tqdm(enumerate(chunks), total=total_chunks, desc="Processing Chunks", unit="chunk"):
        output_file = os.path.join(output_dir, f"chunk_{i + 1}.txt")
        with open(output_file, 'w') as output:
            output.write(chunk)

def train_tokenizer(tokenizer, lines, trainer, report_interval=10000):
    line_count = 0
    chunk = []
    for line in lines:
        line_count += 1
        chunk.append(line.strip())
        if len(chunk) == report_interval:
            tokenizer.train_from_iterator(chunk, trainer=trainer)
            chunk = []
    if chunk:
        tokenizer.train_from_iterator(chunk, trainer=trainer)

def train_tokenizer_on_chunks(tokenizer, chunks_dir, trainer, report_interval=10000):
    total_lines = sum(1 for file_name in os.listdir(chunks_dir) if file_name.endswith(".txt"))
    line_count = 0

    for file_name in tqdm(os.listdir(chunks_dir), desc="Processing Files", unit="file"):
        if file_name.endswith(".txt"):
            file_path = os.path.join(chunks_dir, file_name)
            with open(file_path, 'r') as file:
                lines = [line.strip() for line in file]
                train_tokenizer(tokenizer, lines, trainer=trainer, report_interval=report_interval)
                line_count += len(lines)

def save_tokenizer(tokenizer, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)

def load_tokenizer(file_path):
    return Tokenizer.from_file(file_path)

def main(data_path:str = "../..//Datasets/Human_genome/huixin/24_chromosomes-002-clean.txt",
         save_json_path:str = "./save_json/24_chromosomes-002.json",
         special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
         vocab_size:int = 2 ** 12,
         chunk_size:int = 10000000):
    # 初始化
    tokenizer = Tokenizer(models.BPE())

    # pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 加载数据集
    
    # dataset = read_txt_file(data_path)
    # 训练分词器
    print(f"增加的特殊token有：{special_tokens}")
    print(f"增加了一下指导性设定：字典的大小：{vocab_size}")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=False,
    )
    # 分割大文件
    output_dir = "./chunks"
    split_large_file(data_path, output_dir, chunk_size)

    # 训练分词器
    chunks_dir = output_dir
    train_tokenizer_on_chunks(tokenizer, chunks_dir, trainer)
    # train_tokenizer(tokenizer, data_path, trainer, chunk_size)
    # 保存tokenizer在本地
    save_tokenizer(tokenizer, save_json_path)
    
    os.remove(f"{output_dir}/.txt")

if __name__ == "__main__":
    main()
