from tokenizers import (
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

from transformers import AutoTokenizer
import os
from tqdm import tqdm

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
    line_count = 0

    for file_name in tqdm(os.listdir(chunks_dir), desc="Processing Files", unit="file"):
        if file_name.endswith(".txt"):
            file_path = os.path.join(chunks_dir, file_name)
            with open(file_path, 'r') as file:
                lines = [line.strip() for line in file]
                train_tokenizer(tokenizer, lines, trainer=trainer, report_interval=report_interval)
                line_count += len(lines)

def save_tokenizer(tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.model.save(save_dir)

def load_tokenizer(file_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.model = models.BPE.from_file(file_path)
    return tokenizer

def main(data_path:str = "../..//Datasets/Human_genome/huixin/24_chromosomes-002-1.0.txt",
         save_dir:str = "./save_tokenizer",
         special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
         vocab_size:int = 2 ** 12,
         chunk_size:int = 10000000):
    # 初始化
    tokenizer = Tokenizer(models.BPE())

    # pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

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
    # 保存tokenizer在本地
    # save_tokenizer(tokenizer, save_dir)
    tokenizer.save_pretrained(save_dir)


    # os.remove(f"{output_dir}")

    # 保存为vocab.txt和merges.txt
    # tokenizer.save(save_dir)

if __name__ == "__main__":
    main()

# 加载tokenizer
tokenizer_path = './save_tokenizer'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)