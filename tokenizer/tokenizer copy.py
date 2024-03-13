from tokenizers import (
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
    ByteLevelBPETokenizer
)

import datasets
from datasets import load_dataset
import os
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
            
def train_tokenizer(tokenizer, sequence, trainer, chunk_size=1000, overlap=100):
    start = 0
    end = chunk_size
    chunks = []
    while start < len(sequence):
        chunk = sequence[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        end = min(start + chunk_size, len(sequence))
    chunks_number = len(chunks)
    print(f"分割之后的列表的长度是{chunks_number}")
    tokenizer.train_from_iterator(chunks, trainer=trainer,length=chunks_number)



def train_tokenizer_on_chunks(tokenizer, chunks_dir, trainer, chunk_size=10000000, overlap=512):

    for file_name in tqdm(os.listdir(chunks_dir), desc="Processing Files", unit="file"):
        
        if file_name.endswith(".txt"):
            file_path = os.path.join(chunks_dir, file_name)
            with open(file_path, 'r') as file:
                sequence = file.read().strip()
                tqdm_desc = f"Processing {file_name} (Length: {len(sequence)})"
                tqdm.write(tqdm_desc)  # 更新进度条描述
                train_tokenizer(tokenizer, sequence, trainer, chunk_size, overlap)




def save_tokenizer(tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    print(f"Creating directory: {save_path}")
    save_path = os.path.join(save_path, "config.json")
    tokenizer.save(save_path) # 这种保存方式得到是一个json文件，虽然也包括了一切需要的信息，但是不能直接被RobertaTokenizerFast.from_pretrained 接受,但是可以使用：

    # tokenizer.model.save(save_path) # 这种方法保存是一个merges.txt+vocab.json，可以被RobertaTokenizerFast.from_pretrained直接接受，采用下面这种方法加载：

    

def main(data_path:str = "",
         save_path:str = "",
         special_tokens:list = [],
         vocab_size:int = int(2^8),
         chunk_size:int = int(2^30),
         overlap:int = int(2^12)):
    # 初始化
    tokenizer = Tokenizer(models.BPE())
    # pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 训练分词器
    print(f"增加的特殊token有：{special_tokens}")
    print(f"增加了一下指导性设定：字典的大小：{vocab_size}")
    print(f"一次批次的长度是{chunk_size}")
    print(f"重叠的长度是{overlap}")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=False,
    )
    # 训练分词器
    train_tokenizer_on_chunks(tokenizer, data_path, trainer, chunk_size, overlap)
    # 保存分词器
    save_tokenizer(tokenizer, save_path)


if __name__ == "__main__":
    data_path:str = "../../Datasets/Human_genome/huixin/24-txt/"
    save_path:str = "./my_tokenizer/"
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    vocab_size:int = 2 ** 12
    chunk_size:int = 2 ** 10
    overlap:int = 2 ** 6

    main(data_path=data_path, 
         save_path=save_path, 
         special_tokens=special_tokens, 
         vocab_size=vocab_size, 
         chunk_size=chunk_size,
         overlap=overlap
         )