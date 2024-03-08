import concurrent.futures
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import os
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    models,
    processors,
    Tokenizer,
)

def load_tokenizer(file_path):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.model = BPE()
    tokenizer.decoder = models.BPE()
    tokenizer.post_processor = processors.BPE()
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(max_length=512)
    tokenizer.from_file(file_path)
    return PreTrainedTokenizerFast(tokenizer)

# 指定 tokenizer 文件路径
tokenizer_path = './save_json/24_chromosomes-002.json'

# 使用 load_tokenizer 函数加载 tokenizer
loaded_tokenizer = load_tokenizer(tokenizer_path)

# 将 loaded_tokenizer 转换为 PreTrainedTokenizerFast 类
transformers_tokenizer = PreTrainedTokenizerFast.from_pretrained(loaded_tokenizer)

# 现在您可以在 transformers 中使用加载的 tokenizer
sequence = "AGGGGGGAAAAAATTTTCCCCCCCCCCC"
encoded_output = transformers_tokenizer(sequence)


# 定义读取txt文件的函数
def load_dna_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = [line.strip() for line in file.readlines()]
    return Dataset.from_dict({"sequence": sequences})



# 指定txt文件路径
# txt_file_path:str = "../../Datasets/Human_genome/huixin/24_chromosomes-002-clean.txt"
# print(f"从{txt_file_path}中加载语料库")
# # 加载DNA序列数据集
# dna_dataset = load_dna_sequences(txt_file_path)
# # 打印前几个示例
# print(dna_dataset['sequence'][:5])

# # 定义分词函数
# def tokenize_sequence(sequence):
#     return transformers_tokenizer.encode(sequence)

# # 使用 ThreadPoolExecutor 并行处理分词
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     tokenized_sequences = list(executor.map(tokenize_sequence, dna_dataset['sequence']))

# # 打印分词后的前几个示例
# print(tokenized_sequences[:5])
