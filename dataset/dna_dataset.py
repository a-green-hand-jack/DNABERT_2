import concurrent.futures
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer

# 定义读取txt文件的函数
def load_dna_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = [line.strip() for line in file.readlines()]
    return Dataset.from_dict({"sequence": sequences})

# 指定保存的tokenizer的路径
tokenizer_path = '../tokenizer/save_json/config.json'
print(f"从{tokenizer_path}中加载tokenizer设定")
# 加载tokenizer
tokenizer = Tokenizer.from_file(tokenizer_path)

# 指定txt文件路径
txt_file_path:str = "../../Datasets/Human_genome/huixin/24_chromosomes-002-clean.txt"
print(f"从{txt_file_path}中加载语料库")
# 加载DNA序列数据集
dna_dataset = load_dna_sequences(txt_file_path)
# 打印前几个示例
print(dna_dataset['sequence'][:5])

# 定义分词函数
def tokenize_sequence(sequence):
    return tokenizer.encode(sequence)

# 使用 ThreadPoolExecutor 并行处理分词
with concurrent.futures.ThreadPoolExecutor() as executor:
    tokenized_sequences = list(executor.map(tokenize_sequence, dna_dataset['sequence'][:5]))

# 打印分词后的前几个示例
print(tokenized_sequences[:5])
