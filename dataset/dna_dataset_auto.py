import concurrent.futures
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling

# 定义读取txt文件的函数
def load_dna_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = [line.strip() for line in file.readlines()]
    return Dataset.from_dict({"sequence": sequences})

class DNADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

# 指定保存的tokenizer的路径
pretrained_model_path = '../tokenizer/save_tokenizer'
print(f"从{pretrained_model_path}中加载tokenizer设定")
# 加载tokenizer
# tokenizer = Tokenizer.from_file(tokenizer_path)
auto_tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_path)


# 指定txt文件路径
txt_file_path:str = "../../Datasets/Human_genome/huixin/24_chromosomes-002-1.0.txt"
print(f"从{txt_file_path}中加载语料库")
# 加载DNA序列数据集
dna_dataset = load_dna_sequences(txt_file_path)
# 打印前几个示例
# print(dna_dataset['sequence'][:5])

