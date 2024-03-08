import concurrent.futures
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
tokenizer_path = '../tokenizer/save_json/config.json'
print(f"从{tokenizer_path}中加载tokenizer设定")
# 加载tokenizer
tokenizer = Tokenizer.from_file(tokenizer_path)

# 指定txt文件路径
txt_file_path:str = "../../Datasets/Human_genome/huixin/24_chromosomes-002-1.0.txt"
print(f"从{txt_file_path}中加载语料库")
# 加载DNA序列数据集
dna_dataset = load_dna_sequences(txt_file_path)
# 打印前几个示例
# print(dna_dataset['sequence'][:5])

# 定义分词函数
def tokenize_sequence(sequence):
    return tokenizer.encode(sequence)

# 使用 ThreadPoolExecutor 并行处理分词
with concurrent.futures.ThreadPoolExecutor() as executor:
    tokenized_sequences = list(executor.map(tokenize_sequence, dna_dataset['sequence'][:10]))

# 打印分词后的前几个示例
print("="*10, "这里打印分词之后的情况", "="*10)
print(tokenized_sequences[:5])

# 加载预训练的DNA tokenizer
dna_tokenizer = AutoTokenizer.from_pretrained("dna_tokenizer_model_name")

# 将分词后的序列转换为字典的列表，并添加mask标记
tokenized_data = [{"input_ids": seq.ids, "attention_mask": seq.attention_mask, "labels": [-100 if x == 0 else x for x in seq.ids]} for seq in tokenized_sequences]

# 创建MLMDataset类
class MLMDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

# 使用DataCollatorForLanguageModeling对数据集进行padding和mask处理
data_collator = DataCollatorForLanguageModeling(tokenizer=dna_tokenizer, mlm=True, mlm_probability=0.15)
padded_dataset = data_collator(tokenized_data)

# 封装为MLMDataset
mlm_dataset = MLMDataset(padded_dataset)

# 打印处理后的数据集示例
print("="*10, "padding和mask之后的情况", "="*10)
print(mlm_dataset[:5])