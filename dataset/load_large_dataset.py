from transformers import LineByLineTextDataset
from transformers import RobertaTokenizerFast
from datasets import load_dataset
import itertools
from torch.utils.data import random_split

tokenizer_path = "../tokenizer/save_tokenizer_small"
txt_file_path = "../../../Datasets/Human_genome/huixin/24_chromosomes-002-small.txt"
print(f"从{tokenizer_path}中加载tokenizer")
print(f"从{txt_file_path}中加载数据")
# 现在在teansformers中重新创建tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=512)

# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="../../../Datasets/Human_genome/huixin/24_chromosomes-002-small.txt",
#     block_size=128,
# )

# # 获取数据集中的前几个示例
# num_examples_to_check = 1
# for i in range(num_examples_to_check):
#     example = dataset[i]
#     print(example)
#     tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
#     decoded_text = tokenizer.decode(example['input_ids'], skip_special_tokens=False ,return_tensor='pt')
#     print("Tokenized Text: ", tokens)
#     print("After decoded text: ",decoded_text)

pubmed_dataset_streamed = load_dataset(
    "text", data_files=txt_file_path, streaming=True
)
print(pubmed_dataset_streamed)
print(next(iter(pubmed_dataset_streamed)))
tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
print(next(iter(tokenized_dataset)))

# dataset_head = pubmed_dataset_streamed.take(5)
# print(list(dataset_head))
# 通过 islice 获取前几个样本
# 获取数据集的一个子集
dataset = pubmed_dataset_streamed['train']

# 划分数据集
dataset_len = dataset.info['start'] - dataset.info['end']
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# 获取前几个样本
num_samples = 5
samples = list(itertools.islice(dataset, num_samples))

# 打印样本内容
for sample in samples:
    print(sample)