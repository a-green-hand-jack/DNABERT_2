from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


# 加载一个分词器
# 自定义 ByteLevelBPETokenizer
# tokenizer = ByteLevelBPETokenizer(
#     "./save_tokenizer/vocab.json",
#     "./save_tokenizer/merges.txt",
# )

# 自定义 BertProcessing 后处理器
# post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )

# # 设置 ByteLevelBPETokenizer 的后处理器
# tokenizer.post_processor = post_processor

# 启用截断
# tokenizer.enable_truncation(max_length=512)

# DNA序列
# dna_sequence = "ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"


# print(tokenizer.encode(dna_sequence))

# print(tokenizer.encode(dna_sequence).tokens)

# 现在在teansformers中重新创建tokenizer
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./save_tokenizer_small", max_len=512)

# 加载一个数据集
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../../../Datasets/Human_genome/huixin/24_chromosomes-002-small.txt",
    block_size=128,
)
# 验证数据集的情况
# 获取数据集中的前几个示例
num_examples_to_check = 1
for i in range(num_examples_to_check):
    example = dataset[i]
    print(example)
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    decoded_text = tokenizer.decode(example['input_ids'], skip_special_tokens=False ,return_tensor='pt')
    # 打印示例中的分词
    # print(f"Example {i + 1}:")
    # print("Original Text:", example["text"])
    print("Tokenized Text: ", tokens)
    print("After decoded text: ",decoded_text)
    # print()

# 定义一个 data_collator
# 创建一个小的示例文本
dna_sequence = "ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"
# 这是一个小助手，这将帮助我们将数据集的不同样本批处理到一个Pytorch知道如何执行的backprop中.
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# 确保 block_size 是 tokenizer 的 block_size
block_size = 128
# 使用 data_collator 处理数据
processed_data = data_collator([tokenizer.encode(dna_sequence, max_length=block_size, truncation=True)])
# 输出处理后的数据
print(processed_data)

# 解码处理后的数据
decoded_text = tokenizer.batch_decode(processed_data['input_ids'])[0]

# 打印结果
print(decoded_text)
