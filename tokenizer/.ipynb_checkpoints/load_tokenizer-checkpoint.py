from transformers import LineByLineTextDataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

def load_and_convert_tokenizer(load_path):
    new_tokenizer = Tokenizer.from_file(load_path)
    # print(new_tokenizer.mask_token)
    transformer_tokenizer = PreTrainedTokenizerFast(tokenizer_object=new_tokenizer, mask_token = "[MASK]")
    return transformer_tokenizer

# Example usage:
# save_path = "./tokenizer/save_tokenizer_small"
save_path = "./save_json/config.json"
tokenizer = load_and_convert_tokenizer(save_path)


# tokenizer = RobertaTokenizerFast.from_pretrained("./save_tokenizer_small", max_len=512)

# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="../../Datasets/Human_genome/huixin/24_chromosomes-002-small.txt",
#     block_size=128,
# )
# 验证数据集的情况
# 获取数据集中的前几个示例
# num_examples_to_check = 1
# for i in range(num_examples_to_check):
#     example = "ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"
#     print(example)
#     tokens = tokenizer.convert_ids_to_tokens(example)
#     decoded_text = tokenizer.decode(example, skip_special_tokens=False ,return_tensor='pt')
#     # 打印示例中的分词
#     # print(f"Example {i + 1}:")
#     # print("Original Text:", example["text"])
#     print("Tokenized Text: ", tokens)
#     print("After decoded text: ",decoded_text)
#     # print()

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
