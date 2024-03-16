import json
from transformers import BertTokenizer
from transformers import LineByLineTextDataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
import torch

# 文件路径
  # 注意，这里一定要规定`do_lower_case=False`!!!!!

def insert_spaces(dna_sequence, interval):
    # 初始化结果字符串
    result = ""
    
    # 遍历 DNA 序列
    for i, base in enumerate(dna_sequence):
        # 每隔指定间隔插入一个空格
        if i % interval == 0 and i != 0:
            result += " "
        # 添加当前碱基
        result += base
    
    return result

def convert_tokens_to_unique_ids(tokens, tokenizer):
    # 将token转换为唯一的token ID，确保不会发生冲突
    unique_ids = []
    for token in tokens:
        unique_ids.append(tokenizer.convert_tokens_to_ids(token) + len(tokenizer))  # 将低级tokenizer的token ID偏移
    return unique_ids

def tokenize_sequence(tokenizer_high, tokenizer_low, sequence, high_token_len:int=6, low_token_len:int=3):
    # 计算两个tokenizer分词后的token数量
    tokens_high = tokenizer_high.tokenize(insert_spaces(sequence, high_token_len), add_special_tokens=True, truncation=True)
    tokens_low = tokenizer_low.tokenize(insert_spaces(sequence, low_token_len), add_special_tokens=True, truncation=True)

    print("观察 high-leveltokenize的结果:\n", tokens_high)
    print("观察 low-leveltokenize的结果:\n", tokens_low)

    # 将token转换为对应的token ID
    high_token_ids = tokenizer_high.convert_tokens_to_ids(tokens_high)
    low_token_ids = tokenizer_low.convert_tokens_to_ids(tokens_low)
    print("观察 high-level 从token到ids后的ids：\n", high_token_ids)
    print("观察 low-level 从tokens到ids后的ids：\n", low_token_ids)

    # # 将token ID转换为tensor
    high_token_tensor = torch.tensor(high_token_ids, dtype=torch.long)
    low_token_tensor = torch.tensor(low_token_ids, dtype=torch.long) + torch.tensor(len(tokenizer_high.vocab), dtype=torch.long)

    # 使用torch.cat()函数连接两个tensor
    combined_tensor = torch.cat((high_token_tensor, low_token_tensor), dim=0)

    return combined_tensor


# 加载 tokenizer
high_vocab_file = "./dnabert-config/bert-config-6/vocab.txt"
high_tokenizer = BertTokenizer(vocab_file=high_vocab_file, do_lower_case=False) 
low_vocab_file = "./dnabert-config/bert-config-3/vocab.txt"
low_tokenizer = BertTokenizer(vocab_file=low_vocab_file, do_lower_case=False) 



# 创建一个小的示例文本
dna_sequence = "AAAAAAAAAAATAAAAACAAAAAGAAAATACTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAAAAAAAAAAATAAAAACAAAAAGAAAATACTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAAAAAAAAAAATAAAAACAAAAAGAAAATACTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"



# interval = 6
# result_6 = insert_spaces(dna_sequence, 6)
# print(result_6)
# tokens = tokenizer.tokenize(result_6)
# print(tokens)

combined_token_tensor = tokenize_sequence(high_tokenizer, low_tokenizer, dna_sequence)
print(combined_token_tensor)
print(combined_token_tensor.size())  # 应该输出torch.Size([512])


# 这是一个小助手，这将帮助我们将数据集的不同样本批处理到一个Pytorch知道如何执行的backprop中.
# from transformers import DataCollatorForLanguageModeling

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# # 确保 block_size 是 tokenizer 的 block_size
# block_size = 128
# # 使用 data_collator 处理数据
# processed_data = data_collator([tokenizer.encode(result_6, max_length=block_size, truncation=True)])
# print(tokenizer.batch_encode_plus([result_6], max_length=block_size, truncation=True))
# # 输出处理后的数据
# print(processed_data)

# # 解码处理后的数据
# decoded_text = tokenizer.batch_decode(processed_data['input_ids'])[0]

# # 打印结果
# print(decoded_text)
