import json
from transformers import BertTokenizer
from transformers import LineByLineTextDataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

def load_tokenizer(vocab_file, special_tokens_map_file, tokenizer_config_file):
    # 读取特殊 token 映射关系
    with open(special_tokens_map_file, "r", encoding="utf-8") as f:
        special_tokens_map = json.load(f)

    # 读取 tokenizer 配置信息
    with open(tokenizer_config_file, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    # 加载词汇表
    tokenizer = BertTokenizer(vocab_file=vocab_file, **special_tokens_map, **tokenizer_config)

    return tokenizer

# 文件路径
vocab_file = "./dnabert-config/bert-config-3/vocab.txt"
special_tokens_map_file = "./dnabert-config/bert-config-3/special_tokens_map.json"
tokenizer_config_file = "./dnabert-config/bert-config-3/tokenizer_config.json"

# 加载 tokenizer
tokenizer = load_tokenizer(vocab_file, special_tokens_map_file, tokenizer_config_file)

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
