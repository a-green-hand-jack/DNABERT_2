from tokenizers import (
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
    ByteLevelBPETokenizer
)
import os

tokenizer = ByteLevelBPETokenizer()
special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
chunks_dir = "../../Datasets/Human_genome/huixin/24-txt"
file_paths = [os.path.join(chunks_dir, file_name) for file_name in os.listdir(chunks_dir) if file_name.endswith(".txt")]
print(file_paths)
tokenizer.train(file_paths, vocab_size=20000, special_tokens=special_tokens)

save_path = "./temp_tokenizer"
save_path = os.path.join(save_path, "config.json")
tokenizer.save_model(save_path)	# 保存模型，生成vocab.txt文件
