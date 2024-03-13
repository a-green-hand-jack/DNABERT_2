from tokenizers import (
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import shutil
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tqdm import tqdm

class TokenizerTrainer:
    def __init__(self, special_tokens=[], vocab_size=2000, data_folder = '', save_path = '', chunk_size: int = 2046, overlap_size: int=512):

        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=False
        )

        self.data_folder = data_folder
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.save_tokenizer_path = save_path

    def split_large_file(self, input_file, output_dir):
        print(f"开始分隔文件{input_file}，并且保存临时文件夹{output_dir}, 分割后形成的小片段的长度是{self.chunk_size}，并且两个小片段之间的重叠长度是{self.overlap_size}")
        with open(input_file, 'r') as file:
            content = file.read()

        num_chunks = (len(content) - (self.overlap_size * (self.chunk_size - 1))) // (self.chunk_size - self.overlap_size) + 1
        
        chunks = []

        for i in tqdm(range(0, len(content) - self.chunk_size + 1, self.chunk_size - self.overlap_size), total=num_chunks, desc="Processing Chunks", unit="chunk"):
            chunk = content[i:i + self.chunk_size]
            chunks.append(chunk)

        os.makedirs("./temp/" + output_dir, exist_ok=True)

        for i, chunk in enumerate(chunks):
            output_file = os.path.join("./temp/" + output_dir, f"chunk_{i + 1}.txt")
            with open(output_file, 'w') as output:
                output.write(chunk)

        print(f"预计形成{len(chunks)}个小片段")
        return "./temp/" + output_dir



    def train_tokenizer_on_chunks(self, folder_path):
        
        # 滑动提取器的步长
        
        file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith(".txt")]
        # stride = len(file_paths) // 10
        print(f"从{folder_path}中提取信息, 有{len(file_paths)}个文件夹，也就是字段")

         # 使用 tqdm 添加进度条
        progress_bar = tqdm(total=len(file_paths), desc="Training Tokenizer", unit="file")

        # 遍历整个文件路径列表
        for file_path in file_paths:
            # 训练当前文件内容
            self.tokenizer.train([file_path], trainer=self.trainer)

            # 更新进度条
            progress_bar.update(1)

        # 完成训练，关闭进度条
        progress_bar.close()
            # break

        # 保存训练好的分词器
        # self.save_tokenizer()


    def save_tokenizer(self):
        os.makedirs(self.save_tokenizer_path, exist_ok=True)
        print(f"Creating directory: {self.save_tokenizer_path}")
        save_path = os.path.join(self.save_tokenizer_path, "config.json")
        self.tokenizer.save(save_path)

    def split_and_train(self, file_name):
        temp_folder = file_name.split("/")[-1].split(".")[0]
        temp_file = self.split_large_file(file_name, temp_folder)
        self.train_tokenizer_on_chunks(temp_file)
        shutil.rmtree(temp_file)
        print(f"Temporary folder {temp_folder} deleted")
        

    def train_and_save(self):
        file_paths = [os.path.join(self.data_folder, file_name) for file_name in os.listdir(self.data_folder) if file_name.endswith(".txt")]

        for file_path in file_paths:
            print(f"现在正在处理{file_path}")
            self.split_and_train(file_name=file_path)
            break
        self.save_tokenizer()



if __name__ == "__main__":
    data_path:str = "../../Datasets/Human_genome/huixin/24-txt"
    save_path:str = "./my_tokenizer2/"
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    vocab_size:int = 2 ** 12
    chunk_size:int = 10000000
    overlap_size:int = 2 ** 8

    # 调用示例
    trainer = TokenizerTrainer(special_tokens=special_tokens, vocab_size=3000, chunk_size=chunk_size, overlap_size=overlap_size, data_folder=data_path, save_path=save_path)
    trainer.train_and_save()

    