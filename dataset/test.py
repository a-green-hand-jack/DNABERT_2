from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader



def load_and_split_dataset(txt_file_path, test_size=0.2, random_state=42):
    # 加载数据集
    data_files = {"train": f"{txt_file_path}/train.txt", "test": f"{txt_file_path}/eval.txt"}
    pubmed_dataset_streamed = load_dataset("text", data_files=data_files, streaming=True)

    return pubmed_dataset_streamed['train'], pubmed_dataset_streamed['test']

class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        # print(self.texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        return encoding
    
if __name__ == "__main__":

    # 替换成你的文件路径和其他参数
    txt_file_path = "../../Datasets/Human_genome/huixin"
    tokenizer_path = "../tokenizer/save_tokenizer_small"
    print(f"从{tokenizer_path}中加载tokenizer")
    print(f"从{txt_file_path}中加载数据")

    train_dataset, eval_dataset = load_and_split_dataset(txt_file_path, test_size=0.01)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, padding=True, truncation=True, max_length=512, return_tensors="pt")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    print(eval_dataset)
    # print(eval_dataset['text'])
    # 假设 eval_dataset 是 IterableDataset 对象
    eval_dataset_iter = iter(eval_dataset)

    # 使用迭代器遍历数据集
    for item in eval_dataset_iter:
        # 使用 tokenizer 和 data_collator 处理 item['text']
        print(item['text'])
        # inputs = tokenizer(item['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        inputs = tokenizer.encode(item['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        output = data_collator([inputs])
        # print(outputs)
        # 输出的 input_ids 表示输入的 token IDs
        # print("Input IDs:", output['input_ids'])
        print(inputs)
        print(output)

        # 输出的 attention_mask 表示模型在哪些位置需要关注
        # print("Attention Mask:", output['attention_mask'])

        # 输出的 labels 表示用于训练的标签
        # print("Labels:", output['labels'])
        # print(f"输入的序列的长度是：{len(item['text'])}")
        # print(f"经过分词器后的长度{len(inputs)}")
        # print(f"输出的经过处理的形状是{output['labels'].shape}")
        # 解码处理后的数据
        # 将 output['input_ids'] 展平为一维列表
        flat_input_ids = output['input_ids'].flatten().tolist()

        # 使用 batch_decode 处理展平后的列表
        decoded_text = tokenizer.batch_decode([flat_input_ids])

        # 打印结果
        print(decoded_text)
        break
    

