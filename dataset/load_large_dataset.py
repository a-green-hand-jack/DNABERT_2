import pandas as pd
from tqdm import tqdm
import random
import os

def write_sequences_to_files(df, folder_path):
    # 创建文件夹路径
    os.makedirs(folder_path, exist_ok=True)
    
    # 使用 tqdm 创建进度条，并应用到 DataFrame 的每一行
    tqdm.pandas(desc="Writing sequences")
    
    # 定义处理每一行的函数
    def process_row(row):
        # 提取 ID 和 Sequence
        sequence_id = row['ID']
        sequence = row['Sequence']

        # 直接删除序列中的所有 "N"
        sequence = sequence.replace("N", "")

        # 构建文件名
        file_name = f"{sequence_id}.txt"

        # 写入文件
        with open(os.path.join(folder_path, file_name), 'w') as file:
            file.write(sequence)
        
    # 应用进度条到 DataFrame 的每一行
    df.progress_apply(process_row, axis=1)

def main(csv_file_path:str = "", txt_output_path: str = ""):

    dataframe = pd.read_csv(csv_file_path)
    # print(dataframe.head())
    print(f"Shape of the DataFrame: {dataframe.shape}")
    write_sequences_to_files(dataframe, txt_output_path)
    return dataframe


if __name__ == "__main__":
    csv_file_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002.csv"
    txt_output_path = "../../Datasets/Human_genome/huixin/24-txt/"
    # small_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002.txt"

    data_csv = main(csv_file_path = csv_file_path, txt_output_path =txt_output_path)

    