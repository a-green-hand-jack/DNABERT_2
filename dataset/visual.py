import pandas as pd
from tqdm import tqdm
import random

def select_random_lines(input_file, output_file, ratio):
    # 读取原始文件的所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 计算需要选择的行数
    num_lines_to_select = int(len(lines) * ratio)

    # 随机选择行
    selected_lines = random.sample(lines, num_lines_to_select)

    # 将选定的行写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(selected_lines)



def process_sequences(df, line_length:int = 512):
    """
    Process DNA sequences from the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a 'Sequence' column.

    Returns:
    - str: Resulting processed text.
    """
    concatenated_sequence = ''.join(df['Sequence'])

    split_sequences = []
    for i in tqdm(range(0, len(concatenated_sequence), line_length), desc="Processing"):
        sequence_chunk = concatenated_sequence[i:i + line_length]
        if sequence_chunk.count('N') / len(sequence_chunk) <= 0.15:
            split_sequences.append(sequence_chunk)

    return '\n'.join(split_sequences)

def process_and_save_file(input_path, output_path):
    with open(input_path, 'r') as input_file:
        lines = input_file.readlines()

    # Filter lines based on the condition (less than 15% 'N' characters)
    filtered_lines = [line for line in lines if (line.count('N') / len(line)) <= 0.15]

    with open(output_path, 'w') as output_file:
        output_file.writelines(filtered_lines)


def main(csv_file_path:str = "", txt_output_path: str = "", line_length:int = 512):

    dataframe = pd.read_csv(csv_file_path)
    print(dataframe.head())
    print(f"Shape of the DataFrame: {dataframe.shape}")

    resulting_text = process_sequences(dataframe, line_length)
    with open(txt_output_path, 'w') as file:
        file.write(resulting_text)

if __name__ == "__main__":
    csv_file_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002.csv"
    txt_output_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002-256.txt"
    # small_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002.txt"

    main(csv_file_path = csv_file_path, txt_output_path =txt_output_path, line_length=256)
    # 示例用法：选择原文件的30%的行并保存到新文件
    # select_random_lines(txt_output_path, small_path, 0.5)

