import pandas as pd
from tqdm import tqdm

dna_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002.csv"
print(f"load data from {dna_path}")

# 读取CSV文件
df = pd.read_csv(dna_path)  # 替换为你的文件路径

# 显示数据集的前几行
print(df.head())

# 显示DataFrame的形状
print(f"Shape of the DataFrame: {df.shape}")

# 打印 'ID' 列的数据
print("ID column:")
print(df['ID'])

# 打印 'Description' 列的数据
print("\nDescription column:")
print(df['Description'])

# 打印 'Sequence' 列每一行的数据长度
print("\nSequence column lengths:")
for idx, sequence in enumerate(df['Sequence']):
    sequence_length = len(sequence)
    print(f"Row {idx + 1}: Length - {sequence_length}")

# 拼接 'Sequence' 列的每一行
concatenated_sequence = ''.join(df['Sequence'])

# 分隔每512个字符，每行包含512个字符，并且忽略含有15%以上N的部分
split_sequences = []
for i in tqdm(range(0, len(concatenated_sequence), 512), desc="Processing"):
    sequence_chunk = concatenated_sequence[i:i + 512]
    if sequence_chunk.count('N') / len(sequence_chunk) <= 0.15:
        split_sequences.append(sequence_chunk)

# 将分隔后的序列用换行连接
resulting_text = '\n'.join(split_sequences)

# 保存到txt文件
output_file_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002.txt"
with open(output_file_path, 'w') as file:
    file.write(resulting_text)

print(f"Resulting text saved to {output_file_path}")


# 加载并读取txt文件
with open(output_file_path, 'r') as file:
    loaded_text = file.read()

# 打印txt文件的前5行
print("\nLoaded text from TXT file - First 5 lines:")
print(loaded_text[:512 * 5])

