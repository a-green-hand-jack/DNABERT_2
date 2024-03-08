import pandas as pd
from tqdm import tqdm
import random

def process_sequences(df):
    """
    Process DNA sequences from the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a 'Sequence' column.

    Returns:
    - str: Resulting processed text.
    """
    concatenated_sequence = ''.join(df['Sequence'])

    split_sequences = []
    for i in tqdm(range(0, len(concatenated_sequence), 512), desc="Processing"):
        sequence_chunk = concatenated_sequence[i:i + 512]
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


def main():
    # Modify file paths as needed
    csv_file_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002.csv"
    txt_output_path =f"../../Datasets/Human_genome/huixin/24_chromosomes-002.txt"

    dataframe = pd.read_csv(csv_file_path)
    print(dataframe.head())
    print(f"Shape of the DataFrame: {dataframe.shape}")

    resulting_text = process_sequences(dataframe)
    with open(txt_output_path, 'w') as file:
        file.write(resulting_text)

    output_file_path = f"../../Datasets/Human_genome/huixin/24_chromosomes-002-clean.txt"
    process_and_save_file(txt_output_path, output_file_path)

if __name__ == "__main__":
    main()


