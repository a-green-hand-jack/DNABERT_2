import pandas as pd
from tqdm import tqdm
import random

def load_data(file_path):
    """
    Load DNA data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    print(f"Load data from {file_path}")
    return pd.read_csv(file_path)

def print_dataframe_info(dataframe):
    """
    Print information about the DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame.
    """
    print(dataframe.head())
    print(f"Shape of the DataFrame: {dataframe.shape}")

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


def set_random_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    - seed (int): Random seed value.
    """
    random.seed(seed)

def save_resulting_text_with_sampling(resulting_text, output_path, sample_ratio=0.1, random_seed=42):
    """
    Save a randomly sampled portion of the resulting text to a TXT file.

    Parameters:
    - resulting_text (str): Processed text.
    - output_path (str): Path to the output TXT file.
    - sample_ratio (float): Proportion of the resulting text to randomly sample. Default is 0.1 (10%).
    - random_seed (int): Random seed for reproducibility. Default is None.

    Returns:
    - str: Sampled text.
    """
    if random_seed is not None:
        set_random_seed(random_seed)

    total_lines = resulting_text.count('\n')
    sample_size = int(total_lines * sample_ratio)

    sampled_lines = random.sample(resulting_text.split('\n'), sample_size)
    sampled_text = '\n'.join(sampled_lines)

    with open(output_path, 'w') as file:
        file.write(sampled_text)

    print(f"Sampled text saved to {output_path} (Sample Ratio: {sample_ratio * 100}%, Random Seed: {random_seed})")
    return sampled_text



def main():
    # Modify file paths as needed
    csv_file_path = "../../Datasets/Human_genome/huixin/24_chromosomes-002.csv"
    sample_ratio = 0.01
    txt_output_path =f"../../Datasets/Human_genome/huixin/24_chromosomes-002-{sample_ratio*100}.txt"

    df = load_data(csv_file_path)
    print_dataframe_info(df)

    resulting_text = process_sequences(df)
    save_resulting_text_with_sampling(resulting_text, txt_output_path, sample_ratio)

    # Load and print the first 5 lines of the TXT file
    with open(txt_output_path, 'r') as file:
        loaded_text = file.read()

    print("\nLoaded text from TXT file - First 5 lines:")
    print(loaded_text[:512 * 5])

if __name__ == "__main__":
    main()


