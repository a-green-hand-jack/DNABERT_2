from transformers import DNABertTokenizer, DNABertForPreTraining
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load your custom dataset using the 'datasets' module
dataset = load_dataset("text", data_files= "../../Datasets/Human_genome/huixin/24_chromosomes-002.txt")

# Tokenize the DNA sequences
tokenizer = DNABertTokenizer.from_pretrained("nateraw/dnabert_large")

# Initialize the DNABERT model
model = DNABertForPreTraining.from_pretrained("nateraw/dnabert_large")

# Define a DataLoader for efficient data loading
data_loader = DataLoader(dataset["train"], batch_size=128, shuffle=True)

# Implement your pre-training loop
for batch in data_loader:
    inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    # Backward pass and optimization steps go here

# Save the pre-trained model
model.save_pretrained("path/to/save/pretrained_dnabert")
