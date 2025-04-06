import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from pathlib import Path
# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Map model outputs to labels
def get_label(predictions, label_map):
    return label_map[predictions.argmax(dim=1).item()]

def main():
    # Paths
    dataset_path = "./task21_LMs_for_NLI/datasets/MultiNLI_small"
    mismatched_file = Path(dataset_path) / "dev_mismatched_sampled-1.jsonl"
    matched_file = Path(dataset_path) / "dev_matched_sampled-1.jsonl"

    # Load dataset
    mismatched_data = load_dataset(mismatched_file)
    matched_data = load_dataset(matched_file)

    # Load pre-trained model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name,device_map="auto", torch_dtype=torch.float16)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Label mapping
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

    # Evaluate on the dataset
    for data in mismatched_data[:5]:  # Example: process first 5 rows
        sentence1 = data["sentence1"]
        sentence2 = data["sentence2"]

        # Tokenize inputs
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding=True).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get predicted label
        predicted_label = get_label(predictions, label_map)
        print(f"Sentence1: {sentence1}")
        print(f"Sentence2: {sentence2}")
        print(f"Gold Label: {data['gold_label']}, Predicted Label: {predicted_label}\n")

if __name__ == "__main__":
    main()