import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Map model outputs to labels
def get_label(predictions, label_map):
    return label_map[predictions.argmax(dim=1).item()]

def compute_accuracy(predictions, gold_labels):
    correct = sum(p == g for p, g in zip(predictions, gold_labels))
    return correct / len(gold_labels)

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
    predictions = []
    gold_labels = []

    for data in mismatched_data:  # Process the entire dataset
        sentence1 = data["sentence1"]
        sentence2 = data["sentence2"]

        # Tokenize inputs and move them to the same device as the model
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding=True).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get predicted label
        predicted_label = get_label(probs, label_map)
        predictions.append(predicted_label)
        gold_labels.append(data["gold_label"])

    # Compute accuracy
    accuracy = compute_accuracy(predictions, gold_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()