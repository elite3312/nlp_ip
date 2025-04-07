import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pathlib import Path

# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def compute_accuracy(predictions, gold_labels):
    correct = sum(p == g for p, g in zip(predictions, gold_labels))
    return correct / len(gold_labels)

def construct_prompt(sentence1, sentence2):
    return (
        "Determine the relationship between the following two sentences. "
        "The possible labels are: contradiction, neutral, and entailment.\n\n"
        f"Sentence 1: {sentence1}\n"
        f"Sentence 2: {sentence2}\n"
        f"Label:"
    )

def main():
    # Paths
    dataset_path = "./task21_LMs_for_NLI/datasets/MultiNLI_small"
    mismatched_file = Path(dataset_path) / "dev_mismatched_sampled-1.jsonl"
    matched_file = Path(dataset_path) / "dev_matched_sampled-1.jsonl"

    # Load dataset
    mismatched_data = load_dataset(mismatched_file)
    matched_data = load_dataset(matched_file)
    full_data = mismatched_data + matched_data

    # Split dataset into training and testing (50/50 split)
    _, test_data = train_test_split(full_data, test_size=0.5, random_state=42)

    # Load fine-tuned model and tokenizer
    model_name = "./fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate on the test dataset
    predictions = []
    gold_labels = []

    for data in test_data:
        sentence1 = data["sentence1"]
        sentence2 = data["sentence2"]
        gold_label = data["gold_label"]

        # Construct the prompt
        prompt = construct_prompt(sentence1, sentence2)

        # Tokenize the prompt and move it to the same device as the model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

        # Generate completion
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=10)
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Append predictions and gold labels
        predictions.append(completion.strip())
        gold_labels.append(gold_label)

    # Compute accuracy
    accuracy = compute_accuracy(predictions, gold_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()