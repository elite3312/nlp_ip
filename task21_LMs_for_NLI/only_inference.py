import argparse  # Import argparse for command-line arguments
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pathlib import Path
import re

# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def compute_accuracy(predictions, gold_labels):
    correct = sum(p == g for p, g in zip(predictions, gold_labels))
    return correct / len(gold_labels)

def construct_prompt(sentence1, sentence2):
    """
    Construct a prompt for in-context learning.
    """
    task_description = (
        "Determine the relationship between the following two sentences. "
        "The possible labels are: contradiction, neutral, and entailment.\n\n"
    )
    example = (
        f"Sentence 1: {sentence1}\n"
        f"Sentence 2: {sentence2}\n"
        f"Label:?"
    )
    return task_description + example

def map_to_label_exact(completion, label_map):
    # Exact Matching:
    completion_lower = completion.lower()
    for label in label_map.values():
        if label in completion_lower:
            return label
    return "unknown"  # Default if no label is found

def map_to_label_confidence_score(completion, label_map):
    # Exact Matching with Confidence Scoring:
    completion_lower = completion.lower()
    scores = {label: completion_lower.count(label) for label in label_map.values()}
    return max(scores, key=scores.get)  # Return the label with the highest score

def map_to_label_re(completion, label_map):
    # Use regular expressions to match labels more robustly, accounting for variations in capitalization, punctuation, or phrasing.
    for label in label_map.values():
        if re.search(rf'\b{label}\b', completion, re.IGNORECASE):
            return label
    return "unknown"  # Default if no label is found

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference using a pre-trained language model.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the pre-trained model or model name from Hugging Face.")
    args = parser.parse_args()

    # Paths
    dataset_path = "./task21_LMs_for_NLI/_datasets/MultiNLI_small"
    mismatched_file = Path(dataset_path) / "dev_mismatched_sampled-1.jsonl"
    matched_file = Path(dataset_path) / "dev_matched_sampled-1.jsonl"

    # Load dataset
    mismatched_data = load_dataset(mismatched_file)
    matched_data = load_dataset(matched_file)

    # Load pre-trained model and tokenizer
    model_name = args.model_name #"google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Label mapping
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

    # Evaluate on the dataset
    predictions = []
    gold_labels = []

    test_data = mismatched_data + matched_data  # Combine both datasets for evaluation
    for data in test_data[:]:  # Process the entire dataset
        sentence1 = data["sentence1"]
        sentence2 = data["sentence2"]
        gold_label = data["gold_label"]

        # Construct the prompt
        prompt = construct_prompt(sentence1, sentence2)

        # Tokenize the prompt and move it to the same device as the model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

        # Generate completion
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Map the completion to one of the labels
        predicted_label = map_to_label_confidence_score(completion, label_map)

        # Append predictions and gold labels
        predictions.append(predicted_label)
        gold_labels.append(gold_label)

    # Compute accuracy
    accuracy = compute_accuracy(predictions, gold_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()