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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Label mapping
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

    # Evaluate on the dataset
    predictions = []
    gold_labels = []

    for data in matched_data+mismatched_data:  # Process the entire dataset
        sentence1 = data["sentence1"]
        sentence2 = data["sentence2"]
        gold_label = data["gold_label"]

        # Construct the prompt
        prompt = construct_prompt(sentence1, sentence2)

        # Tokenize the prompt and move it to the same device as the model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

        # Generate completion
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Map the completion to one of the labels
        predicted_label = None
        for label in label_map.values():
            if label in completion.lower():
                predicted_label = label
                break

        # Append predictions and gold labels
        predictions.append(predicted_label)
        gold_labels.append(gold_label)

    # Compute accuracy
    accuracy = compute_accuracy(predictions, gold_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()