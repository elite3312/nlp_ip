import argparse  # Import argparse for command-line arguments
from collections import Counter
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from datasets import load_dataset


def compute_metrics(predictions, gold_labels):
    accuracy = sum(p == g for p, g in zip(predictions, gold_labels)) / len(gold_labels)
    precision = precision_score(gold_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(gold_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(gold_labels, predictions, average="weighted", zero_division=0)
    return accuracy, precision, recall, f1

def construct_prompt(reference, claim):
    """
    Construct a prompt for in-context learning.
    """
    task_description = (
        "Determine if the claim is supported by the reference."
        "The possible labels are: yes, no.\n\n"
    )
    example = (
        f"reference : {reference}\n"
        f"claim : {claim}\n"
        f"Label:yes or no"
    )
    return task_description + example

def map_to_label(most_common_element):
    if most_common_element == "major_inaccurate":
        return 'no'
    elif most_common_element == "minor_inaccurate":
        return 'no'
    elif most_common_element == "accurate":
        return 'yes'
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference using a pre-trained language model.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the pre-trained model or model name from Hugging Face.")
    args = parser.parse_args()
    dataset_file = "potsawee/wiki_bio_gpt3_hallucination"

    # Load dataset
    dataset = load_dataset(dataset_file)

    # Load pre-trained model and tokenizer
    model_name =args.model_name#"google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)



    # Evaluate on the dataset
    # Evaluate on the dataset
    predictions = []
    gold_labels = []

    # Counter to track the number of entries printed
    entry_count = 0

    for data in dataset['evaluation']:
        premise = data["wiki_bio_text"]
        hypothesis = data["gpt3_text"]
        annotation = data["annotation"]  # a list

        # Find the most common element as our gold label
        counter = Counter(annotation)
        most_common_element, frequency = counter.most_common(1)[0]

        gold_label = map_to_label(most_common_element)
        # Construct the prompt
        prompt = construct_prompt(premise, hypothesis)

        # Tokenize the prompt and move it to the same device as the model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

        # Generate completion
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Map the completion to one of the labels
        predicted_label = completion

        # Append predictions and gold labels
        predictions.append(predicted_label)
        gold_labels.append(gold_label)

        # Print the first 10 entries
        if entry_count < 10:
            print(f"Entry {entry_count + 1}:")
            print(f"Prediction: {predicted_label}")
            print(f"Gold Label: {gold_label}")
            print("-" * 50)
            entry_count += 1

    # Compute metrics
    accuracy, precision, recall, f1 = compute_metrics(predictions, gold_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    #print("Predictions:", predictions)
    #print("Gold Labels:", Counter(gold_labels))
if __name__ == "__main__":
    main()