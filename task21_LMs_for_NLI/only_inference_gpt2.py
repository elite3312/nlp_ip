import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM  # Use AutoModelForCausalLM for GPT-2
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
    """d
    Construct a prompt for in-context learning.
    """
   
    prompt="\
    Please help me answer this problem:\n\
    Premise: {%s}\n\
    Hypothesis: {%s}\n\
    Question: Is the hypothesis entailed by the premise?\n\
    Answer:\
    "%(sentence1,sentence2)

    return prompt
    

def verbalizer(completion, label_map):
    """
    Map the model's output to one of the predefined labels or 'unknown'.
    """
    completion_lower = completion.lower()
    completion=completion[:completion.find("Answer:")] # Find the index of "Answer:"
    if "entail" in completion_lower or "yes" in completion_lower:
        return "entailment"
    elif "contradict" in completion_lower or "no" in completion_lower:
        return "contradiction"
    elif "neutral" in completion_lower or "maybe" in completion_lower:
        return "neutral"


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference using a pre-trained language model.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the pre-trained model or model name from Hugging Face.")
    parser.add_argument("--data", type=str, required=True, help="matched or mismatched")
    
    args = parser.parse_args()

    # Paths
    dataset_path = "./task21_LMs_for_NLI/_datasets/MultiNLI_small"
    mismatched_file = Path(dataset_path) / "dev_mismatched_sampled-1.jsonl"
    matched_file = Path(dataset_path) / "dev_matched_sampled-1.jsonl"

    # Load dataset
    mismatched_data = load_dataset(mismatched_file)
    matched_data = load_dataset(matched_file)

    # Load pre-trained model and tokenizer
    model_name = args.model_name  # e.g., "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set a padding token for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
   
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Label mapping
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

    # Evaluate on the dataset
    predictions = []
    gold_labels = []

    if args.data == "mismatched":
        test_data = mismatched_data
    elif args.data == "matched":
        test_data = matched_data
    i=0
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
            outputs = model.generate(**inputs, max_new_tokens=100)
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if i<10:
                #print(f"Prompt: {prompt}")
                print(f"Completion: {completion}")
                print(f"Gold Label: {gold_label}")
                print("-" * 50)
        # Map the completion to one of the labels
        predicted_label = verbalizer(completion, label_map)

        # Append predictions and gold labels
        predictions.append(predicted_label)
        gold_labels.append(gold_label)
        i+=1
    # Compute accuracy
    accuracy = compute_accuracy(predictions, gold_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()