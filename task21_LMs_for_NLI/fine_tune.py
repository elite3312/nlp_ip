import json
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from pathlib import Path

# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Preprocess the dataset for fine-tuning
def preprocess_data(data, tokenizer, max_length=128):
    inputs = []
    targets = []
    for item in data:
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        label = item["gold_label"]
        prompt = (
            "Determine the relationship between the following two sentences. "
            "The possible labels are: contradiction, neutral, and entailment.\n\n"
            f"Sentence 1: {sentence1}\n"
            f"Sentence 2: {sentence2}\n"
            f"Label:"
        )
        inputs.append(prompt)
        targets.append(label)
    tokenized_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    tokenized_targets = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    return Dataset.from_dict({
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"]
    })

def main():
    # Paths
    dataset_path = "./task21_LMs_for_NLI/_datasets/MultiNLI_sampled"
    train_file = Path(dataset_path) / "multinli_1.0_train_sampled.jsonl"

    # Load dataset
    train_data = load_dataset(train_file)
    full_data = train_data

    # Sample 10% of the dataset
    sample_size = int(0.1 * len(full_data))
    sampled_data = random.sample(full_data, sample_size)

    # Split dataset into training and testing (50/50 split)
    train_data, test_data = train_test_split(sampled_data, test_size=0.5, random_state=42)

    # Load pre-trained model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocess the data
    train_dataset = preprocess_data(train_data, tokenizer)
    test_dataset = preprocess_data(test_data, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",  # Directory to save the model
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",           # Save the model at the end of each epoch
        eval_strategy="epoch",     # Evaluate the model at the end of each epoch
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,     # Load the best model at the end of training
        push_to_hub=False
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    main()