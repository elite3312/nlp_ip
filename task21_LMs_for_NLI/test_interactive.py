import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_label(predictions, label_map):
    return label_map[predictions.argmax(dim=1).item()]

def main():
    # Load pre-trained model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Label mapping
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

    print("Interactive Inference with google/flan-t5-base")
    print("Enter two sentences to analyze their relationship.")
    print("Type 'exit' to quit.")

    while True:
        # Get user input
        sentence1 = input("Sentence 1: ")
        if sentence1.lower() == "exit":
            break
        sentence2 = input("Sentence 2: ")
        if sentence2.lower() == "exit":
            break

        # Tokenize inputs and move them to the same device as the model
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding=True).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get predicted label
        predicted_label = get_label(probs, label_map)
        print(f"Predicted Relationship: {predicted_label}\n")

if __name__ == "__main__":
    main()