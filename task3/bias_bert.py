import torch
from transformers import BertTokenizer, BertForMaskedLM
from datasets import load_dataset
import random

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
model.eval()

# Function to calculate pseudo-log-likelihood
def pseudo_log_likelihood(sentence, model, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    log_likelihood = 0.0

    for i in range(1, len(input_ids[0]) - 1):  # Skip special tokens
        masked_input = input_ids.clone()
        masked_input[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            outputs = model(masked_input)
            logits = outputs.logits
        token_log_prob = torch.log_softmax(logits[0, i], dim=-1)[input_ids[0, i]]
        log_likelihood += token_log_prob.item()

    return log_likelihood

def evaluate_bias(pairs, model, tokenizer):
    stereotyping_higher_count = 0

    for pair in pairs:
        sentence1, sentence2 = pair
        pll1 = pseudo_log_likelihood(sentence1, model, tokenizer)
        pll2 = pseudo_log_likelihood(sentence2, model, tokenizer)

        # Check if the stereotyping sentence (S1) has a higher pseudo-log-likelihood
        if pll1 > pll2:
            stereotyping_higher_count += 1

    # Calculate the percentage of examples where S1 > S2
    percentage = (stereotyping_higher_count / len(pairs)) * 100
    return percentage

# Load CrowS-Pairs dataset and filter for gender examples
def load_crows_pairs(num_examples=80):
    dataset = load_dataset("crows_pairs", split="test")
    gender_examples = [item for item in dataset if item["bias_type"] == 2]
    '''
    2 for gender
    '''
    selected_examples =  gender_examples[:81]#random.sample(gender_examples, num_examples)
    pairs = [(item["sent_more"], item["sent_less"]) for item in selected_examples]
    return pairs

if __name__ == "__main__":
    # Load 80 gender examples
    pairs = load_crows_pairs(num_examples=80)

    # Print the first 10 entries with PLLs for case study
    print("First 10 pairs with pseudo-log-likelihoods (PLLs):")
    for i, (sent_more, sent_less) in enumerate(pairs[:10], start=1):
        pll1 = pseudo_log_likelihood(sent_more, model, tokenizer)
        pll2 = pseudo_log_likelihood(sent_less, model, tokenizer)
        print(f"Pair {i}:")
        print(f"  Stereotyping Sentence (S1): {sent_more}")
        print(f"    PLL(S1): {pll1:.4f}")
        print(f"  Less Stereotyping Sentence (S2): {sent_less}")
        print(f"    PLL(S2): {pll2:.4f}")
        print()

    # Evaluate bias
    percentage = evaluate_bias(pairs, model, tokenizer)

    # Print results
    print(f"Percentage of examples where S1 > S2: {percentage:.2f}%")