import json
import random
import os

# Paths
input_file = "task21_LMs_for_NLI/_datasets/MultiNLI/multinli_1.0_train.jsonl"
output_dir = "task21_LMs_for_NLI/_datasets/MultiNLI_sampled"
output_file = os.path.join(output_dir, "multinli_1.0_train_sampled.jsonl")

# Number of records to sample
sample_size = 5000

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the large JSONL file and sample records
def sample_records(input_file, sample_size):
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read all lines into memory
        lines = f.readlines()
    
    # Check if the file has fewer records than the sample size
    if len(lines) < sample_size:
        raise ValueError(f"The file contains only {len(lines)} records, which is less than the requested sample size of {sample_size}.")
    
    # Randomly sample 5000 lines
    sampled_lines = random.sample(lines, sample_size)
    return sampled_lines

# Save the sampled records to a new JSONL file
def save_sampled_records(output_file, sampled_lines):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)

# Main script
if __name__ == "__main__":
    try:
        # Sample records
        sampled_lines = sample_records(input_file, sample_size)
        
        # Save sampled records
        save_sampled_records(output_file, sampled_lines)
        
        print(f"Successfully sampled {sample_size} records and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")