# nlp_ip

## setup

```sh
# inside a conda env, python 3.11
pip install transformers
pip install torch
pip install accelerate
pip install scikit-learn
pip install datasets
```

## task 2.1 Applying LMs for NLI

### test results

- flan-t5-base
  - only inference

    ```sh
    (.conda) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$ python task21_LMs_for_NLI/only_inference_flant5.py --model_name google/flan-t5-base --data matched
    Prompt: Determine the relationship between the following two sentences. The possible labels are: contradiction, neutral, and entailment.

    Sentence 1: oh that sounds interesting too
    Sentence 2: That is not very attention grabbing. 
    Label:?
    Completion: neutral
    Gold Label: contradiction
    --------------------------------------------------
    Prompt: Determine the relationship between the following two sentences. The possible labels are: contradiction, neutral, and entailment.

    Sentence 1: ( sums up the millennium coverage from around the globe, and  examines whether the Y2K preparations were a waste.)
    Sentence 2: (The millennium coverage from around the globe is summed up and examined, but results are not out yet).
    Label:?
    Completion: neutral
    Gold Label: neutral
  
    Accuracy: 58.48%
    (.conda) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$ python task21_LMs_for_NLI/only_inference_flant5.py --model_name google/flan-t5-base --data mismatched
    Prompt: Determine the relationship between the following two sentences. The possible labels are: contradiction, neutral, and entailment.

    Sentence 1: Further, there is no universally accepted way to transliterate Arabic words and names into English.
    Sentence 2: Arabic words and names are easily translated.
    Label:?
    Completion: neutral
    Gold Label: contradiction
    --------------------------------------------------
    Prompt: Determine the relationship between the following two sentences. The possible labels are: contradiction, neutral, and entailment.

    Sentence 1: One letter writer to The Times [21 June 1988] 
    Sentence 2: A letter written at 5 in the morning on June 21st.
    Label:?
    Completion: neutral
    Gold Label: neutral
    --------------------------------------------------
    
    Accuracy: 58.36%
    ```
  - fine tuned on 5000 entries from Multinli
    
    ```sh
    (.conda) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$ python task21_LMs_for_NLI/finetune_flant5.py # finetune and save to fine_tuned_model
    (.conda) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$ python task21_LMs_for_NLI/only_inference_flant5.py --model_name fine_tuned_model_flan-t5-base --data matched
    Accuracy: 81.00%
    (.conda) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$ python task21_LMs_for_NLI/only_inference_flant5.py --model_name fine_tuned_model_flan-t5-base --data mismatched
    Accuracy: 80.84%
    ```
- gpt2
  - only inference
  ```sh
  (.conda) (base) perry@DESKTOP-LGGEMNE:python task21_LMs_for_NLI/only_inference_gpt2.py --model_name gpt2-medium --data matched
  Completion:     Please help me answer this problem:
    Premise: {That will only be the first step toward healing the pain of having been abandoned several times, neglected, and denied the happy childhood that should be every child's birthright.}
    Hypothesis: {No steps will be taken to help them overcome those abuses.}
    Question: Is the hypothesis entailed by the premise?
    Answer:                                  
    Gold Label: contradiction
    Accuracy: 35.68%
    (.conda) (base) perry@DESKTOP-LGGEMNE:python task21_LMs_for_NLI/only_inference_gpt2.py --model_name gpt2-medium --data mismatched
    Accuracy: 36.00%
    ```

## task 2.2 NLI for hallucination detection

- loading the model without finetuning

```sh
(base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$  cd /home/perry/nlp2025/nlp_ip ; /usr/bin/env /home/perry/nlp2025/nlp_ip/.conda/bin/python /home/perry/.vscode-server/extensions/ms-python.debugpy-2025.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 39791 -- /home/perry/nlp2025/nlp_ip/task22_NLI_for_hallucination_detection/classify.py --model google/flan-t5-base 
Accuracy: 52.10%
Precision: 68.35%
Recall: 52.10%
F1 Score: 53.50%
```

- loading the finetuned model

```sh
(base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$  cd /home/perry/nlp2025/nlp_ip ; /usr/bin/env /home/perry/nlp2025/nlp_ip/.conda/bin/python /home/perry/.vscode-server/extensions/ms-python.debugpy-2025.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 60487 -- /home/perry/nlp2025/nlp_ip/task22_NLI_for_hallucination_detection/classify.py --model fine_tuned_model 
Accuracy: 71.43%
Precision: 71.43%
Recall: 71.43%
F1 Score: 71.43%
```



## useful commands

```sh
du /home/perry/.cache/huggingface/ -h # to check huggingface disk usage
```