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

- only inference

  ```sh
  (.conda) (base) perry@DESKTOP-LGGEMNE:python task21_LMs_for_NLI/only_inference.py --model_name google/flan-t5-base
  Accuracy: 58.42%
  ```
- fine tuned on 5000 entries from Multinli
   
   ```sh
   (.conda) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$ python task21_LMs_for_NLI/finetune.py # finetune and save to fine_tuned_model
   (.conda) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$ python task21_LMs_for_NLI/only_inference.py --model_name fine_tuned_model
   Accuracy: 80.92%
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