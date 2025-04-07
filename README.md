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
## useful commands

```sh
du /home/perry/.cache/huggingface/ -h # to check huggingface disk usage
```
