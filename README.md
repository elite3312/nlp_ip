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

  ```txt
  (.conda) (base) perry@DESKTOP-LGGEMNE:~/nlp2025/nlp_ip$ /home/perry/nlp2025/nlp_ip/.conda/bin/python /home/perry/nlp2025/nlp_ip/task21_LMs_for_NLI/main.py
  Accuracy: 58.42%
  ```

## useful commands

```sh
du /home/perry/.cache/huggingface/ -h # to check huggingface disk usage
```
