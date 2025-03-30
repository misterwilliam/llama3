# llama3
Llama3 in JAX

Implementation based upon:
https://github.com/saurabhaloneai/Llama-3-From-Scratch-In-Pure-Jax/tree/main


## Install

```sh
$ python3 -m venv ~/venvs/llama3
$ . ~/venvs/llama3/bin/activate
$ pip install jax
$ pip install tiktoken
```

## Train

```sh
$ python3 llama3/train.py

# Resume training from checkpoint
$ python3 llama3/train.py --checkpoint=checkpoints/epoch_4.pkl
```

## Predict

```sh
$ python3 llama3/generate.py checkpoints/epoch_4.pkl "Hello"
```
