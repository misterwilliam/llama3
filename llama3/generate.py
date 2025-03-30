import jax

import jax.numpy as jnp
from jax import random

from model import model_forward

import argparse
import pickle
import tiktoken
import time

enc = tiktoken.get_encoding("gpt2")

def load_params(filepath):
  with open(filepath, 'rb') as f:
      numpy_params = pickle.load(f)
  # convert back to JAX arrays
  params = jax.tree.map(lambda x: jnp.array(x), numpy_params)
  return params

def generate(params, prompt_tokens, max_new_tokens, config):
  start_ts = time.time()
  ttft_ts = None
  x = jnp.array(prompt_tokens)
  for i in range(max_new_tokens):
      x_crop = x[-config.max_seq_len:]
      logits, _ = model_forward(params, x_crop[None, :], config)
      logits = logits[0, -1, :]  # take the last logit
      next_token = random.categorical(random.PRNGKey(0), logits, shape=(1,))[0]
      x = jnp.concatenate([x, jnp.array([next_token])])
      if i == 0:
        ttft_ts = time.time()
  end_ts = time.time()
  print("TTFT (secs)=%.3f Tok/S=%.3f" % (ttft_ts - start_ts,
                                        (end_ts - start_ts) / len(x)))
  return x.tolist()

class ModelConfig:
  vocab_size = enc.n_vocab
  dim = 256
  n_layers = 6
  n_heads = 8
  n_kv_heads = 4
  max_seq_len = 512
  batch_size = 32
  learning_rate = 3e-4
  dropout_rate = 0.0


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_weights", type=str, help="Path to model weights.",
                      default="model_final.pkl")
  parser.add_argument("prompt", type=str, help="Prompt to feed model.")
  args = parser.parse_args()

  config = ModelConfig()

  print("Loading model...")
  params = load_params(args.model_weights)
  print("Generating output...")
  output = generate(params, jnp.array(enc.encode(args.prompt)), 20, config)

  print(enc.decode(output))


if __name__ == "__main__":
  main()
