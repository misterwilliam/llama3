import jax

import jax.numpy as jnp
from jax import random

import tiktoken

import argparse
import time

import model
import serialize


def generate(params, prompt_tokens, max_new_tokens, config):
  start_ts = time.time()
  ttft_ts = None
  x = jnp.array(prompt_tokens)
  for i in range(max_new_tokens):
      x_crop = x[-config.max_seq_len:]
      logits, _ = model.model_forward(params, x_crop[None, :], config)
      logits = logits[0, -1, :]  # take the last logit
      next_token = random.categorical(random.PRNGKey(0), logits, shape=(1,))[0]
      x = jnp.concatenate([x, jnp.array([next_token])])
      if i == 0:
        ttft_ts = time.time()
  end_ts = time.time()
  print("TTFT (secs)=%.3f Tok/S=%.3f" % (ttft_ts - start_ts,
                                        (end_ts - start_ts) / len(x)))
  return x.tolist()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint", type=str, help="Path to model weights.",
                      default="model_final.pkl")
  parser.add_argument("prompt", type=str, help="Prompt to feed model.")
  args = parser.parse_args()

  print("Loading model...")
  checkpoint = serialize.load_params(args.checkpoint)

  print("Generating output...")
  enc = tiktoken.get_encoding("gpt2")
  config = model.ModelConfig()
  output = generate(checkpoint.params, jnp.array(enc.encode(args.prompt)), 20, config)

  print(enc.decode(output))


if __name__ == "__main__":
  main()
