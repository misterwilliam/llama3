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
      x_crop = x[-config.context_len:]
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

def generate_with_cache(params, prompt_tokens, max_new_tokens, config):
  start_ts = time.time()
  ttft_ts = None
  x = jnp.array(prompt_tokens)

  # Generate first token without cache, but pass full prompt into model_forward.
  kv_cache = None
  x_crop = x[-config.context_len:]
  logits, kv_cache = model.model_forward(params, x_crop[None, :], config, cache=kv_cache)
  logits = logits[0, -1, :]  # take the last logit
  next_token = random.categorical(random.PRNGKey(0), logits, shape=(1,))[0]
  x = jnp.concatenate([x, jnp.array([next_token])])
  ttft_ts = time.time()

  # Generate subsequent tokens using kv_cache, but only passing in last token to
  # model_forward
  for i in range(1, max_new_tokens):
      x_crop = x[-1:]
      logits, kv_cache = model.model_forward(params, x_crop[None, :], config, cache=kv_cache)
      logits = logits[0, -1, :]  # take the last logit
      next_token = random.categorical(random.PRNGKey(0), logits, shape=(1,))[0]
      x = jnp.concatenate([x, jnp.array([next_token])])

  # Output timing
  end_ts = time.time()
  print("TTFT (secs)=%.3f Tok/S=%.3f" % (ttft_ts - start_ts,
                                        (end_ts - start_ts) / len(x)))
  return x.tolist()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint", type=str, help="Path to model weights.",
                      default="model_final.pkl")
  parser.add_argument("prompt", type=str, help="Prompt to feed model.")
  parser.add_argument("--disable-kv-cache", action="store_true")
  args = parser.parse_args()

  print("Loading model...")
  checkpoint = serialize.load_params(args.checkpoint)

  print("Generating output...")
  enc = tiktoken.get_encoding("gpt2")
  config = model.model_config
  if args.disable_kv_cache:
    output = generate(checkpoint.params, jnp.array(enc.encode(args.prompt)), 20, config)
  else:
    output = generate_with_cache(checkpoint.params, jnp.array(enc.encode(args.prompt)), 20, config)

  print(enc.decode(output))


if __name__ == "__main__":
  main()
