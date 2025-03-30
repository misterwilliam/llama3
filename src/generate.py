import jax
import jax.numpy as jnp
from jax import random, vmap
from model import model_forward

import pickle
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def load_params(filepath):
    with open(filepath, 'rb') as f:
        numpy_params = pickle.load(f)
    # convert back to JAX arrays
    params = jax.tree.map(lambda x: jnp.array(x), numpy_params)
    return params

def generate(params, prompt_tokens, max_new_tokens, config):
    x = jnp.array(prompt_tokens)
    for _ in range(max_new_tokens):
        x_crop = x[-config.max_seq_len:]
        logits, _ = model_forward(params, x_crop[None, :], config)
        logits = logits[0, -1, :]  # take the last logit
        next_token = random.categorical(random.PRNGKey(0), logits, shape=(1,))[0]
        x = jnp.concatenate([x, jnp.array([next_token])])
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

config = ModelConfig()

params = load_params("model_final.pkl")
output = generate(params, jnp.array(enc.encode("How are you?")), 20, config)

print(enc.decode(output))
