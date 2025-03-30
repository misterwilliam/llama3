import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random, vmap

import tiktoken

from model import init_model_params, model_forward
import serialize

from functools import partial

import argparse
import pickle
import time


enc = tiktoken.get_encoding("gpt2")

# Model configuration
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



def get_batch(key, data, batch_size, seq_len):
    # Generate random starting indices
    ix = random.randint(key, (batch_size,), 0, len(data) - seq_len)

    # Vectorized operation to get input and target sequences
    x = vmap(lambda i: lax.dynamic_slice(data, (i,), (seq_len,)))(ix)
    y = vmap(lambda i: lax.dynamic_slice(data, (i + 1,), (seq_len,)))(ix)

    return x, y

def compute_loss(params, batch, config):
    inputs, targets = batch
    logits, _ = model_forward(params, inputs, config)
    logits = logits.reshape(-1, config.vocab_size)
    targets = targets.reshape(-1)
    loss = -jnp.mean(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits),
            targets[:, None],
            axis=1
        )
    )
    return loss

@partial(jax.jit, static_argnames=("config",))
def update_step(params, batch, config):
    loss, grads = jax.value_and_grad(compute_loss)(params, batch, config)
    params = jax.tree.map(
        lambda p, g: p - config.learning_rate * g,
        params,
        grads
    )
    return params, loss

def train(data, config, checkpoint, num_epochs=30, steps_per_epoch=1):
    key = random.PRNGKey(0)
    params_state = checkpoint.params

    epoch_losses = []
    for epoch in range(checkpoint.epoch, checkpoint.epoch+num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        epoch_start_ts = time.time()

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            key, batch_key = random.split(key)

            # Get batch
            batch = get_batch(batch_key, data, config.batch_size, config.max_seq_len)

            # Update model
            params_state, loss = update_step(params_state, batch, config)
            epoch_loss += loss

            if True:#if step % 10 == 0:
                print(f"epoch {epoch + 1}, step {step}/{steps_per_epoch}: loss = {loss:.4f}")
                print("Step duration: %.3f" % (time.time() - epoch_start_ts))

        avg_epoch_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(avg_epoch_loss)
        print(f"\nepoch {epoch + 1} | average loss: {avg_epoch_loss:.4f}")
        print("Avg step duration: %.3f" % ((time.time() - epoch_start_ts) / steps_per_epoch))

        serialize.save_params(params_state, epoch, f'checkpoints/epoch_{epoch+1}.pkl')

    print("Loss by epoch:")
    for epoch, loss in enumerate(epoch_losses, 1):
        print(f"Epoch {epoch}: {loss:.4f}")

    # Save final model
    serialize.save_params(params_state, epoch, 'model_final.pkl')
    return params_state


def get_params(checkpoint: str, config):
    if checkpoint == "":
        print("Randomly initializing weights")
        key = random.PRNGKey(0)
        return serialize.Checkpoint(0, init_model_params(
          key=key,
          vocab_size=config.vocab_size,
          dim=config.dim,
          n_layers=config.n_layers,
          n_heads=config.n_heads,
          n_kv_heads=config.n_kv_heads
        ))
    print("Restarting from %s" % checkpoint)
    return serialize.load_params(checkpoint)


def main():
  parser =  argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=str, help="Checkpoint to resume from.", default="", required=False)
  args = parser.parse_args()

  print("JAX devices:", jax.devices())
  # Initialize tokenizer and load data
  with open('data/shakespeare.txt', 'r') as f:
      text = f.read()
  tokens = enc.encode(text)
  data = jnp.array(tokens)

  config = ModelConfig()

  # Initialize model
  key = random.PRNGKey(0)
  params = get_params(args.checkpoint, config)
  # Train the model
  trained_params = train(data, config, params)

if __name__ == "__main__":
  main()
