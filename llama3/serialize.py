import jax
import jax.numpy as jnp

import os
import dataclasses
import pathlib
import pickle

@dataclasses.dataclass
class Checkpoint:
    epoch: int
    params: dict


def save_params(params, epoch, filepath):
    # Create parent directory if necessary
    parentdir = pathlib.PosixPath(filepath).parent
    os.makedirs(parentdir, exist_ok=True)
    # Save checkpoint
    numpy_params = jax.tree.map(lambda x: x.copy(), params)
    with open(filepath, 'wb') as f:
        pickle.dump(Checkpoint(epoch, numpy_params), f)

def load_params(filepath):
  with open(filepath, 'rb') as f:
      checkpoint = pickle.load(f)
  # convert back to JAX arrays
  checkpoint.params = jax.tree.map(lambda x: jnp.array(x), checkpoint.params)
  return checkpoint
