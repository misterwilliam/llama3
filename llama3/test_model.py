import unittest

import jax.numpy as jnp

import llama3.model

class TestFeedForward(unittest.TestCase):

  def test_feedforward(self):
    dim = 3
    expansion = 2
    params = {
      "w1": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(dim, expansion * dim),
      "w2": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(dim, expansion * dim),
      "w3": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(expansion * dim, dim),
    }
    x = jnp.arange(dim, dtype=jnp.float32).reshape(dim)
    output = llama3.model.feed_forward(params, x)
    self.assertEqual(output.shape, (dim,))

  def test_feedforward_batched(self):
    batch = 5
    context_len = 3
    dim = 3
    expansion = 2
    params = {
      "w1": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(dim, expansion * dim),
      "w2": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(dim, expansion * dim),
      "w3": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(expansion * dim, dim),
    }
    x = jnp.arange(batch * context_len * dim, dtype=jnp.float32).reshape(batch, context_len, dim)
    output = llama3.model.feed_forward(params, x)
    self.assertEqual(output.shape, (batch, context_len, dim))

if __name__ == '__main__':
    unittest.main()
