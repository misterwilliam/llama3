import unittest

import jax.numpy as jnp

import llama3.model

class TestFeedForward(unittest.TestCase):

  def test_feedforward(self):
    dim = 3
    expansion = 2
    params = {
      "up": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(dim, expansion * dim),
      "gate": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(dim, expansion * dim),
      "down": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(expansion * dim, dim),
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
      "up": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(dim, expansion * dim),
      "gate": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(dim, expansion * dim),
      "down": jnp.arange(dim * dim * expansion, dtype=jnp.float32).reshape(expansion * dim, dim),
    }
    x = jnp.arange(batch * context_len * dim, dtype=jnp.float32).reshape(batch, context_len, dim)
    output = llama3.model.feed_forward(params, x)
    self.assertEqual(output.shape, (batch, context_len, dim))

  def test_get_mask(self):
     context_len = 4
     mask = llama3.model.get_mask(context_len, jnp.float32, mask_val=-1)
     self.assertEqual(mask.shape, (1, 1, context_len, context_len))
     self.assertTrue(
       jnp.array_equal(mask[0, 0],
                       jnp.array([
                          [0, -1, -1, -1],
                          [0,  0, -1, -1],
                          [0,  0,  0, -1],
                          [0,  0,  0,  0],
                        ])))


if __name__ == '__main__':
    unittest.main()
