import unittest

import jax.numpy as jnp

import llama3.model


class TestRmsNorm(unittest.TestCase):

  def test_nonbatched(self):
    x = jnp.array([1, 2, 3])
    weight = 2
    output = llama3.model.rms_norm(x, jnp.array([weight, weight, weight]), eps=0)
    root_mean_square = pow((1 + 4 + 9) / 3, 0.5)
    self.assertTrue(jnp.allclose(output, x * weight / root_mean_square))

  def test_multiple_tokens(self):
    x = jnp.array([[1, 2, 3],
                   [1, 2, 3]])
    weight = 2
    output = llama3.model.rms_norm(x, jnp.array([weight, weight, weight]), eps=0)
    root_mean_square = pow((1 + 4 + 9) / 3, 0.5)
    self.assertTrue(jnp.allclose(output, x * weight / root_mean_square))

  def test_batched_multiple_tokens(self):
    x = jnp.array([[[1, 2, 3],
                    [1, 2, 3]],
                   [[1, 2, 3],
                    [1, 2, 3]]])
    weight = 2
    output = llama3.model.rms_norm(x, jnp.array([weight, weight, weight]), eps=0)
    root_mean_square = pow((1 + 4 + 9) / 3, 0.5)
    self.assertTrue(jnp.allclose(output, x * weight / root_mean_square))


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

class TestGetMask(unittest.TestCase):

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
