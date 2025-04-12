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


class TestRope(unittest.TestCase):

  def test_precompute(self):
    # Verify that RoPE embedding dimensions are correct. Assume the values are
    # correct.
    token_embedding_dim = 4
    context_len = 3
    rotary_embedding = llama3.model.precompute_freqs_cis(token_embedding_dim, context_len)
    self.assertEqual(rotary_embedding.shape, (context_len, token_embedding_dim // 2))


class TestAttention(unittest.TestCase):

  def test_attention(self):
    config = llama3.model.ModelConfig(dim=4,
                                      num_heads=1,
                                      num_kv_heads=1,
                                      batch_size=1,
                                      context_len=4)

    params = {
      "wq": jnp.identity(config.dim, dtype=jnp.float32),
      "wk": jnp.identity(config.dim, dtype=jnp.float32),
      "wv": jnp.identity(config.dim, dtype=jnp.float32),
      "wo": jnp.identity(config.dim, dtype=jnp.float32),
    }
    rotary_embedding = llama3.model.precompute_freqs_cis(config.dim // config.num_heads,
                                                         config.context_len)
    x = (jnp.arange(config.batch_size * config.dim * config.context_len, dtype=jnp.float32)
            .reshape(config.batch_size, config.context_len, config.dim))

    output, _ = llama3.model.attention(params, x, None, rotary_embedding,
                                       config)
    self.assertEqual(output.shape, (config.batch_size, config.context_len, config.dim))


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
