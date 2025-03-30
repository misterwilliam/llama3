import unittest

import jax.numpy as jnp

import llama3.model

class TestFeedForward(unittest.TestCase):

  def test_feedforward(self):
    params = {
      "w1": jnp.array([1.0, 2.0, 3.0]),
      "w2": jnp.array([1.0, 2.0, 3.0]),
      "w3": jnp.array([1.0, 2.0, 3.0]),
    }
    x = jnp.array([1.0, 2.0, 3.0])
    output = llama3.model.feed_forward(params, x)
    self.assertEqual(output.shape, (3,))

if __name__ == '__main__':
    unittest.main()
