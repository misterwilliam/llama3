import jax
import jax.numpy as jnp
from jax import random
import tiktoken

import dataclasses
import math

enc = tiktoken.get_encoding("gpt2")

@dataclasses.dataclass(frozen=True)
class ModelConfig:
  vocab_size: int | None = None
  dim: int | None = None
  num_layers: int | None = None
  num_heads: int | None = None
  num_kv_heads: int | None = None
  context_len: int | None = None
  batch_size: int | None = None
  learning_rate: float | None = None
  dropout_rate: float | None = None

model_config = ModelConfig(vocab_size=enc.n_vocab,
                           dim=256,
                           num_layers=6,
                           num_heads=8,
                           num_kv_heads=4,
                           context_len=512,
                           batch_size=32,
                           learning_rate=3e-4,
                           dropout_rate=0.0)

def rms_norm(x: jnp.array, weight: jnp.array, eps=1e-5):
    """Normalize x by its root mean square and weight by weight.

    Input can be a single token's embedding, list of token embeddings, or a
    batch of multiple token embeddings.
    """
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * weight * jnp.reciprocal(rms)

def precompute_freqs_cis(token_embedding_dim: int, context_len: int, theta: float = 10000.0):
    """Return RoPE embedding

    A RoPE embedding is a matrix of shape: (context_len, dim // 2). It
    represents a list of context_len vectors that specify a rotation for each
    token position. It treats each token embedding as a list of pairs, and
    rotates each pair. For example if a token embedding is [a, b, c, d], RoPE
    imagines the embedding is [(a, b), (c, d)], and provides a rotation for each
    pair. Therefore the RoPE embedding dimension is dim // 2.
    """
    if token_embedding_dim % 2 != 0:
      raise ValueError(
          "Got token embedding dimension %i. The dimension must be even." % token_embedding_dim)
    freqs = 1.0 / (
        theta ** (jnp.arange(token_embedding_dim // 2, dtype=jnp.float32) / token_embedding_dim)
      )
    t = jnp.arange(context_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.complex64(jnp.exp(1j * freqs))

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_r, xk_r = jnp.reshape(xq, (*xq.shape[:-1], -1, 2)), jnp.reshape(xk, (*xk.shape[:-1], -1, 2))
    xq_complex = jnp.complex64(xq_r[..., 0] + 1j * xq_r[..., 1])
    xk_complex = jnp.complex64(xk_r[..., 0] + 1j * xk_r[..., 1])
    freqs_cis = jnp.reshape(freqs_cis, (1, freqs_cis.shape[0], 1, freqs_cis.shape[1]))
    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis
    xq = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)
    return xq, xk

def repeat_kv(x, n_rep):
    return x if n_rep == 1 else jnp.repeat(x, n_rep, axis=2)

def init_weight(key, shape, scale=None):
    scale = 1.0 / math.sqrt(shape[0]) if scale is None else scale
    return jax.random.normal(key, shape) * scale

def init_attention_weights(key, dim, num_heads, num_kv_heads):
    keys = jax.random.split(key, 4)
    head_dim = dim // num_heads
    return {
        'wq': init_weight(keys[0], (dim, num_heads * head_dim)),
        'wk': init_weight(keys[1], (dim, num_kv_heads * head_dim)),
        'wv': init_weight(keys[2], (dim, num_kv_heads * head_dim)),
        'wo': init_weight(keys[3], (num_heads * head_dim, dim))
    }

def init_ffn_weights(key, dim):
    keys = jax.random.split(key, 3)
    return {
        'up': init_weight(keys[0], (dim, 4 * dim)),
        'gate': init_weight(keys[1], (dim, 4 * dim)),
        'down': init_weight(keys[2], (4 * dim, dim))}

def init_transformer_block(key, dim, num_heads, num_kv_heads):
    keys = jax.random.split(key, 4)
    return {
        'attention': init_attention_weights(keys[0], dim, num_heads, num_kv_heads),
        'ffn': init_ffn_weights(keys[1], dim),
        'attention_norm': init_weight(keys[2], (dim,), scale=1.0),
        'ffn_norm': init_weight(keys[3], (dim,), scale=1.0)}

def init_model_params(key, vocab_size, dim, num_layers, num_heads, num_kv_heads):
    keys = jax.random.split(key, 4)
    params = {
        'token_embedding': init_weight(keys[0], (vocab_size, dim)),
        'norm_f': init_weight(keys[1], (dim,), scale=1.0),
        'output': init_weight(keys[2], (dim, vocab_size))
    }
    block_keys = jax.random.split(keys[3], num_layers)
    params['blocks'] = [init_transformer_block(k, dim, num_heads, num_kv_heads) for k in block_keys]
    return params

def attention(params, x, mask, freqs_cis, config, cache=None, position=0):
    B, T, input_embedding_dim = x.shape
    if input_embedding_dim != config.dim:
      raise ValueError(
          ("Input shape is batch size: %i token len: %i input embedding dim: %i,"
           " but expected input embedding dim to be: %i") % (B, T, input_embedding_dim, config.dim))
    head_dim = config.dim // config.num_heads
    q = jnp.dot(x, params['wq']).reshape(B, T, config.num_heads, head_dim)
    k = jnp.dot(x, params['wk']).reshape(B, T, config.num_kv_heads, head_dim)
    v = jnp.dot(x, params['wv']).reshape(B, T, config.num_kv_heads, head_dim)
    q, k = apply_rotary_emb(q, k, freqs_cis[position:position + T])
    if cache is not None:
        k = jnp.concatenate([cache[0], k], axis=1)
        v = jnp.concatenate([cache[1], v], axis=1)
    new_cache = (k, v)
    k = repeat_kv(k, config.num_heads // config.num_kv_heads)
    v = repeat_kv(v, config.num_heads // config.num_kv_heads)
    q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask[:, :, :T, :T]
    scores = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(scores, v)
    output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
    return jnp.dot(output, params['wo']), new_cache

def feed_forward(params, x):
  # feed_forward implements SwiGLU activation. Works batched and non-batched.
  # x.shape = (batch, context length, dim)
  # up.shape = (dim, 4 * dim)
  # gate.shape = (dim, 4 * dim)
  # down.shape = (4 * dim, dim)
  #
  # The function:
  # 1. Projects input to a higher dimension with up weights
  # 2. Creates a gate using SiLU activation with gate weights
  # 3. Element-wise multiplies these intermediate values
  # 4. Projects back to original dimension with down weights
  return jnp.dot(
      jnp.dot(x, params["up"]) *
      jax.nn.silu(jnp.dot(x, params["gate"])),
    params["down"])

def transformer_block(params, x, mask, freqs_cis, config,
                      cache=None, position=0, training=False, dropout_rate=0.0,
                      key=None):
    if training and key is None and dropout_rate > 0:
        assert False, "key must be provided when training with drop out"
    x = rms_norm(x, params['attention_norm'])
    attn_output, new_cache = attention(params['attention'], x, mask, freqs_cis,
                                       config, cache, position)
    if training:
        dropout_key, key = jax.random.split(key)
        attn_output = (
            jax.random.bernoulli(dropout_key,
                                 1-dropout_rate,
                                 shape=attn_output.shape) * attn_output / (1-dropout_rate)
        )
    h = x + attn_output
    ffn_output = feed_forward(params['ffn'], rms_norm(h, params['ffn_norm']))
    if training:
        dropout_key, key = jax.random.split(key)
        ffn_output = (
            jax.random.bernoulli(dropout_key,
                                 1-dropout_rate,
                                 shape=ffn_output.shape) * ffn_output / (1-dropout_rate)
        )
    out = h + ffn_output
    return out, new_cache

def get_mask(context_len, dtype, mask_val=-1e-9):
  # Return lower triangle of 0s and top triangle filled with mask_val. Shape of
  # return value is (1, 1, context_len, context_len). The intended us case is to
  # add mask to attention scores and then soft max. Therefore to zero out the
  # the attention in the soft max we need to use a large negative number, not 0s.
  mask = jnp.tril(jnp.ones((context_len, context_len)))
  mask = jnp.where(mask == 0, mask_val, 0.0)
  mask = mask.astype(dtype)
  return mask[None, None, :, :]

def model_forward(params, inputs, config, key=None, training=False, cache=None, position=0):
    # B, T = inputs.shape
    h = params['token_embedding'][inputs]
    freqs_cis = precompute_freqs_cis(config.dim // config.num_heads, config.context_len)
    mask = get_mask(config.context_len, h.dtype)
    new_caches = []
    for i, block in enumerate(params['blocks']):
        layer_cache = cache[i] if cache is not None else None
        h, layer_cache = transformer_block(block, h, mask, freqs_cis, config,
                                           layer_cache, position,
                                           training=training, dropout_rate=config.dropout_rate,
                                           key=key)
        new_caches.append(layer_cache)
    h = rms_norm(h, params['norm_f'])
    logits = jnp.dot(h, params['output'])
    return logits, new_caches
