# omega-autograd-engine

A from-scratch reverse-mode autograd engine (numpy-backed, no PyTorch/JAX
dependency for the engine itself) built up to a working, trainable
transformer — Tensor + backprop → NN layers → optimizers → causal
multi-head attention → a small character-level language model.

Everything here is **verified**, not just written: every op has a
gradient-check test comparing analytic backward passes against numerical
(finite-difference) gradients, and both training scripts (`train_mnist.py`,
`train_transformer.py`) run end-to-end and actually learn.

## Structure

```
autograd/
    tensor.py           # Tensor: data, grad, computational graph, core ops
    ops.py               # relu, softmax, sigmoid, tanh (+ add/matmul aliases)
    attention_ops.py      # masked_softmax, scaled_dot_product_attention,
                           #   layer_norm, gelu — all with analytic backward
nn.py                    # Linear, Embedding, LayerNorm, MultiHeadAttention,
                          #   FeedForward, TransformerBlock (pre-LN)
optimizers.py             # Optimizer, SGD(+momentum), Adam, AdamW
data.py                   # CharTokenizer, LM batching, digit-image loading
profiling.py               # forward/backward timing, O(n^2) attention
                            #   memory-scaling measurement
tests/
    test_engine.py         # gradcheck: add, mul, matmul, relu, softmax,
                            #   sum, mean, reshape, transpose, pow, div
    test_attention.py      # gradcheck: masked_softmax, attention, layer_norm,
                            #   gelu, + causal-mask leakage sanity check
train_mnist.py             # MLP sanity check (digit classification)
train_transformer.py       # character-level transformer LM
```

## Running it

```bash
pip install numpy scipy scikit-learn --break-system-packages

python3 tests/test_engine.py        # core op gradchecks
python3 tests/test_attention.py     # attention/layernorm/gelu gradchecks
python3 profiling.py                # attention memory scaling demo
python3 train_mnist.py              # MLP sanity check
python3 train_transformer.py        # transformer LM training
```

## What's actually verified vs. approximated

- **Gradients**: every custom backward pass (including the hand-derived
  LayerNorm backward, which is the easiest one to get subtly wrong) is
  checked against numerical differentiation, not just "looks right."
  This sandbox doesn't have PyTorch installed, so the tests use
  finite-difference gradcheck directly rather than comparing to
  `torch.autograd` — which is what `torch.autograd.gradcheck` does
  internally anyway. If you have PyTorch locally, you can additionally
  mirror any op in torch and diff against `.grad` for extra confidence.
- **"MNIST"**: this sandbox has no internet access to download the real
  28x28 MNIST files, so `train_mnist.py` uses scikit-learn's built-in
  `load_digits` dataset (1797 8x8 grayscale digit images) as a stand-in.
  Structurally the same problem, but **not** real MNIST. Swap in
  `torchvision.datasets.MNIST` (or the raw files) if you have internet —
  the training loop itself doesn't need to change.
- **Transformer training corpus**: no internet access to pull an external
  text corpus either, so `train_transformer.py` trains on a small
  built-in sample paragraph (repeated to give enough data to sample
  chunks from). It's enough to prove the full attention + LayerNorm +
  GELU + residual stack trains and starts reproducing structure from the
  text — not a meaningful language model. Swap `load_corpus()` to read
  your own `.txt` file for a real run.

## Design notes worth knowing if you extend this

- **Pre-LN transformer blocks** (`x = x + Attn(LN(x))`, not
  `LN(x + Attn(x))`) — pre-LN trains more stably without a learning-rate
  warmup, which matters more here since there's no fused/optimized
  attention kernel to fall back on.
- **AdamW uses decoupled weight decay** (applied directly to the
  parameters, not folded into the gradient before the moment estimates)
  — that's the actual difference from `Adam(weight_decay=...)`, not just
  a naming difference.
- **`masked_softmax` uses an additive mask** (0 = allowed, -1e9 =
  disallowed), matching how causal/padding masks are typically built in
  real transformer implementations, rather than a boolean mask.
- **Attention memory is O(seq_len²)** — `profiling.py` measures this
  directly (doubling seq_len ≈ quadruples the score-matrix size), which
  is the concrete motivation for techniques like FlashAttention or
  sliding-window attention on longer sequences.

## Known limitations

- Pure numpy — no GPU support, no fused kernels. This is an engine for
  *understanding* autograd/transformers, not for training anything at
  real scale.
- No dropout implemented (not needed at this corpus size, but you'd want
  it before scaling up to avoid overfitting).
- Single-machine-only; no distributed/parallel training utilities.
