import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd.checkpoint_ops import checkpoint
from nn import TransformerBlock, causal_mask

RNG = np.random.default_rng(3)


def test_checkpoint_matches_direct_call_transformer_block():
    np.random.seed(0)
    seq, d_model, n_heads, d_ff = 10, 16, 4, 32
    block = TransformerBlock(d_model, n_heads, d_ff)
    mask = causal_mask(seq)

    x_data = RNG.standard_normal((1, seq, d_model))

    x1 = Tensor(x_data.copy(), requires_grad=True)
    block.zero_grad()
    out1 = block(x1, mask=mask)
    out1.sum().backward()
    grad_x1 = x1.grad.copy()
    param_grads1 = [p.grad.copy() for p in block.parameters()]


    x2 = Tensor(x_data.copy(), requires_grad=True)
    block.zero_grad()
    out2 = checkpoint(lambda xx: block(xx, mask=mask), x2)
    out2.sum().backward()
    grad_x2 = x2.grad.copy()
    param_grads2 = [p.grad.copy() for p in block.parameters()]

    val_err = np.max(np.abs(out1.data - out2.data))
    status = "PASS" if val_err < 1e-10 else "FAIL"
    print(f"[{status}] checkpoint forward value matches direct call: max abs diff = {val_err:.2e}")
    assert val_err < 1e-10

    grad_x_err = np.max(np.abs(grad_x1 - grad_x2))
    status = "PASS" if grad_x_err < 1e-10 else "FAIL"
    print(f"[{status}] checkpoint gradient wrt input matches direct call: max abs diff = {grad_x_err:.2e}")
    assert grad_x_err < 1e-10

    max_param_err = max(np.max(np.abs(g1 - g2)) for g1, g2 in zip(param_grads1, param_grads2))
    status = "PASS" if max_param_err < 1e-10 else "FAIL"
    print(f"[{status}] checkpoint gradients wrt block parameters match direct call: max abs diff = {max_param_err:.2e}")
    assert max_param_err < 1e-10


def test_checkpoint_chain_of_blocks():
    np.random.seed(1)
    seq, d_model, n_heads, d_ff = 8, 12, 3, 24
    blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(3)]
    mask = causal_mask(seq)
    x_data = RNG.standard_normal((1, seq, d_model))

    # direct
    x1 = Tensor(x_data.copy(), requires_grad=True)
    h = x1
    for b in blocks:
        b.zero_grad()
        h = b(h, mask=mask)
    h.sum().backward()
    grad_direct = x1.grad.copy()

    # checkpointed chain
    x2 = Tensor(x_data.copy(), requires_grad=True)
    h = x2
    for b in blocks:
        b.zero_grad()
        h = checkpoint(lambda xx, blk=b: blk(xx, mask=mask), h)
    h.sum().backward()
    grad_checkpointed = x2.grad.copy()

    err = np.max(np.abs(grad_direct - grad_checkpointed))
    status = "PASS" if err < 1e-9 else "FAIL"
    print(f"[{status}] checkpointed 3-block chain gradient matches direct chain: max abs diff = {err:.2e}")
    assert err < 1e-9


if __name__ == "__main__":
    test_checkpoint_matches_direct_call_transformer_block()
    test_checkpoint_chain_of_blocks()
    print("\nAll checkpoint checks passed.")