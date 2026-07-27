import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd import attention_ops as A
from autograd.flash_attention import flash_attention

RNG = np.random.default_rng(7)


def numerical_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
    return grad


def test_forward_matches_naive():
    seq_q, seq_k, d_k, d_v = 37, 41, 16, 12  
    q_data = RNG.standard_normal((seq_q, d_k))
    k_data = RNG.standard_normal((seq_k, d_k))
    v_data = RNG.standard_normal((seq_k, d_v))

    q, k, v = Tensor(q_data), Tensor(k_data), Tensor(v_data)
    naive_out, _ = A.scaled_dot_product_attention(q, k, v)

    q2, k2, v2 = Tensor(q_data), Tensor(k_data), Tensor(v_data)
    flash_out = flash_attention(q2, k2, v2, block_size=16)

    err = np.max(np.abs(naive_out.data - flash_out.data))
    status = "PASS" if err < 1e-8 else "FAIL"
    print(f"[{status}] flash vs naive forward values: max abs diff = {err:.2e}")
    assert err < 1e-8


def test_gradients_match_naive():
    seq_q, seq_k, d_k, d_v = 20, 24, 8, 6
    q_data = RNG.standard_normal((seq_q, d_k))
    k_data = RNG.standard_normal((seq_k, d_k))
    v_data = RNG.standard_normal((seq_k, d_v))

    # naive
    q1 = Tensor(q_data.copy(), requires_grad=True)
    k1 = Tensor(k_data.copy(), requires_grad=True)
    v1 = Tensor(v_data.copy(), requires_grad=True)
    naive_out, _ = A.scaled_dot_product_attention(q1, k1, v1)
    naive_out.sum().backward()

    # flash
    q2 = Tensor(q_data.copy(), requires_grad=True)
    k2 = Tensor(k_data.copy(), requires_grad=True)
    v2 = Tensor(v_data.copy(), requires_grad=True)
    flash_out = flash_attention(q2, k2, v2, block_size=7)  # awkward block size on purpose
    flash_out.sum().backward()

    for name, g1, g2 in [("q", q1.grad, q2.grad), ("k", k1.grad, k2.grad), ("v", v1.grad, v2.grad)]:
        err = np.max(np.abs(g1 - g2))
        status = "PASS" if err < 1e-6 else "FAIL"
        print(f"[{status}] flash vs naive gradient (wrt {name}): max abs diff = {err:.2e}")
        assert err < 1e-6


def test_gradcheck_finite_difference():
    seq_q, seq_k, d_k, d_v = 12, 15, 6, 5
    q_data = RNG.standard_normal((seq_q, d_k))
    k = Tensor(RNG.standard_normal((seq_k, d_k)))
    v = Tensor(RNG.standard_normal((seq_k, d_v)))

    q = Tensor(q_data.copy(), requires_grad=True)
    out = flash_attention(q, k, v, block_size=5)
    out.sum().backward()
    ag = q.grad.copy()

    def f(qd):
        qt = Tensor(qd, requires_grad=False)
        return flash_attention(qt, k, v, block_size=5).data.sum()

    ng = numerical_grad(f, q_data.copy())
    err = np.max(np.abs(ag - ng))
    status = "PASS" if err < 1e-4 else "FAIL"
    print(f"[{status}] flash attention finite-diff gradcheck (wrt q): max abs error = {err:.2e}")
    assert err < 1e-4


def test_causal_mask():
    from nn import causal_mask
    seq, d_k = 16, 8
    q_data = RNG.standard_normal((seq, d_k))
    k_data = RNG.standard_normal((seq, d_k))
    v_data = RNG.standard_normal((seq, d_k))
    mask = causal_mask(seq)

    q1, k1, v1 = Tensor(q_data), Tensor(k_data), Tensor(v_data)
    naive_out, naive_w = A.scaled_dot_product_attention(q1, k1, v1, mask=mask)

    q2, k2, v2 = Tensor(q_data), Tensor(k_data), Tensor(v_data)
    flash_out = flash_attention(q2, k2, v2, mask=mask, block_size=5)

    err = np.max(np.abs(naive_out.data - flash_out.data))
    status = "PASS" if err < 1e-8 else "FAIL"
    print(f"[{status}] flash attention causal mask matches naive: max abs diff = {err:.2e}")
    assert err < 1e-8


if __name__ == "__main__":
    test_forward_matches_naive()
    test_gradients_match_naive()
    test_gradcheck_finite_difference()
    test_causal_mask()
    print("\nAll flash attention checks passed.")