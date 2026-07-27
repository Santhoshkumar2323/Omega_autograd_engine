import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd import attention_ops as A

RNG = np.random.default_rng(1)


def numerical_grad(f, x: np.ndarray, eps=1e-6):
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


def check_wrt(name, build_fn, param_getter, param_setter, shape, tol=1e-4):
    x0 = param_getter().copy()

    loss = build_fn()
    loss.backward()
    autograd_grad = None  

    def f(xd):
        param_setter(xd)
        return build_fn().data.sum()

    num_grad = numerical_grad(f, x0.copy())
    param_setter(x0)  
    return num_grad


def test_masked_softmax():
    x_data = RNG.standard_normal((2, 4))
    mask = np.array([[0, 0, -1e9, -1e9],
                      [0, 0, 0, 0]])

    x = Tensor(x_data.copy(), requires_grad=True)
    weights_fixed = Tensor(RNG.standard_normal((2, 4)))
    out = A.masked_softmax(x, mask=mask, axis=-1)
    loss = (out * weights_fixed).sum()
    loss.backward()
    autograd_grad = x.grad.copy()

    def f(xd):
        xt = Tensor(xd, requires_grad=False)
        out = A.masked_softmax(xt, mask=mask, axis=-1)
        return (out.data * weights_fixed.data).sum()

    num_grad = numerical_grad(f, x_data.copy())
    err = np.max(np.abs(autograd_grad - num_grad))
    status = "PASS" if err < 1e-4 else "FAIL"
    print(f"[{status}] masked_softmax: max abs error = {err:.2e}")
    assert err < 1e-4

    assert out.data[0, 2] < 1e-6 and out.data[0, 3] < 1e-6, \
        "masked positions should receive ~zero softmax weight"


def test_scaled_dot_product_attention():
    seq, d_k, d_v = 4, 8, 6
    q_data = RNG.standard_normal((seq, d_k))
    k_data = RNG.standard_normal((seq, d_k))
    v_data = RNG.standard_normal((seq, d_v))

    q = Tensor(q_data.copy(), requires_grad=True)
    k = Tensor(k_data, requires_grad=False)
    v = Tensor(v_data, requires_grad=False)

    out, _ = A.scaled_dot_product_attention(q, k, v)
    loss = out.sum()
    loss.backward()
    autograd_grad = q.grad.copy()

    def f(qd):
        qt = Tensor(qd, requires_grad=False)
        out, _ = A.scaled_dot_product_attention(qt, k, v)
        return out.data.sum()

    num_grad = numerical_grad(f, q_data.copy())
    err = np.max(np.abs(autograd_grad - num_grad))
    status = "PASS" if err < 1e-4 else "FAIL"
    print(f"[{status}] scaled_dot_product_attention (wrt q): max abs error = {err:.2e}")
    assert err < 1e-4


def test_causal_mask_attention():
    seq, d_k = 4, 6
    q = Tensor(RNG.standard_normal((seq, d_k)))
    k = Tensor(RNG.standard_normal((seq, d_k)))
    v = Tensor(RNG.standard_normal((seq, d_k)))

    causal_mask = np.triu(np.full((seq, seq), -1e9), k=1)  
    out, weights = A.scaled_dot_product_attention(q, k, v, mask=causal_mask)

    upper = np.triu(weights.data, k=1)
    max_leak = np.max(np.abs(upper))
    status = "PASS" if max_leak < 1e-6 else "FAIL"
    print(f"[{status}] causal_mask blocks future positions: max leak = {max_leak:.2e}")
    assert max_leak < 1e-6


def test_layer_norm():
    n, d = 3, 6
    x_data = RNG.standard_normal((n, d))
    gamma_data = RNG.standard_normal((d,)) * 0.5 + 1.0
    beta_data = RNG.standard_normal((d,)) * 0.1

    x = Tensor(x_data.copy(), requires_grad=True)
    gamma = Tensor(gamma_data.copy(), requires_grad=True)
    beta = Tensor(beta_data.copy(), requires_grad=True)

    out = A.layer_norm(x, gamma, beta)
    loss = out.sum()
    loss.backward()

    ag_x, ag_g, ag_b = x.grad.copy(), gamma.grad.copy(), beta.grad.copy()

    def f_x(xd):
        xt = Tensor(xd, requires_grad=False)
        return A.layer_norm(xt, gamma, beta).data.sum()

    def f_g(gd):
        gt = Tensor(gd, requires_grad=False)
        return A.layer_norm(x, gt, beta).data.sum()

    def f_b(bd):
        bt = Tensor(bd, requires_grad=False)
        return A.layer_norm(x, gamma, bt).data.sum()

    ng_x = numerical_grad(f_x, x_data.copy())
    ng_g = numerical_grad(f_g, gamma_data.copy())
    ng_b = numerical_grad(f_b, beta_data.copy())

    err_x = np.max(np.abs(ag_x - ng_x))
    err_g = np.max(np.abs(ag_g - ng_g))
    err_b = np.max(np.abs(ag_b - ng_b))
    for name, err in [("layer_norm wrt x", err_x), ("layer_norm wrt gamma", err_g),
                       ("layer_norm wrt beta", err_b)]:
        status = "PASS" if err < 1e-4 else "FAIL"
        print(f"[{status}] {name}: max abs error = {err:.2e}")
        assert err < 1e-4, f"{name} failed"


    x2 = Tensor(x_data.copy())
    g1 = Tensor(np.ones(d))
    b0 = Tensor(np.zeros(d))
    norm_only = A.layer_norm(x2, g1, b0)
    row_mean = norm_only.data.mean(axis=-1)
    row_var = norm_only.data.var(axis=-1)
    assert np.allclose(row_mean, 0, atol=1e-6), "layer_norm output mean should be ~0"
    assert np.allclose(row_var, 1, atol=1e-4), "layer_norm output var should be ~1"
    print("[PASS] layer_norm sanity: normalized rows have mean~0, var~1")


def test_gelu():
    x_data = RNG.standard_normal((10,))
    x = Tensor(x_data.copy(), requires_grad=True)
    out = A.gelu(x)
    loss = out.sum()
    loss.backward()
    autograd_grad = x.grad.copy()

    def f(xd):
        xt = Tensor(xd, requires_grad=False)
        return A.gelu(xt).data.sum()

    num_grad = numerical_grad(f, x_data.copy())
    err = np.max(np.abs(autograd_grad - num_grad))
    status = "PASS" if err < 1e-4 else "FAIL"
    print(f"[{status}] gelu: max abs error = {err:.2e}")
    assert err < 1e-4


if __name__ == "__main__":
    test_masked_softmax()
    test_scaled_dot_product_attention()
    test_causal_mask_attention()
    test_layer_norm()
    test_gelu()
    print("\nAll attention gradchecks passed.")
