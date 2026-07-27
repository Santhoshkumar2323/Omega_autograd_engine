import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autograd.tensor import Tensor
from autograd import ops

RNG = np.random.default_rng(0)


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


def check(name, build_fn, x_shape, tol=1e-4):
    x_data = RNG.standard_normal(x_shape)

    x = Tensor(x_data.copy(), requires_grad=True)
    loss = build_fn(x)
    loss.backward()
    autograd_grad = x.grad.copy()

    def f(xd):
        xt = Tensor(xd, requires_grad=False)
        return build_fn(xt).data.sum() if build_fn(xt).data.ndim > 0 else build_fn(xt).item()

    num_grad = numerical_grad(f, x_data.copy())

    err = np.max(np.abs(autograd_grad - num_grad))
    status = "PASS" if err < tol else "FAIL"
    print(f"[{status}] {name}: max abs error = {err:.2e}")
    assert err < tol, f"{name} gradcheck failed: max err {err:.2e}"


def test_add():
    other = Tensor(RNG.standard_normal((4, 3)))
    check("add", lambda x: (x + other).sum(), (4, 3))


def test_mul():
    other = Tensor(RNG.standard_normal((4, 3)))
    check("mul", lambda x: (x * other).sum(), (4, 3))


def test_matmul():
    w = Tensor(RNG.standard_normal((3, 5)))
    check("matmul", lambda x: x.matmul(w).sum(), (4, 3))


def test_relu():
    check("relu", lambda x: ops.relu(x).sum(), (6,))


def test_softmax():
    fixed_weights = Tensor(RNG.standard_normal((4, 5)))

    def loss_fn(x):
        s = ops.softmax(x, axis=-1)
        return (s * fixed_weights).sum()
    check("softmax", loss_fn, (4, 5))


def test_sum_and_mean():
    check("sum", lambda x: x.sum(), (3, 4))
    check("mean", lambda x: x.mean(), (3, 4))


def test_reshape():
    check("reshape", lambda x: x.reshape(2, 6).sum(), (3, 4))


def test_transpose():
    w = Tensor(RNG.standard_normal((3, 5)))
    check("transpose", lambda x: x.T.matmul(w).sum(), (3, 4))


def test_pow():
    check("pow", lambda x: (x ** 2).sum(), (5,))


def test_div():
    other = Tensor(RNG.standard_normal((4,)) + 3.0)  # avoid near-zero
    check("div", lambda x: (x / other).sum(), (4,))


if __name__ == "__main__":
    test_add()
    test_mul()
    test_matmul()
    test_relu()
    test_softmax()
    test_sum_and_mean()
    test_reshape()
    test_transpose()
    test_pow()
    test_div()
    print("\nAll core engine gradchecks passed.")
