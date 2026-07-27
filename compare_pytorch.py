import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit(
        "PyTorch is not installed. Run: pip install torch\n"
        "(this is intentionally NOT in requirements.txt -- it's a large "
        "dependency only needed for this optional verification script)"
    )

from autograd.tensor import Tensor
from autograd import ops
from autograd import attention_ops as A

RNG = np.random.default_rng(42)


def report(name, err, tol=1e-6):
    status = "PASS" if err < tol else "FAIL"
    print(f"[{status}] {name}: max abs diff vs PyTorch = {err:.2e}")
    return err < tol


def compare_add():
    a_data = RNG.standard_normal((4, 3))
    b_data = RNG.standard_normal((4, 3))

    a = Tensor(a_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)
    (a + b).sum().backward()

    ta = torch.tensor(a_data.copy(), requires_grad=True)
    tb = torch.tensor(b_data.copy(), requires_grad=True)
    (ta + tb).sum().backward()

    err = max(np.max(np.abs(a.grad - ta.grad.numpy())),
              np.max(np.abs(b.grad - tb.grad.numpy())))
    return report("add", err)


def compare_matmul():
    a_data = RNG.standard_normal((4, 5))
    w_data = RNG.standard_normal((5, 3))

    a = Tensor(a_data.copy(), requires_grad=True)
    w = Tensor(w_data.copy(), requires_grad=True)
    a.matmul(w).sum().backward()

    ta = torch.tensor(a_data.copy(), requires_grad=True)
    tw = torch.tensor(w_data.copy(), requires_grad=True)
    (ta @ tw).sum().backward()

    err = max(np.max(np.abs(a.grad - ta.grad.numpy())),
              np.max(np.abs(w.grad - tw.grad.numpy())))
    return report("matmul", err)


def compare_relu():
    x_data = RNG.standard_normal((10,))
    x = Tensor(x_data.copy(), requires_grad=True)
    ops.relu(x).sum().backward()

    tx = torch.tensor(x_data.copy(), requires_grad=True)
    torch.relu(tx).sum().backward()

    err = np.max(np.abs(x.grad - tx.grad.numpy()))
    return report("relu", err)


def compare_softmax():
    x_data = RNG.standard_normal((4, 6))
    weights_data = RNG.standard_normal((4, 6))

    x = Tensor(x_data.copy(), requires_grad=True)
    weights = Tensor(weights_data.copy())
    (ops.softmax(x, axis=-1) * weights).sum().backward()

    tx = torch.tensor(x_data.copy(), requires_grad=True)
    tw = torch.tensor(weights_data.copy())
    (torch.softmax(tx, dim=-1) * tw).sum().backward()

    err = np.max(np.abs(x.grad - tx.grad.numpy()))
    return report("softmax", err)


def compare_layer_norm():
    n, d = 5, 8
    x_data = RNG.standard_normal((n, d))
    gamma_data = RNG.standard_normal((d,)) * 0.5 + 1.0
    beta_data = RNG.standard_normal((d,)) * 0.1

    x = Tensor(x_data.copy(), requires_grad=True)
    gamma = Tensor(gamma_data.copy(), requires_grad=True)
    beta = Tensor(beta_data.copy(), requires_grad=True)
    A.layer_norm(x, gamma, beta).sum().backward()

    tx = torch.tensor(x_data.copy(), requires_grad=True)
    tgamma = torch.tensor(gamma_data.copy(), requires_grad=True)
    tbeta = torch.tensor(beta_data.copy(), requires_grad=True)
    tout = torch.nn.functional.layer_norm(tx, (d,), weight=tgamma, bias=tbeta, eps=1e-5)
    tout.sum().backward()

    err = max(
        np.max(np.abs(x.grad - tx.grad.numpy())),
        np.max(np.abs(gamma.grad - tgamma.grad.numpy())),
        np.max(np.abs(beta.grad - tbeta.grad.numpy())),
    )
    return report("layer_norm", err)


def compare_gelu():
    x_data = RNG.standard_normal((10,))
    x = Tensor(x_data.copy(), requires_grad=True)
    A.gelu(x).sum().backward()

    tx = torch.tensor(x_data.copy(), requires_grad=True)
    torch.nn.functional.gelu(tx, approximate="none").sum().backward()

    err = np.max(np.abs(x.grad - tx.grad.numpy()))
    return report("gelu", err)


def compare_scaled_dot_product_attention():
    seq, d_k, d_v = 6, 8, 6
    q_data = RNG.standard_normal((seq, d_k))
    k_data = RNG.standard_normal((seq, d_k))
    v_data = RNG.standard_normal((seq, d_v))

    q = Tensor(q_data.copy(), requires_grad=True)
    k = Tensor(k_data.copy(), requires_grad=True)
    v = Tensor(v_data.copy(), requires_grad=True)
    out, _ = A.scaled_dot_product_attention(q, k, v)
    out.sum().backward()

    tq = torch.tensor(q_data.copy(), requires_grad=True)
    tk = torch.tensor(k_data.copy(), requires_grad=True)
    tv = torch.tensor(v_data.copy(), requires_grad=True)
    scale = 1.0 / (d_k ** 0.5)
    scores = (tq @ tk.T) * scale
    weights = torch.softmax(scores, dim=-1)
    tout = weights @ tv
    tout.sum().backward()

    err = max(
        np.max(np.abs(q.grad - tq.grad.numpy())),
        np.max(np.abs(k.grad - tk.grad.numpy())),
        np.max(np.abs(v.grad - tv.grad.numpy())),
    )
    return report("scaled_dot_product_attention", err)


if __name__ == "__main__":
    print(f"Comparing against PyTorch {torch.__version__}\n")
    results = [
        compare_add(),
        compare_matmul(),
        compare_relu(),
        compare_softmax(),
        compare_layer_norm(),
        compare_gelu(),
        compare_scaled_dot_product_attention(),
    ]
    print()
    if all(results):
        print("All ops match PyTorch's autograd exactly (within floating point tolerance).")
    else:
        print("Some ops diverged from PyTorch -- see FAIL lines above.")
