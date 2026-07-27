import numpy as np
from .tensor import Tensor


def masked_softmax(x: Tensor, mask=None, axis: int = -1) -> Tensor:
    data = x.data
    if mask is not None:
        data = data + mask  

    shifted = data - np.max(data, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    s = exp / np.sum(exp, axis=axis, keepdims=True)

    out = Tensor(s, requires_grad=x.requires_grad, _children=(x,), _op="masked_softmax")

    def _backward():
        if x.requires_grad:
            g = out.grad
            dot = np.sum(g * s, axis=axis, keepdims=True)
            x.grad += s * (g - dot)

    out._backward = _backward
    return out


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
    d_k = q.shape[-1]
    scale = 1.0 / np.sqrt(d_k)

    scores = q.matmul(k.transpose(*range(k.ndim - 2), k.ndim - 1, k.ndim - 2)) * scale
    weights = masked_softmax(scores, mask=mask, axis=-1)
    out = weights.matmul(v)
    return out, weights


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, axis: int = -1, eps: float = 1e-5) -> Tensor:
    mu = np.mean(x.data, axis=axis, keepdims=True)
    var = np.var(x.data, axis=axis, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x.data - mu) * std_inv

    y = gamma.data * x_hat + beta.data

    req = x.requires_grad or gamma.requires_grad or beta.requires_grad
    out = Tensor(y, requires_grad=req, _children=(x, gamma, beta), _op="layer_norm")

    n = x.data.shape[axis]

    def _backward():
        g = out.grad

        if gamma.requires_grad:
            sum_axes = tuple(i for i in range(g.ndim) if i != (g.ndim + axis) % g.ndim)
            gamma.grad += np.sum(g * x_hat, axis=sum_axes) if sum_axes else np.sum(g * x_hat, axis=None)
            # reshape in case gamma is 1D over the feature axis
            gamma.grad = gamma.grad.reshape(gamma.data.shape)
        if beta.requires_grad:
            sum_axes = tuple(i for i in range(g.ndim) if i != (g.ndim + axis) % g.ndim)
            beta.grad += np.sum(g, axis=sum_axes) if sum_axes else np.sum(g, axis=None)
            beta.grad = beta.grad.reshape(beta.data.shape)

        if x.requires_grad:
            dx_hat = g * gamma.data
            dvar_term = np.sum(dx_hat * (x.data - mu), axis=axis, keepdims=True) * (-0.5) * std_inv ** 3
            dmu_term = np.sum(dx_hat * -std_inv, axis=axis, keepdims=True) + \
                dvar_term * np.mean(-2.0 * (x.data - mu), axis=axis, keepdims=True)
            dx = dx_hat * std_inv + dvar_term * 2.0 * (x.data - mu) / n + dmu_term / n
            x.grad += dx

    out._backward = _backward
    return out


def gelu(x: Tensor) -> Tensor:
    from scipy.special import erf
    cdf = 0.5 * (1.0 + erf(x.data / np.sqrt(2.0)))
    y = x.data * cdf

    out = Tensor(y, requires_grad=x.requires_grad, _children=(x,), _op="gelu")

    def _backward():
        if x.requires_grad:
            pdf = np.exp(-0.5 * x.data ** 2) / np.sqrt(2.0 * np.pi)
            dgelu = cdf + x.data * pdf
            x.grad += out.grad * dgelu

    out._backward = _backward
    return out
