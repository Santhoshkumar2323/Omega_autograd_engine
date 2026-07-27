import numpy as np
from .tensor import Tensor, _unbroadcast


def add(a: Tensor, b: Tensor) -> Tensor:
    return a + b


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)


def relu(x: Tensor) -> Tensor:
    mask = (x.data > 0).astype(np.float64)
    out = Tensor(x.data * mask, requires_grad=x.requires_grad,
                 _children=(x,), _op="relu")

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * mask

    out._backward = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    s = exp / np.sum(exp, axis=axis, keepdims=True)

    out = Tensor(s, requires_grad=x.requires_grad, _children=(x,), _op="softmax")

    def _backward():
        if x.requires_grad:
            g = out.grad
            dot = np.sum(g * s, axis=axis, keepdims=True)
            x.grad += s * (g - dot)

    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor(s, requires_grad=x.requires_grad, _children=(x,), _op="sigmoid")

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * s * (1.0 - s)

    out._backward = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    t = np.tanh(x.data)
    out = Tensor(t, requires_grad=x.requires_grad, _children=(x,), _op="tanh")

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * (1.0 - t ** 2)

    out._backward = _backward
    return out
