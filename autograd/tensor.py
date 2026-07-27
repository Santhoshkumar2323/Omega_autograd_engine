import numpy as np


def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def item(self):
        return self.data.item()

    def numpy(self):
        return self.data

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


    def backward(self, grad=None):
        topo = []
        visited = set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() with no grad argument requires a scalar output "
                    f"(got shape {self.shape}); pass an explicit grad instead."
                )
            grad = np.ones_like(self.data)

        if self.requires_grad:
            self.grad = grad.astype(np.float64)

        for v in reversed(topo):
            v._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, requires_grad=req,
                     _children=(self, other), _op="+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad,
                     _children=(self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, requires_grad=req,
                     _children=(self, other), _op="*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only scalar powers supported"
        out = Tensor(self.data ** power, requires_grad=self.requires_grad,
                     _children=(self,), _op=f"**{power}")

        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, requires_grad=req,
                     _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                if self.data.ndim == 1:
                    grad = np.outer(out.grad, other.data) if other.data.ndim == 1 \
                        else out.grad @ other.data.T
                else:
                    grad = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += _unbroadcast(grad, self.data.shape)
            if other.requires_grad:
                if other.data.ndim == 1:
                    grad = self.data.T @ out.grad
                else:
                    grad = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += _unbroadcast(grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad,
                     _children=(self,), _op="reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else None
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad,
                     _children=(self,), _op="transpose")

        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad += np.transpose(out.grad)
                else:
                    inv = np.argsort(axes)
                    self.grad += np.transpose(out.grad, inv)

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad,
                     _children=(self,), _op="getitem")

        def _backward():
            if self.requires_grad:
                g = np.zeros_like(self.data)
                g[idx] = out.grad
                self.grad += g

        out._backward = _backward
        return out
