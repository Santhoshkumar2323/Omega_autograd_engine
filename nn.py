import numpy as np
from autograd.tensor import Tensor
from autograd import ops
from autograd import attention_ops as A


class Module:
    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _init_weight(shape, fan_in):
    limit = 1.0 / np.sqrt(fan_in)
    return np.random.uniform(-limit, limit, size=shape)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        w = _init_weight((in_features, out_features), in_features)
        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w = np.random.normal(0, 0.02, size=(num_embeddings, embedding_dim))
        self.weight = Tensor(w, requires_grad=True)

    def forward(self, idx: np.ndarray) -> Tensor:
        out_data = self.weight.data[idx]
        out = Tensor(out_data, requires_grad=self.weight.requires_grad,
                     _children=(self.weight,), _op="embedding")

        def _backward():
            if self.weight.requires_grad:
                np.add.at(self.weight.grad, idx, out.grad)

        out._backward = _backward
        return out


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return A.layer_norm(x, self.gamma, self.beta, axis=-1, eps=self.eps)


class MultiHeadAttention(Module):
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = Linear(d_model, d_model, bias=False)
        self.w_k = Linear(d_model, d_model, bias=False)
        self.w_v = Linear(d_model, d_model, bias=False)
        self.w_o = Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: Tensor, batch: int, seq: int) -> Tensor:
        # (batch, seq, d_model) -> (batch, n_heads, seq, d_head)
        x = x.reshape(batch, seq, self.n_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor, batch: int, seq: int) -> Tensor:
        # (batch, n_heads, seq, d_head) -> (batch, seq, d_model)
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq, self.d_model)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        batch, seq, _ = x.shape

        q = self._split_heads(self.w_q(x), batch, seq)
        k = self._split_heads(self.w_k(x), batch, seq)
        v = self._split_heads(self.w_v(x), batch, seq)

        attn_out, _ = A.scaled_dot_product_attention(q, k, v, mask=mask)
        merged = self._merge_heads(attn_out, batch, seq)
        return self.w_o(merged)


class FeedForward(Module):

    def __init__(self, d_model: int, d_ff: int):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(A.gelu(self.fc1(x)))


class TransformerBlock(Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


def causal_mask(seq_len: int) -> np.ndarray:
    return np.triu(np.full((seq_len, seq_len), -1e9), k=1)
