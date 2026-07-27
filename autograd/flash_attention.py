import numpy as np
from autograd.tensor import Tensor


def _exp(x: Tensor) -> Tensor:
    e = np.exp(x.data)
    out = Tensor(e, requires_grad=x.requires_grad, _children=(x,), _op="exp")

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * e

    out._backward = _backward
    return out


def _elementwise_max(a: Tensor, b: Tensor) -> Tensor:
    a_wins = a.data >= b.data
    data = np.where(a_wins, a.data, b.data)
    req = a.requires_grad or b.requires_grad
    out = Tensor(data, requires_grad=req, _children=(a, b), _op="max")

    def _backward():
        if a.requires_grad:
            a.grad += out.grad * a_wins
        if b.requires_grad:
            b.grad += out.grad * (~a_wins)

    out._backward = _backward
    return out


def _row_max(x: Tensor) -> Tensor:
    data = x.data
    idx = np.argmax(data, axis=-1)
    rows = np.arange(data.shape[0])
    m = data[rows, idx].reshape(-1, 1)
    out = Tensor(m, requires_grad=x.requires_grad, _children=(x,), _op="row_max")

    def _backward():
        if x.requires_grad:
            grad = np.zeros_like(x.data)
            grad[rows, idx] = out.grad.reshape(-1)
            x.grad += grad

    out._backward = _backward
    return out


def _cat_rows(tensors) -> Tensor:
    out_data = np.concatenate([t.data for t in tensors], axis=0)
    req = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=req, _children=tuple(tensors), _op="cat_rows")
    sizes = [t.data.shape[0] for t in tensors]

    def _backward():
        start = 0
        for t, sz in zip(tensors, sizes):
            if t.requires_grad:
                t.grad += out.grad[start:start + sz]
            start += sz

    out._backward = _backward
    return out


def flash_attention(q: Tensor, k: Tensor, v: Tensor, mask=None, block_size: int = 64) -> Tensor:
    seq_q, d_k = q.shape
    seq_k, _ = k.shape
    d_v = v.shape[-1]
    scale = 1.0 / np.sqrt(d_k)

    n_q_blocks = (seq_q + block_size - 1) // block_size
    n_k_blocks = (seq_k + block_size - 1) // block_size

    out_blocks = []
    for qi in range(n_q_blocks):
        q_start, q_end = qi * block_size, min((qi + 1) * block_size, seq_q)
        Qi = q[q_start:q_end]
        bq = q_end - q_start

        O_i = Tensor(np.zeros((bq, d_v)), requires_grad=False)
        l_i = Tensor(np.zeros((bq, 1)), requires_grad=False)
        m_i = Tensor(np.full((bq, 1), -1e30), requires_grad=False)

        for kj in range(n_k_blocks):
            k_start, k_end = kj * block_size, min((kj + 1) * block_size, seq_k)
            Kj = k[k_start:k_end]
            Vj = v[k_start:k_end]

            S_ij = Qi.matmul(Kj.transpose()) * scale  # (bq, bk)
            if mask is not None:
                S_ij = S_ij + Tensor(mask[q_start:q_end, k_start:k_end], requires_grad=False)

            m_ij = _row_max(S_ij)                       # (bq, 1)
            m_new = _elementwise_max(m_i, m_ij)          # (bq, 1)

            alpha = _exp(m_i - m_new)                    # rescale factor for old accumulator
            P_ij = _exp(S_ij - m_new)                    # (bq, bk), broadcasts (bq,1) against (bq,bk)
            l_ij = P_ij.sum(axis=-1, keepdims=True)       # (bq, 1)

            l_i = alpha * l_i + l_ij
            O_i = O_i * alpha + P_ij.matmul(Vj)
            m_i = m_new

        out_blocks.append(O_i / l_i)

    return _cat_rows(out_blocks)