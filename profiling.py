import time
import numpy as np
from autograd.tensor import Tensor
from autograd import attention_ops as A


def time_forward_backward(build_fn, n_repeats: int = 5):
    fwd_times, bwd_times = [], []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        loss = build_fn()
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)

    return {
        "forward_ms": float(np.mean(fwd_times)),
        "backward_ms": float(np.mean(bwd_times)),
        "forward_std_ms": float(np.std(fwd_times)),
        "backward_std_ms": float(np.std(bwd_times)),
    }


def profile_attention_memory_scaling(seq_lens=(16, 32, 64, 128, 256, 512),
                                      d_model: int = 64, n_heads: int = 4,
                                      verbose: bool = True):
    d_head = d_model // n_heads
    rng = np.random.default_rng(0)
    results = []

    if verbose:
        print(f"{'seq_len':>8} | {'score matrix MB':>18} | {'fwd ms':>8} | {'bwd ms':>8} | {'ratio to prev (MB)':>20}")
        print("-" * 72)

    prev_mb = None
    for seq_len in seq_lens:
        q = Tensor(rng.standard_normal((n_heads, seq_len, d_head)), requires_grad=True)
        k = Tensor(rng.standard_normal((n_heads, seq_len, d_head)), requires_grad=True)
        v = Tensor(rng.standard_normal((n_heads, seq_len, d_head)), requires_grad=True)

        def build():
            out, weights = A.scaled_dot_product_attention(q, k, v)
            return out.sum()

        timing = time_forward_backward(build, n_repeats=3)

        score_matrix_bytes = n_heads * seq_len * seq_len * 8  # float64
        score_matrix_mb = score_matrix_bytes / (1024 ** 2)
        ratio = (score_matrix_mb / prev_mb) if prev_mb else float("nan")
        prev_mb = score_matrix_mb

        results.append({
            "seq_len": seq_len,
            "score_matrix_mb": score_matrix_mb,
            "forward_ms": timing["forward_ms"],
            "backward_ms": timing["backward_ms"],
        })

        if verbose:
            ratio_str = f"{ratio:.2f}x" if not np.isnan(ratio) else "--"
            print(f"{seq_len:>8} | {score_matrix_mb:>18.4f} | {timing['forward_ms']:>8.3f} "
                  f"| {timing['backward_ms']:>8.3f} | {ratio_str:>20}")

    if verbose:
        print("\nNote: each time seq_len doubles, the score matrix grows ~4x "
              "(quadratic), while d_model/d_head stay fixed. This is the O(n^2) "
              "memory scaling that motivates FlashAttention / sparse attention "
              "for long sequences.")

    return results


if __name__ == "__main__":
    profile_attention_memory_scaling()
