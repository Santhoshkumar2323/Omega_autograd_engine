import numpy as np

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def get_lm_batch(data: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator):
    max_start = len(data) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([data[s:s + seq_len] for s in starts])
    y = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y

def load_digit_images(test_size: float = 0.2, seed: int = 0):
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0  
    y = digits.target.astype(np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_test = int(len(X) * test_size)
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]
    return X_train, y_train, X_test, y_test


def batch_iter(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator, shuffle=True):
    n = len(X)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]
