import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autograd.tensor import Tensor
from autograd import ops
from autograd import attention_ops as A
from nn import Module, Linear, causal_mask, TransformerBlock
from optimizers import Adam, AdamW
from data import load_digit_images, batch_iter, CharTokenizer, get_lm_batch
from profiling import profile_attention_memory_scaling

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

PLOT_STYLE = {"figsize": (8, 5), "dpi": 120}


def plot_gradcheck_errors():
    from tests.test_engine import check as engine_check, RNG as engine_rng
    import tests.test_engine as te
    import tests.test_attention as ta

    errors = {}

    def run_and_capture(name, build_fn, shape):
        x_data = te.RNG.standard_normal(shape)
        x = Tensor(x_data.copy(), requires_grad=True)
        loss = build_fn(x)
        loss.backward()
        ag = x.grad.copy()

        def f(xd):
            xt = Tensor(xd, requires_grad=False)
            return build_fn(xt).data.sum()

        ng = te.numerical_grad(f, x_data.copy())
        errors[name] = np.max(np.abs(ag - ng))

    other = Tensor(te.RNG.standard_normal((4, 3)))
    run_and_capture("add", lambda x: (x + other).sum(), (4, 3))
    run_and_capture("mul", lambda x: (x * other).sum(), (4, 3))
    w = Tensor(te.RNG.standard_normal((3, 5)))
    run_and_capture("matmul", lambda x: x.matmul(w).sum(), (4, 3))
    run_and_capture("relu", lambda x: ops.relu(x).sum(), (6,))
    fixed_weights = Tensor(te.RNG.standard_normal((4, 5)))
    run_and_capture("softmax", lambda x: (ops.softmax(x, axis=-1) * fixed_weights).sum(), (4, 5))
    run_and_capture("sum", lambda x: x.sum(), (3, 4))
    run_and_capture("reshape", lambda x: x.reshape(2, 6).sum(), (3, 4))
    run_and_capture("pow", lambda x: (x ** 2).sum(), (5,))

    # attention ops
    x_data = ta.RNG.standard_normal((10,))
    x = Tensor(x_data.copy(), requires_grad=True)
    out = A.gelu(x)
    out.sum().backward()
    ag = x.grad.copy()
    ng = ta.numerical_grad(lambda xd: A.gelu(Tensor(xd)).data.sum(), x_data.copy())
    errors["gelu"] = np.max(np.abs(ag - ng))

    n, d = 3, 6
    x_data = ta.RNG.standard_normal((n, d))
    gamma = Tensor(np.ones(d), requires_grad=True)
    beta = Tensor(np.zeros(d), requires_grad=True)
    x = Tensor(x_data.copy(), requires_grad=True)
    A.layer_norm(x, gamma, beta).sum().backward()
    ag = x.grad.copy()
    ng = ta.numerical_grad(lambda xd: A.layer_norm(Tensor(xd), gamma, beta).data.sum(), x_data.copy())
    errors["layer_norm"] = np.max(np.abs(ag - ng))

    names = list(errors.keys())
    vals = [max(errors[n], 1e-16) for n in names]  

    fig, ax = plt.subplots(**PLOT_STYLE)
    bars = ax.bar(names, vals, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_ylabel("max abs error (analytic vs numerical gradient)")
    ax.set_title("Gradcheck results: autograd vs. finite-difference gradients")
    ax.axhline(1e-4, color="red", linestyle="--", linewidth=1, label="pass threshold (1e-4)")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gradcheck_errors.png")
    plt.savefig(path)
    plt.close()
    print(f"saved {path}")



def plot_mnist_training():
    class MLP(Module):
        def __init__(self, in_dim=64, hidden=64, out_dim=10):
            self.fc1 = Linear(in_dim, hidden)
            self.fc2 = Linear(hidden, out_dim)

        def forward(self, x):
            return self.fc2(ops.relu(self.fc1(x)))

    def cross_entropy_loss(logits, y):
        probs = ops.softmax(logits, axis=-1)
        batch = y.shape[0]
        true_probs = probs[np.arange(batch), y]
        eps = 1e-12
        log_data = np.log(true_probs.data + eps)
        out = Tensor(log_data, requires_grad=true_probs.requires_grad,
                     _children=(true_probs,), _op="log")

        def _backward():
            if true_probs.requires_grad:
                true_probs.grad += out.grad / (true_probs.data + eps)

        out._backward = _backward
        return -out.mean()

    def accuracy(logits, y):
        return float(np.mean(np.argmax(logits.data, axis=-1) == y))

    np.random.seed(0)
    rng = np.random.default_rng(0)
    X_train, y_train, X_test, y_test = load_digit_images(seed=0)
    model = MLP()
    optimizer = Adam(model.parameters(), lr=1e-2)

    losses, accs = [], []
    for epoch in range(15):
        epoch_losses = []
        for X_batch, y_batch in batch_iter(X_train, y_train, 32, rng):
            logits = model(Tensor(X_batch))
            loss = cross_entropy_loss(logits, y_batch)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(np.mean(epoch_losses))
        accs.append(accuracy(model(Tensor(X_test)), y_test))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    ax1.plot(range(1, 16), losses, marker="o", color="#C44E52")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax1.set_title("MLP training loss (digit classification)")

    ax2.plot(range(1, 16), accs, marker="o", color="#55A868")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0.8, 1.0)
    ax2.set_title(f"MLP test accuracy (final: {accs[-1]:.2%})")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "mnist_training.png")
    plt.savefig(path)
    plt.close()
    print(f"saved {path}")


def plot_transformer_training_and_attention():
    from nn import Embedding, LayerNorm
    from train_transformer import (TinyTransformerLM, cross_entropy_loss_seq, load_corpus)

    np.random.seed(0)
    rng = np.random.default_rng(0)

    text = load_corpus()
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    seq_len = 48
    batch_size = 16
    model = TinyTransformerLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4,
                               n_layers=3, d_ff=256, max_seq_len=seq_len)
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    steps_recorded, losses = [], []
    n_steps = 300
    for step in range(1, n_steps + 1):
        x, y = get_lm_batch(data, batch_size, seq_len, rng)
        logits = model(x)
        loss = cross_entropy_loss_seq(logits, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            steps_recorded.append(step)
            losses.append(loss.item())

    fig, ax = plt.subplots(**PLOT_STYLE)
    ax.plot(steps_recorded, losses, color="#8172B2")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Transformer LM training loss (character-level)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "transformer_loss.png")
    plt.savefig(path)
    plt.close()
    print(f"saved {path}")

    prompt = "the fox drinks from the stream"
    ids = tokenizer.encode(prompt)
    context = np.array(ids)[None, :]
    seq = context.shape[1]

    tok = model.token_emb(context)
    pos = model.pos_emb(np.arange(seq))
    x = tok + pos
    mask = causal_mask(seq)

    block = model.blocks[0]
    ln_x = block.ln1(x)
    batch = 1
    q = block.attn._split_heads(block.attn.w_q(ln_x), batch, seq)
    k = block.attn._split_heads(block.attn.w_k(ln_x), batch, seq)
    v = block.attn._split_heads(block.attn.w_v(ln_x), batch, seq)
    _, weights = A.scaled_dot_product_attention(q, k, v, mask=mask)
    w_head0 = weights.data[0, 0]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    im = ax.imshow(w_head0, cmap="viridis")
    chars = list(prompt)
    ax.set_xticks(range(seq))
    ax.set_xticklabels(chars, fontsize=8)
    ax.set_yticks(range(seq))
    ax.set_yticklabels(chars, fontsize=8)
    ax.set_xlabel("attending TO (key position)")
    ax.set_ylabel("attending FROM (query position)")
    ax.set_title("Causal self-attention weights (layer 1, head 1)\nafter training")
    plt.colorbar(im, label="attention weight")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "attention_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"saved {path}")


def plot_memory_scaling():
    results = profile_attention_memory_scaling(verbose=False)
    seq_lens = [r["seq_len"] for r in results]
    mb = [r["score_matrix_mb"] for r in results]
    fwd = [r["forward_ms"] for r in results]
    bwd = [r["backward_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    ax1.plot(seq_lens, mb, marker="o", color="#C44E52")
    ax1.set_xlabel("sequence length")
    ax1.set_ylabel("attention score matrix size (MB)")
    ax1.set_title("Attention memory: O(seq_len\u00b2) growth")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)

    ax2.plot(seq_lens, fwd, marker="o", label="forward", color="#4C72B0")
    ax2.plot(seq_lens, bwd, marker="o", label="backward", color="#55A868")
    ax2.set_xlabel("sequence length")
    ax2.set_ylabel("time (ms)")
    ax2.set_title("Attention forward/backward wall-clock time")
    ax2.set_xscale("log", base=2)
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "memory_scaling.png")
    plt.savefig(path)
    plt.close()
    print(f"saved {path}")


if __name__ == "__main__":
    print("Generating gradcheck error plot...")
    plot_gradcheck_errors()
    print("\nTraining MLP and generating loss/accuracy plots...")
    plot_mnist_training()
    print("\nTraining transformer and generating loss + attention heatmap...")
    plot_transformer_training_and_attention()
    print("\nGenerating attention memory scaling plot...")
    plot_memory_scaling()
    print(f"\nAll plots saved to {OUT_DIR}/")
