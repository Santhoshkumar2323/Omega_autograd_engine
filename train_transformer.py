import numpy as np

from autograd.tensor import Tensor
from autograd import ops
from nn import Module, Linear, Embedding, LayerNorm, TransformerBlock, causal_mask
from optimizers import AdamW
from data import CharTokenizer, get_lm_batch


SAMPLE_TEXT = """
the quick brown fox jumps over the lazy dog. the dog barks back and the fox
runs into the forest. deep in the forest there is an old oak tree, and under
the oak tree there is a small stream. the stream flows down to the river,
and the river flows down to the sea. the fox drinks from the stream every
morning before the sun rises, and every evening it returns to its den under
the roots of the old oak tree. the forest is quiet and full of small sounds:
birds calling, leaves rustling, and the distant murmur of the stream.
""".strip()


def load_corpus() -> str:
    return SAMPLE_TEXT * 40  


class TinyTransformerLM(Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=3, d_ff=256, max_seq_len=64):
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_seq_len, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx: np.ndarray) -> Tensor:
        batch, seq_len = idx.shape
        assert seq_len <= self.max_seq_len, "sequence longer than max_seq_len"

        tok = self.token_emb(idx)                                  
        pos_idx = np.arange(seq_len)
        pos = self.pos_emb(pos_idx)                                 
        x = tok + pos                                                

        mask = causal_mask(seq_len)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)                                   
        return logits


def cross_entropy_loss_seq(logits: Tensor, y: np.ndarray) -> Tensor:
    batch, seq, vocab = logits.shape
    flat_logits = logits.reshape(batch * seq, vocab)
    probs = ops.softmax(flat_logits, axis=-1)

    flat_y = y.reshape(-1)
    true_probs = probs[np.arange(batch * seq), flat_y]

    eps = 1e-12
    log_true_probs_data = np.log(true_probs.data + eps)
    out = Tensor(log_true_probs_data, requires_grad=true_probs.requires_grad,
                 _children=(true_probs,), _op="log")

    def _backward():
        if true_probs.requires_grad:
            true_probs.grad += out.grad / (true_probs.data + eps)

    out._backward = _backward
    return -out.mean()


@np.errstate(over="ignore")
def generate(model: TinyTransformerLM, tokenizer: CharTokenizer, prompt: str,
             n_new_tokens: int = 100, temperature: float = 0.8, rng=None):
    rng = rng or np.random.default_rng(0)
    ids = list(tokenizer.encode(prompt))

    for _ in range(n_new_tokens):
        context = np.array(ids[-model.max_seq_len:])[None, :]  # (1, seq)
        logits = model(context)
        last_logits = logits.data[0, -1] / temperature
        probs = np.exp(last_logits - last_logits.max())
        probs /= probs.sum()
        next_id = rng.choice(len(probs), p=probs)
        ids.append(int(next_id))

    return tokenizer.decode(ids)


def main():
    np.random.seed(0)
    rng = np.random.default_rng(0)

    text = load_corpus()
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    print(f"corpus length: {len(text)} chars | vocab size: {tokenizer.vocab_size}")

    seq_len = 48
    batch_size = 16
    model = TinyTransformerLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4,
                               n_layers=3, d_ff=256, max_seq_len=seq_len)
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    n_steps = 300
    for step in range(1, n_steps + 1):
        x, y = get_lm_batch(data, batch_size, seq_len, rng)
        logits = model(x)
        loss = cross_entropy_loss_seq(logits, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    print("\n--- sample generation ---")
    sample = generate(model, tokenizer, prompt="the fox", n_new_tokens=150, rng=rng)
    print(sample)


if __name__ == "__main__":
    main()
