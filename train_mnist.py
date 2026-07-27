import numpy as np
from autograd.tensor import Tensor
from autograd import ops
from nn import Module, Linear
from optimizers import Adam
from data import load_digit_images, batch_iter


class MLP(Module):
    def __init__(self, in_dim=64, hidden=64, out_dim=10):
        self.fc1 = Linear(in_dim, hidden)
        self.fc2 = Linear(hidden, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = ops.relu(self.fc1(x))
        return self.fc2(h)


def cross_entropy_loss(logits: Tensor, y: np.ndarray) -> Tensor:
    probs = ops.softmax(logits, axis=-1)
    batch = y.shape[0]
    true_probs = probs[np.arange(batch), y]
    eps = 1e-12
    log_true_probs_data = np.log(true_probs.data + eps)
    out = Tensor(log_true_probs_data, requires_grad=true_probs.requires_grad,
                 _children=(true_probs,), _op="log")

    def _backward():
        if true_probs.requires_grad:
            true_probs.grad += out.grad / (true_probs.data + eps)

    out._backward = _backward
    loss = -out.mean()
    return loss


def accuracy(logits: Tensor, y: np.ndarray) -> float:
    preds = np.argmax(logits.data, axis=-1)
    return float(np.mean(preds == y))


def main():
    rng = np.random.default_rng(0)
    np.random.seed(0)

    X_train, y_train, X_test, y_test = load_digit_images(test_size=0.2, seed=0)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    model = MLP(in_dim=64, hidden=64, out_dim=10)
    optimizer = Adam(model.parameters(), lr=1e-2)

    n_epochs = 15
    batch_size = 32

    for epoch in range(1, n_epochs + 1):
        epoch_losses = []
        for X_batch, y_batch in batch_iter(X_train, y_train, batch_size, rng):
            x = Tensor(X_batch, requires_grad=False)
            logits = model(x)
            loss = cross_entropy_loss(logits, y_batch)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        test_logits = model(Tensor(X_test, requires_grad=False))
        test_acc = accuracy(test_logits, y_test)
        print(f"epoch {epoch:2d} | train loss {np.mean(epoch_losses):.4f} | test acc {test_acc:.4f}")

    final_logits = model(Tensor(X_test, requires_grad=False))
    print(f"\nFinal test accuracy: {accuracy(final_logits, y_test):.4f}")


if __name__ == "__main__":
    main()
