import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def train_simple_nn(num_samples: int = 1000, hidden_size: int = 16, num_classes: int = 3,
                    epochs: int = 100, lr: float = 0.1):
    """Train a toy neural network with one hidden layer using numpy."""
    X = np.random.randn(num_samples, 10)
    y = np.random.randint(0, num_classes, size=num_samples)

    W1 = np.random.randn(10, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, num_classes) * 0.01
    b2 = np.zeros((1, num_classes))

    for epoch in range(epochs):
        z1 = X.dot(W1) + b1
        a1 = relu(z1)
        logits = a1.dot(W2) + b2
        probs = softmax(logits)

        one_hot = np.eye(num_classes)[y]
        loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-8), axis=1))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss: {loss:.4f}")

        grad_logits = probs - one_hot
        grad_W2 = a1.T.dot(grad_logits) / num_samples
        grad_b2 = grad_logits.mean(axis=0, keepdims=True)

        grad_a1 = grad_logits.dot(W2.T)
        grad_z1 = grad_a1 * (z1 > 0)
        grad_W1 = X.T.dot(grad_z1) / num_samples
        grad_b1 = grad_z1.mean(axis=0, keepdims=True)

        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1

    preds = probs.argmax(axis=1)
    accuracy = (preds == y).mean()
    print(f"Training accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    train_simple_nn()
