import torch
from torch import nn


def train_pytorch_model(num_samples: int = 1000, input_dim: int = 10, hidden_dim: int = 32,
                        num_classes: int = 3, epochs: int = 20):
    """Train a simple feedforward network using PyTorch."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.4f}")

    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    print(f"Training accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    train_pytorch_model()
