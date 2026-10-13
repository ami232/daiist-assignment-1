"""One linear equation, two ways to train it in PyTorch."""
import torch
from torch import nn

DTYPE = torch.float64

class LinearDemand(nn.Module):
    def __init__(self, number_of_features):
        super().__init__()
        self.linear = nn.Linear(number_of_features, 1, dtype=DTYPE)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

def train_manual(x, y, regularization, learning_rate, max_epochs=20000):
    # Raw leaf tensors: no nn.Module or optimizer in this method.
    weights = torch.zeros(x.shape[1], dtype=DTYPE, requires_grad=True)
    bias = torch.zeros((), dtype=DTYPE, requires_grad=True)
    history = []
    for epoch in range(max_epochs):
        prediction = x @ weights + bias
        loss = ((prediction - y) ** 2).mean() + regularization * (weights ** 2).sum()
        loss.backward()
        largest_gradient = max(weights.grad.abs().max().item(), bias.grad.abs().item())
        if epoch % 250 == 0 or largest_gradient < 1e-8 or epoch == max_epochs - 1:
            history.append({"epoch": epoch, "objective": loss.item(), "max_gradient": largest_gradient})
        if largest_gradient < 1e-8:
            break
        # Do not record the update as part of the autograd graph.
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad
        # Gradients accumulate, so clear them before the next iteration.
        weights.grad.zero_()
        bias.grad.zero_()
    return {"weights": weights.detach(), "bias": bias.detach()}, history

def train_standard(x, y, regularization, learning_rate, max_epochs=20000):
    model = LinearDemand(x.shape[1])
    # Both loops start at zero for a fair numerical comparison.
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = ((prediction - y) ** 2).mean() + regularization * (model.linear.weight ** 2).sum()
        loss.backward()
        largest_gradient = max(p.grad.abs().max().item() for p in model.parameters())
        if epoch % 250 == 0 or largest_gradient < 1e-8 or epoch == max_epochs - 1:
            history.append({"epoch": epoch, "objective": loss.item(), "max_gradient": largest_gradient})
        if largest_gradient < 1e-8:
            break
        optimizer.step()
    model.eval()
    return model, history
