# self-pruning-neural-network
self pruning neural network using pytorch
# Self-Pruning Neural Network

This project implements a neural network that learns to prune its own weights during training using learnable gates.

## Features
- Custom PrunableLinear layer
- L1 sparsity loss
- CIFAR-10 training
- Trade-off between accuracy and sparsity

## Results
Lambda 0.01:
- Accuracy: ~44%
- Sparsity: ~58%

## Tech
- Python
- PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# custom layer
class PrunableLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.gate = nn.Parameter(torch.randn(out_f, in_f))

    def forward(self, x):
        g = torch.sigmoid(self.gate)
        w = self.weight * g
        return F.linear(x, w, self.bias)


# model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = PrunableLinear(3072, 512)
        self.l2 = PrunableLinear(512, 256)
        self.l3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# data
transform = transforms.ToTensor()

train_data = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

test_data = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)


# sparsity loss
def get_sparse_loss(model):
    s = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = torch.sigmoid(m.gate)
            s += g.sum()
    return s


# sparsity %
def get_sparsity(model):
    total = 0
    zero = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = torch.sigmoid(m.gate)
            total += g.numel()
            zero += (g < 1e-2).sum().item()
    return 100 * zero / total


# ===========================
# TRAIN FOR DIFFERENT LAMBDA
# ===========================
for lam in [0.001, 0.01, 0.05]:

    print("\n==========================")
    print("Lambda:", lam)
    print("==========================")

    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # more epochs for better pruning
    for e in range(10):
        total_loss = 0

        for x, y in train_loader:

            out = model(x)

            loss1 = loss_fn(out, y)
            loss2 = get_sparse_loss(model)

            loss = loss1 + lam * loss2

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print("epoch", e+1, "loss", round(total_loss, 2))

    # accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            _, pred = torch.max(out, 1)

            total += y.size(0)
            correct += (pred == y).sum().item()

    acc = 100 * correct / total
    sparsity = get_sparsity(model)

    print("accuracy:", round(acc, 2), "%")
    print("sparsity:", round(sparsity, 2), "%")

    # plot only for middle lambda
    if lam == 0.01:
        vals = []
        for m in model.modules():
            if isinstance(m, PrunableLinear):
                g = torch.sigmoid(m.gate).detach().numpy().flatten()
                vals.extend(g)

        plt.hist(vals, bins=50)
        plt.title("gate values (lambda = 0.01)")
        plt.xlabel("gate value")
        plt.ylabel("count")
        plt.show()
