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
