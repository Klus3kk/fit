# FIT: Flexible and Interpretable Training Library

[![Tests](https://github.com/Klus3kk/fit/actions/workflows/ci.yml/badge.svg)](https://github.com/Klus3kk/fit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Klus3kk/fit/branch/main/graph/badge.svg)](https://codecov.io/gh/Klus3kk/fit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FIT is a lightweight machine learning library built from scratch in Python with NumPy. It provides a PyTorch-like API with automatic differentiation and neural network components.

## Features

- **Automatic Differentiation**: Build and train neural networks with automatic differentiation
- **Neural Network Components**: Linear layers, activations (ReLU, Softmax), BatchNorm, Dropout
- **Optimizers**: SGD, SGD with momentum, Adam
- **Training Utilities**: Training loop, learning rate schedulers, gradient clipping
- **Monitoring**: Training metrics tracking and visualization
- **Model I/O**: Save and load your trained models

## Installation

### Using pip (recommended)

```bash
pip install git+https://github.com/Klus3kk/fit.git
```

### From source

```bash
git clone https://github.com/Klus3kk/fit.git
cd fit
pip install -e .
```

### Using Docker

```bash
docker-compose up fit-ml
```

## Quick Start

```python
from fit.core.tensor import Tensor
from fit.nn.sequential import Sequential
from fit.nn.linear import Linear
from fit.nn.activations import ReLU, Softmax
from fit.train.loss import CrossEntropyLoss
from fit.train.optim import Adam
from fit.train.trainer import Trainer

# Create a simple model for binary classification
model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 2),
    Softmax()
)

# Prepare data
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR problem
y = Tensor([0, 1, 1, 0])

# Define loss function and optimizer
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Create trainer and train
trainer = Trainer(model, loss_fn, optimizer)
trainer.fit(X, y, epochs=100)

# Make predictions
predictions = model(X)
```

