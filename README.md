# FIT
A PyTorch-like machine learning library built from scratch with NumPy. Train neural networks with automatic differentiation, no dependencies beyond NumPy.

[Documentation](https://fit-ml.readthedocs.io/) (for now it's incomplete, will finish it soon c:) | [Examples](examples/)

## Why FIT?

- **Lightweight**: Only requires NumPy
- **Educational**: Understand ML from first principles
- **Familiar API**: PyTorch-like interface
- **Production ready**: Type hints, logging, proper error handling

## Installation

```bash
pip install git+https://github.com/Klus3kk/fit.git
```

## Example

Solve XOR problem:

```python
from fit.core.tensor import Tensor
from fit.nn.modules.container import Sequential
from fit.nn.modules.linear import Linear
from fit.nn.modules.activation import ReLU
from fit.loss.regression import MSELoss
from fit.optim.adam import Adam

# XOR dataset
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = Tensor([[0], [1], [1], [0]])

# Model
model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1)
)

# Training
loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    pred = model(X)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Loss: {loss.data:.4f}")

# Test
print(f"Predictions: {model(X).data}")
```

## What's included

**Core**: Tensors with autograd, just like PyTorch
```python
x = Tensor([1, 2, 3], requires_grad=True)
y = x.sum()
y.backward()  # x.grad is now [1, 1, 1]
```

**Layers**: Linear, activations, normalization, attention
```python
from fit.nn.modules.activation import ReLU, GELU
from fit.nn.modules.normalization import BatchNorm1d
```

**Optimizers**: SGD, Adam, and advanced ones like SAM
```python
from fit.optim.adam import Adam
from fit.optim.experimental.sam import SAM
```

**Simple API**: For quick experiments
```python
from fit.simple.models import Classifier
model = Classifier(input_size=784, num_classes=10)
```

## MNIST example

```python
import numpy as np
from fit.core.tensor import Tensor
from fit.nn.modules.container import Sequential
from fit.nn.modules.linear import Linear
from fit.nn.modules.activation import ReLU
from fit.optim.adam import Adam
from fit.loss.classification import CrossEntropyLoss

# Create MNIST classifier 
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 64), 
    ReLU(),
    Linear(64, 10)
)

# Generate some dummy data 
X_train = Tensor(np.random.randn(100, 784))
y_train = Tensor(np.random.randint(0, 10, (100,)))

optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss()

# Training loop
for epoch in range(50):
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

Perfect for learning how neural networks work under the hood, or when you need a lightweight ML library without the complexity of PyTorch.
