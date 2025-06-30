Quick Start Guide
=================

This guide will get you up and running with FIT in just a few minutes.

Basic Tensor Operations
-----------------------

FIT tensors work similarly to PyTorch tensors:

.. code-block:: python

   from fit.core.tensor import Tensor
   
   # Create tensors
   x = Tensor([1, 2, 3], requires_grad=True)
   y = Tensor([4, 5, 6], requires_grad=True)
   
   # Operations
   z = x + y
   loss = z.sum()
   
   # Backpropagation
   loss.backward()
   
   print(f"x.grad: {x.grad}")  # [1, 1, 1]

Building Your First Neural Network
-----------------------------------

Let's solve the XOR problem:

.. code-block:: python

   import numpy as np
   from fit.core.tensor import Tensor
   from fit.nn.modules.container import Sequential
   from fit.nn.modules.linear import Linear
   from fit.nn.modules.activation import ReLU
   from fit.loss.regression import MSELoss
   from fit.optim.adam import Adam

   # XOR dataset
   X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
   y = Tensor([[0], [1], [1], [0]])

   # Create model
   model = Sequential(
       Linear(2, 8),
       ReLU(),
       Linear(8, 1)
   )

   # Setup training
   loss_fn = MSELoss()
   optimizer = Adam(model.parameters(), lr=0.01)

   # Training loop
   for epoch in range(1000):
       # Forward pass
       pred = model(X)
       loss = loss_fn(pred, y)
       
       # Backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       if epoch % 200 == 0:
           print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

   # Test the model
   with torch.no_grad():
       test_pred = model(X)
       print(f"Predictions: {test_pred.data}")

Classification Example
----------------------

For classification tasks:

.. code-block:: python

   from fit.nn.modules.activation import Softmax
   from fit.loss.classification import CrossEntropyLoss

   # MNIST-like classifier
   model = Sequential(
       Linear(784, 128),
       ReLU(),
       Linear(128, 64),
       ReLU(),
       Linear(64, 10),
       Softmax()
   )

   # Dummy data
   X = Tensor(np.random.randn(100, 784))
   y = Tensor(np.random.randint(0, 10, (100,)))

   loss_fn = CrossEntropyLoss()
   optimizer = Adam(model.parameters(), lr=0.001)

   # Training loop
   for epoch in range(100):
       pred = model(X)
       loss = loss_fn(pred, y)
       
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

Available Components
--------------------

**Layers**:
- ``Linear``: Fully connected layer
- ``ReLU``, ``Sigmoid``, ``Tanh``: Activation functions
- ``Softmax``: For classification
- ``BatchNorm1d``: Batch normalization

**Optimizers**:
- ``SGD``: Stochastic gradient descent
- ``Adam``: Adaptive moment estimation
- ``SAM``: Sharpness-aware minimization

**Loss Functions**:
- ``MSELoss``: Mean squared error
- ``CrossEntropyLoss``: For classification
- ``HuberLoss``: Robust regression loss

Next Steps
----------

- Check out the :doc:`tutorials/index` for more detailed examples
- Browse the :doc:`api/core` for complete API reference
- See :doc:`examples/basic` for common use cases