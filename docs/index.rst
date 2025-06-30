FIT: Machine Learning Library
=============================

A PyTorch-like machine learning library built from scratch with NumPy. Train neural networks with automatic differentiation, no dependencies beyond NumPy.

.. image:: https://github.com/Klus3kk/fit/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/Klus3kk/fit/actions/workflows/ci.yml
   :alt: Tests

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Quick Start
-----------

Install FIT:

.. code-block:: bash

   pip install git+https://github.com/Klus3kk/fit.git

Solve XOR problem:

.. code-block:: python

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

Why FIT?
--------

- **Lightweight**: Only requires NumPy
- **Educational**: Understand ML from first principles  
- **Familiar API**: PyTorch-like interface
- **Production ready**: Type hints, logging, proper error handling

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/core
   api/nn
   api/optim
   api/loss
   api/data

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic
   examples/advanced

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`