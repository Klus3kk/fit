Installation
============

Requirements
------------

FIT requires:

- Python 3.9 or higher
- NumPy 1.21.0 or higher
- PyYAML 6.0 or higher

Install from GitHub
-------------------

The recommended way to install FIT is directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/Klus3kk/fit.git

Install from Source
-------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/Klus3kk/fit.git
   cd fit
   pip install -e .

For development with all dependencies:

.. code-block:: bash

   pip install -e ".[dev,examples]"

Verify Installation
-------------------

To verify your installation works:

.. code-block:: python

   import fit
   from fit.core.tensor import Tensor
   
   x = Tensor([1, 2, 3])
   print(f"FIT installed successfully! Tensor: {x}")

Docker Installation
-------------------

You can also use Docker:

.. code-block:: bash

   docker-compose up fit-ml

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error**: Make sure you have Python 3.9+ and NumPy installed.

**Module Not Found**: Ensure you're using the correct import paths (e.g., ``from fit.core.tensor import Tensor``).

**Performance Issues**: FIT is CPU-only. For GPU acceleration, consider using with other frameworks.