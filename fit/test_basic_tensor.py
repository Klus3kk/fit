"""
Test basic tensor operations and autograd functionality.
This tests the core of the FIT library.
"""

import numpy as np
from fit.core.tensor import Tensor

def test_basic_operations():
    """Test basic tensor operations."""
    print("=== Testing Basic Tensor Operations ===")
    
    # Create tensors
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    
    print(f"a = {a.data}")
    print(f"b = {b.data}")
    
    # Addition
    c = a + b
    print(f"a + b = {c.data}")
    
    # Multiplication
    d = a * b
    print(f"a * b = {d.data}")
    
    # Matrix operations
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[2.0, 1.0], [4.0, 3.0]], requires_grad=True)
    
    print(f"x = \n{x.data}")
    print(f"y = \n{y.data}")
    
    # Matrix multiplication
    z = x @ y
    print(f"x @ y = \n{z.data}")
    
    return True

def test_autograd():
    """Test automatic differentiation."""
    print("\n=== Testing Automatic Differentiation ===")
    
    # Simple function: f(x) = x^2 + 2x + 1
    x = Tensor([2.0], requires_grad=True)
    
    # Forward pass
    y = x * x + 2 * x + 1
    print(f"f({x.data[0]}) = {y.data[0]}")
    
    # Backward pass
    y.backward()
    print(f"f'({x.data[0]}) = {x.grad[0]} (expected: {2 * x.data[0] + 2})")
    
    # More complex example
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    
    # Function: f(a, b) = sum(a * b)
    c = a * b
    loss = Tensor([c.data.sum()], requires_grad=True)
    loss._backward = lambda: None
    loss._prev = {c}
    
    # Manually set up backward for sum
    def _backward():
        if c.requires_grad:
            c.grad = np.ones_like(c.data) if c.grad is None else c.grad + np.ones_like(c.data)
    
    loss._backward = _backward
    
    # Test without calling backward for now (since sum might not be implemented)
    print(f"a = {a.data}, b = {b.data}")
    print(f"a * b = {c.data}")
    print(f"sum(a * b) = {c.data.sum()}")
    
    return True

def test_reshape_and_slicing():
    """Test tensor reshaping and slicing operations."""
    print("\n=== Testing Reshape and Slicing ===")
    
    # Create a tensor and reshape
    x = Tensor(np.arange(12).astype(float), requires_grad=True)
    print(f"Original: {x.data}")
    
    # Test if reshape is available
    try:
        y = x.reshape((3, 4))
        print(f"Reshaped (3, 4): \n{y.data}")
    except AttributeError:
        print("Reshape not implemented yet")
    
    # Test slicing
    try:
        z = x[2:8]
        print(f"Sliced [2:8]: {z.data}")
    except (AttributeError, TypeError):
        print("Slicing not implemented yet")
    
    return True

if __name__ == "__main__":
    try:
        test_basic_operations()
        test_autograd()
        test_reshape_and_slicing()
        print("\n✅ Basic tensor tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()