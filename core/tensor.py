import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = np.ones_like(other.data) * out.grad
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        out._prev = set()
        if self.requires_grad:
            out._prev.add(self)
        if other.requires_grad:
            out._prev.add(other)

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = self.data * out.grad
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        out._prev = set()
        if self.requires_grad:
            out._prev.add(self)
        if other.requires_grad:
            out._prev.add(other)

        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other**-1

    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = power * self.data ** (power - 1) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        out._prev = set()
        if self.requires_grad:
            out._prev.add(self)
        if other.requires_grad:
            out._prev.add(other)

        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def sum(self, axis=None, keepdims=False):
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Broadcast grad to match input shape
                shape = np.ones_like(self.data).sum(axis=axis, keepdims=keepdims).shape
                grad = np.broadcast_to(grad, shape)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data) * out.grad / self.data.size
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.exp(self.data) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = (1 / self.data) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out._prev = {self}
        return out

    def max(self, axis=None, keepdims=False):
        data = self.data
        max_data = np.max(data, axis=axis, keepdims=keepdims)
        out = Tensor(max_data)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                # Handle multiple max values per row (non-unique max)
                mask = self.data == np.max(self.data, axis=axis, keepdims=True)
                grad[mask] = 1.0  # distribute equally if multiple max
                grad_tensor = Tensor(grad)
                if not keepdims and axis is not None:
                    grad_tensor = grad_tensor.reshape(out.shape)  # shape match
                self.backward(grad_tensor)

        out._backward = _backward
        out._prev = {self}
        return out

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        print("Backward on:", self.data)
        print("Initial grad:", self.grad)

        visited = set()
        order = []

        def topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    topo(child)
                order.append(tensor)

        topo(self)
        for tensor in reversed(order):
            if tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data)
            tensor._backward()

