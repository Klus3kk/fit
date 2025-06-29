"""
Implementation of activation functions for neural networks.
"""

import numpy as np

from fit.core.tensor import Tensor
from fit.nn.modules.base import Layer


class ReLU(Layer):
    def forward(self, x):
        out = Tensor((x.data > 0) * x.data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                grad = (x.data > 0).astype(
                    float
                ) * out.grad  # Multiply with upstream grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class Softmax(Layer):
    def forward(self, x: Tensor, axis=-1):
        """
        Apply softmax function along specified axis.

        Args:
            x: Input tensor
            axis: Axis along which to apply softmax

        Returns:
            Softmax output
        """
        # Numerically stable softmax
        x_max = Tensor(np.max(x.data, axis=axis, keepdims=True))
        x_shifted = x - x_max
        exp_x = x_shifted.exp()
        sum_exp = Tensor(np.sum(exp_x.data, axis=axis, keepdims=True))

        out = exp_x / sum_exp

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Softmax gradient: softmax * (grad - (softmax * grad).sum())
                s = out.data
                grad_sum = np.sum(out.grad * s, axis=axis, keepdims=True)
                grad = s * (out.grad - grad_sum)
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class Tanh(Layer):
    def forward(self, x):
        out = Tensor(np.tanh(x.data), requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Derivative of tanh is 1 - tanh^2
                grad = out.grad * (1 - out.data * out.data)
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class Sigmoid(Layer):
    def forward(self, x):
        # Numerically stable sigmoid
        out_data = np.where(
            x.data >= 0,
            1 / (1 + np.exp(-x.data)),
            np.exp(x.data) / (1 + np.exp(x.data)),
        )
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Derivative of sigmoid is sigmoid * (1 - sigmoid)
                grad = out.grad * out.data * (1 - out.data)
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class LeakyReLU(Layer):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out_data = np.where(x.data > 0, x.data, self.negative_slope * x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                grad = np.where(x.data > 0, 1.0, self.negative_slope) * out.grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class ELU(Layer):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        out_data = np.where(x.data > 0, x.data, self.alpha * (np.exp(x.data) - 1))
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                grad = np.where(x.data > 0, 1.0, out.data + self.alpha) * out.grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class GELU(Layer):
    def forward(self, x):
        # Gaussian Error Linear Unit: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x.data + 0.044715 * x.data**3)))
        out_data = x.data * cdf
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Approximate GELU derivative
                tanh_arg = sqrt_2_over_pi * (x.data + 0.044715 * x.data**3)
                tanh_val = np.tanh(tanh_arg)
                sech2 = 1 - tanh_val**2

                grad = 0.5 * (1 + tanh_val) + x.data * 0.5 * sech2 * sqrt_2_over_pi * (
                    1 + 3 * 0.044715 * x.data**2
                )
                grad = grad * out.grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class Swish(Layer):
    def forward(self, x):
        # Swish: x * sigmoid(x)
        sigmoid_data = 1 / (1 + np.exp(-np.clip(x.data, -88, 88)))
        out_data = x.data * sigmoid_data
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Derivative: sigmoid + x * sigmoid * (1 - sigmoid)
                sigmoid_val = 1 / (1 + np.exp(-np.clip(x.data, -88, 88)))
                grad = sigmoid_val + x.data * sigmoid_val * (1 - sigmoid_val)
                grad = grad * out.grad
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # Create dropout mask
        mask = np.random.binomial(1, 1 - self.p, x.data.shape) / (1 - self.p)
        out_data = x.data * mask
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad and out.grad is not None:
                grad = out.grad * mask
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class LogSoftmax(Layer):
    def forward(self, x: Tensor, axis=-1):
        """
        Apply log-softmax function along specified axis.

        Args:
            x: Input tensor
            axis: Axis along which to apply log-softmax

        Returns:
            Log-softmax output
        """
        # Numerically stable log-softmax: x - logsumexp(x)
        x_max = Tensor(np.max(x.data, axis=axis, keepdims=True))
        x_shifted = x - x_max
        exp_x = x_shifted.exp()
        sum_exp = Tensor(np.sum(exp_x.data, axis=axis, keepdims=True))
        log_sum_exp = sum_exp.log() + x_max

        out = x - log_sum_exp

        def _backward():
            if x.requires_grad and out.grad is not None:
                # Log-softmax gradient: grad - softmax * grad.sum()
                softmax = exp_x / sum_exp
                grad_sum = np.sum(out.grad, axis=axis, keepdims=True)
                grad = out.grad - softmax.data * grad_sum
                x.grad = grad if x.grad is None else x.grad + grad

        out._backward = _backward
        out._prev = {x}
        return out
