from core.tensor import Tensor
from nn.layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier/Glorot Initialization
        weight = np.random.randn(in_features, out_features) * (1 / np.sqrt(in_features))
        bias = np.zeros(out_features)

        self.weight = Tensor(weight, requires_grad=True)
        self.bias = Tensor(bias, requires_grad=True)

        self.add_parameter(self.weight)
        self.add_parameter(self.bias)

    def forward(self, x):
        return x @ self.weight + self.bias
