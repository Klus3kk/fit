import numpy as np

from core.tensor import Tensor
from nn.layer import Layer


class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for layer in self.layers:
            self.add_child(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def summary(self, input_shape):
        print("╭" + "─" * 68 + "╮")
        print(f"│ {'Layer':<20} | {'Output Shape':<20} | {'# Params':<12} │")
        print("├" + "─" * 68 + "┤")

        x = Tensor(np.zeros((1, *input_shape)), requires_grad=False)
        total_params = 0

        for layer in self.layers:
            x = layer(x)
            shape = tuple(x.data.shape)
            params = sum(np.prod(p.data.shape) for p in layer.parameters())
            total_params += params
            print(
                "| "
                + f"{layer.__class__.__name__:<20} | {str(shape):<20} | {params:<12}"
                + " |"
            )

        print("╰" + "─" * 68 + "╯")
        print(f"Total trainable parameters: {total_params}")

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "layers": [
                {
                    "type": layer.__class__.__name__,
                    **(layer.get_config() if hasattr(layer, "get_config") else {}),
                }
                for layer in self.layers
            ]
        }

    def train(self):
        """Set model to training mode (for layers like Dropout, BatchNorm)."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True
            if hasattr(layer, "train"):
                layer.train()
        return self

    def eval(self):
        """Set model to evaluation mode (for layers like Dropout, BatchNorm)."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False
            if hasattr(layer, "eval"):
                layer.eval()
        return self
