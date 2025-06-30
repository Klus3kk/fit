"""
Container modules for composing neural networks.
"""

from typing import List, Iterator, Dict, Any
from collections import OrderedDict

from fit.core.tensor import Tensor
from fit.nn.modules.base import Layer


class Sequential(Layer):
    """
    Sequential container that chains layers together.

    Layers are executed in the order they are added.
    """

    def __init__(self, *layers):
        """
        Initialize sequential container.

        Args:
            *layers: Variable number of layers to chain
        """
        super().__init__()
        self.layers = []

        for layer in layers:
            self.add_layer(layer)

    def add_layer(self, layer: Layer):
        """Add a layer to the sequence."""
        self.layers.append(layer)
        # Register parameters from the added layer
        for param in layer.parameters():
            self.add_parameter(param)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers in sequence.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def train(self):
        """Set all layers to training mode."""
        super().train()
        for layer in self.layers:
            layer.train()

    def eval(self):
        """Set all layers to evaluation mode."""
        super().eval()
        for layer in self.layers:
            layer.eval()

    def get_config(self):
        """Get configuration for serialization."""
        layers_config = []
        for layer in self.layers:
            layer_config = {"type": layer.__class__.__name__}
            if hasattr(layer, "get_config"):
                layer_config.update(layer.get_config())
            layers_config.append(layer_config)
        return {"layers": layers_config}
    
    


class ModuleList(Layer):
    """
    List container for modules.

    Unlike Sequential, ModuleList doesn't define forward pass -
    you need to implement it yourself.
    """

    def __init__(self, modules: List[Layer] = None):
        """
        Initialize module list.

        Args:
            modules: List of modules to store
        """
        super().__init__()
        self.modules = []

        if modules is not None:
            for module in modules:
                self.append(module)

    def append(self, module: Layer):
        """Add a module to the end of the list."""
        self.modules.append(module)
        for param in module.parameters():
            self.add_parameter(param)

    def extend(self, modules: List[Layer]):
        """Extend the list with multiple modules."""
        for module in modules:
            self.append(module)

    def insert(self, index: int, module: Layer):
        """Insert a module at the given index."""
        self.modules.insert(index, module)
        for param in module.parameters():
            self.add_parameter(param)

    def __len__(self) -> int:
        return len(self.modules)

    def __getitem__(self, idx: int) -> Layer:
        return self.modules[idx]

    def __setitem__(self, idx: int, module: Layer):
        old_module = self.modules[idx]
        # Remove old parameters (this is simplified - full implementation would track parameters)
        self.modules[idx] = module
        for param in module.parameters():
            self.add_parameter(param)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.modules)

    def train(self):
        """Set all modules to training mode."""
        super().train()
        for module in self.modules:
            module.train()

    def eval(self):
        """Set all modules to evaluation mode."""
        super().eval()
        for module in self.modules:
            module.eval()


class ModuleDict(Layer):
    """
    Dictionary container for modules.
    """

    def __init__(self, modules: Dict[str, Layer] = None):
        """
        Initialize module dictionary.

        Args:
            modules: Dictionary of modules
        """
        super().__init__()
        self.modules = OrderedDict()

        if modules is not None:
            for key, module in modules.items():
                self[key] = module

    def __getitem__(self, key: str) -> Layer:
        return self.modules[key]

    def __setitem__(self, key: str, module: Layer):
        self.modules[key] = module
        for param in module.parameters():
            self.add_parameter(param)

    def __delitem__(self, key: str):
        del self.modules[key]

    def __len__(self) -> int:
        return len(self.modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self.modules)

    def keys(self):
        return self.modules.keys()

    def values(self):
        return self.modules.values()

    def items(self):
        return self.modules.items()

    def update(self, modules: Dict[str, Layer]):
        """Update with multiple modules."""
        for key, module in modules.items():
            self[key] = module

    def train(self):
        """Set all modules to training mode."""
        super().train()
        for module in self.modules.values():
            module.train()

    def eval(self):
        """Set all modules to evaluation mode."""
        super().eval()
        for module in self.modules.values():
            module.eval()


class Parallel(Layer):
    """
    Parallel container that applies multiple layers to the same input.
    """

    def __init__(self, *layers):
        """
        Initialize parallel container.

        Args:
            *layers: Layers to apply in parallel
        """
        super().__init__()
        self.layers = []

        for layer in layers:
            self.add_layer(layer)

    def add_layer(self, layer: Layer):
        """Add a layer to parallel execution."""
        self.layers.append(layer)
        for param in layer.parameters():
            self.add_parameter(param)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Apply all layers to input in parallel.

        Args:
            x: Input tensor

        Returns:
            List of outputs from each layer
        """
        outputs = []
        for layer in self.layers:
            outputs.append(layer(x))
        return outputs

    def train(self):
        """Set all layers to training mode."""
        super().train()
        for layer in self.layers:
            layer.train()

    def eval(self):
        """Set all layers to evaluation mode."""
        super().eval()
        for layer in self.layers:
            layer.eval()


class Residual(Layer):
    """
    Residual connection: output = input + layer(input)
    """

    def __init__(self, layer: Layer):
        """
        Initialize residual connection.

        Args:
            layer: Layer to wrap with residual connection
        """
        super().__init__()
        self.layer = layer
        for param in layer.parameters():
            self.add_parameter(param)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            x + layer(x)
        """
        return x + self.layer(x)

    def train(self):
        """Set layer to training mode."""
        super().train()
        self.layer.train()

    def eval(self):
        """Set layer to evaluation mode."""
        super().eval()
        self.layer.eval()


class Highway(Layer):
    """
    Highway connection: output = gate * layer(input) + (1 - gate) * input
    """

    def __init__(self, layer: Layer, gate_layer: Layer = None):
        """
        Initialize highway connection.

        Args:
            layer: Transform layer
            gate_layer: Gate layer (if None, creates a linear layer)
        """
        super().__init__()
        self.layer = layer

        if gate_layer is None:
            # Create default gate layer - requires knowing input size
            # This is a simplified version
            from fit.nn.modules.linear import Linear
            from fit.nn.modules.activation import Sigmoid

            self.gate = Sequential(
                Linear(layer.in_features, layer.in_features), Sigmoid()
            )
        else:
            self.gate = gate_layer

        for param in self.layer.parameters():
            self.add_parameter(param)
        for param in self.gate.parameters():
            self.add_parameter(param)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with highway connection.

        Args:
            x: Input tensor

        Returns:
            gate * layer(x) + (1 - gate) * x
        """
        transform = self.layer(x)
        gate = self.gate(x)
        return gate * transform + (Tensor(1.0) - gate) * x

    def train(self):
        """Set layers to training mode."""
        super().train()
        self.layer.train()
        self.gate.train()

    def eval(self):
        """Set layers to evaluation mode."""
        super().eval()
        self.layer.eval()
        self.gate.eval()
