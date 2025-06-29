"""
Base classes for neural network modules.
"""

from typing import List, Iterator, Dict, Any
from abc import ABC, abstractmethod

from fit.core.tensor import Tensor


class Layer(ABC):
    """
    Base class for all neural network layers.

    All layers should inherit from this class and implement the forward method.
    """

    def __init__(self):
        """Initialize the layer."""
        self._parameters = []
        self._modules = {}
        self.training = True

    def add_parameter(self, param: Tensor):
        """
        Add a parameter to this layer.

        Args:
            param: Parameter tensor to add
        """
        if param not in self._parameters:
            self._parameters.append(param)

    def parameters(self) -> List[Tensor]:
        """
        Return all parameters of this layer.

        Returns:
            List of parameter tensors
        """
        params = list(self._parameters)

        # Add parameters from child modules
        for module in self._modules.values():
            params.extend(module.parameters())

        return params

    def named_parameters(self, prefix: str = "") -> Iterator[tuple]:
        """
        Return iterator over module parameters with names.

        Args:
            prefix: Prefix to add to parameter names

        Yields:
            (name, parameter) tuples
        """
        for i, param in enumerate(self._parameters):
            name = f"{prefix}param_{i}" if prefix else f"param_{i}"
            yield name, param

        for name, module in self._modules.items():
            submodule_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from module.named_parameters(submodule_prefix)

    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters():
            param.zero_grad()

    def train(self):
        """Set the layer to training mode."""
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        """Set the layer to evaluation mode."""
        self.training = False
        for module in self._modules.values():
            module.eval()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the layer.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def __call__(self, *args, **kwargs):
        """
        Make the layer callable.

        This calls the forward method.
        """
        return self.forward(*args, **kwargs)

    def __repr__(self):
        """Return string representation of the layer."""
        extra_repr = self.extra_repr()
        if extra_repr:
            return f"{self.__class__.__name__}({extra_repr})"
        return f"{self.__class__.__name__}()"

    def extra_repr(self) -> str:
        """
        Return extra representation string for this layer.

        Subclasses can override this to provide additional information.
        """
        return ""

    def state_dict(self) -> Dict[str, Any]:
        """
        Return state dictionary containing layer's state.

        Returns:
            Dictionary mapping parameter names to their values
        """
        state = {}

        # Add own parameters
        for i, param in enumerate(self._parameters):
            state[f"param_{i}"] = param.data.copy()

        # Add child module states
        for name, module in self._modules.items():
            module_state = module.state_dict()
            for key, value in module_state.items():
                state[f"{name}.{key}"] = value

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state from state dictionary.

        Args:
            state_dict: Dictionary containing state to load
        """
        # Load own parameters
        for i, param in enumerate(self._parameters):
            key = f"param_{i}"
            if key in state_dict:
                param.data = state_dict[key].copy()

        # Load child module states
        for name, module in self._modules.items():
            module_state = {}
            prefix = f"{name}."

            for key, value in state_dict.items():
                if key.startswith(prefix):
                    module_key = key[len(prefix) :]
                    module_state[module_key] = value

            if module_state:
                module.load_state_dict(module_state)

    def apply(self, fn):
        """
        Apply function to all parameters.

        Args:
            fn: Function to apply to each parameter
        """
        for param in self._parameters:
            fn(param)

        for module in self._modules.values():
            module.apply(fn)

    def cuda(self):
        """Move layer to CUDA (placeholder - not implemented)."""
        print("CUDA support not implemented YET in FIT framework")
        return self

    def cpu(self):
        """Move layer to CPU (already on CPU)."""
        return self

    def to(self, device):
        """Move layer to specified device (placeholder)."""
        if device != "cpu":
            print(f"Device '{device}' not supported in FIT framework")
        return self

    def add_child(self, module: "Layer"):
        """
        Add a child module to this layer.

        Args:
            module: Child module to add
        """
        # Generate a unique name for the module
        name = f"child_{len(self._modules)}"
        self._modules[name] = module


class Module(Layer):
    """
    Alias for Layer class to match PyTorch naming convention.
    """

    pass


class Identity(Layer):
    """
    Identity layer that returns input unchanged.
    """

    def forward(self, x):
        return x


class Lambda(Layer):
    """
    Layer that applies a function to its input.
    """

    def __init__(self, func):
        """
        Initialize lambda layer.

        Args:
            func: Function to apply
        """
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class MultiInputLayer(Layer):
    """
    Base class for layers that take multiple inputs.
    """

    def forward(self, *inputs):
        """
        Forward pass with multiple inputs.

        Args:
            *inputs: Variable number of input tensors
        """
        raise NotImplementedError("Subclasses must implement forward method")


class ParameterList(Layer):
    """
    Container for a list of parameters.
    """

    def __init__(self, parameters=None):
        """
        Initialize parameter list.

        Args:
            parameters: Initial list of parameters
        """
        super().__init__()
        if parameters is not None:
            for param in parameters:
                self.append(param)

    def append(self, parameter: Tensor):
        """Add a parameter to the list."""
        self.add_parameter(parameter)

    def extend(self, parameters: List[Tensor]):
        """Extend the list with multiple parameters."""
        for param in parameters:
            self.append(param)

    def forward(self, x):
        """Parameter lists don't have forward pass."""
        raise RuntimeError("ParameterList has no forward method")


class ParameterDict(Layer):
    """
    Container for a dictionary of parameters.
    """

    def __init__(self, parameters=None):
        """
        Initialize parameter dictionary.

        Args:
            parameters: Initial dictionary of parameters
        """
        super().__init__()
        self._param_dict = {}

        if parameters is not None:
            for key, param in parameters.items():
                self[key] = param

    def __setitem__(self, key: str, parameter: Tensor):
        """Set a parameter in the dictionary."""
        self._param_dict[key] = parameter
        self.add_parameter(parameter)

    def __getitem__(self, key: str) -> Tensor:
        """Get a parameter from the dictionary."""
        return self._param_dict[key]

    def __delitem__(self, key: str):
        """Delete a parameter from the dictionary."""
        del self._param_dict[key]
        # Note: We don't remove from _parameters list for simplicity

    def keys(self):
        return self._param_dict.keys()

    def values(self):
        return self._param_dict.values()

    def items(self):
        return self._param_dict.items()

    def forward(self, x):
        """Parameter dicts don't have forward pass."""
        raise RuntimeError("ParameterDict has no forward method")
