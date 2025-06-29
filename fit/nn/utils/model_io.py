import json
import os
import pickle

import numpy as np

from fit.core.tensor import Tensor


def save_model(model, path):
    """
    Save model to file.

    Args:
        model: Model to save
        path: Path to save to
    """
    # Create directory if it doesn't exist
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Collect model data
    model_data = {
        "model_type": model.__class__.__name__,
        "parameters": [],
        "config": {},
    }

    # Save parameters
    for param in model.parameters():
        model_data["parameters"].append(
            {"data": param.data.tolist(), "requires_grad": param.requires_grad}
        )

    # Save configuration if available
    if hasattr(model, "get_config"):
        model_data["config"] = model.get_config()
    elif hasattr(model, "layers") and isinstance(model.layers, list):
        # For Sequential models, save layer configurations
        layers_config = []
        for layer in model.layers:
            layer_config = {"type": layer.__class__.__name__}

            # Add layer-specific config if available
            if hasattr(layer, "get_config"):
                layer_config.update(layer.get_config())

            layers_config.append(layer_config)

        model_data["config"]["layers"] = layers_config

    # Save to file
    with open(path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {path}")


def load_model(path, model_class=None):
    """
    Load model from file.

    Args:
        path: Path to load from
        model_class: Model class to instantiate (optional)

    Returns:
        Loaded model
    """
    # Load model data
    with open(path, "rb") as f:
        model_data = pickle.load(f)

    # Instantiate model if class provided
    if model_class is not None:
        if "config" in model_data and model_data["config"]:
            model = model_class(**model_data["config"])
        else:
            model = model_class()
    else:
        # Try to infer model class from model_type
        model_type = model_data.get("model_type", "")

        if "Sequential" in model_type:
            from fit.nn.modules.container import Sequential

            # Create empty Sequential model
            model = Sequential()

            # Add layers if config is available
            if "config" in model_data and "layers" in model_data["config"]:
                layers_config = model_data["config"]["layers"]

                for layer_config in layers_config:
                    layer_type = layer_config.pop("type", "")

                    # Import and instantiate each layer
                    if layer_type == "Linear":
                        from fit.nn.modules.linear import Linear

                        in_features = layer_config.get("in_features")
                        out_features = layer_config.get("out_features")
                        layer = Linear(in_features, out_features)
                    elif layer_type == "ReLU":
                        from fit.nn.modules.activation import ReLU

                        layer = ReLU()
                    elif layer_type == "Softmax":
                        from fit.nn.modules.activation import Softmax

                        layer = Softmax()
                    elif layer_type == "Dropout":
                        from fit.nn.modules.activation import Dropout

                        p = layer_config.get("p", 0.5)
                        layer = Dropout(p)
                    elif layer_type == "BatchNorm":
                        from fit.nn.modules.normalization import BatchNorm

                        num_features = layer_config.get("num_features")
                        eps = layer_config.get("eps", 1e-5)
                        momentum = layer_config.get("momentum", 0.1)
                        layer = BatchNorm(num_features, eps, momentum)
                    else:
                        raise ValueError(f"Unknown layer type: {layer_type}")

                    model.layers.append(layer)
        else:
            raise ValueError(f"Cannot instantiate model of type {model_type}")

    # Load parameters
    if "parameters" in model_data:
        params = model.parameters()

        if len(params) != len(model_data["parameters"]):
            raise ValueError(
                "Model has "
                + str(len(params))
                + " parameters, but "
                + str(len(model_data["parameters"]))
                + " were saved"
            )

        for i, param_data in enumerate(model_data["parameters"]):
            params[i].data = np.array(param_data["data"])
            params[i].requires_grad = param_data["requires_grad"]

    print(f"Model loaded from {path}")
    return model
