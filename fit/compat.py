"""Backward compatibility imports code"""

import warnings

def _deprecated_import(old_path, new_path):
    warnings.warn(
        f"Import from '{old_path}' is deprecated. "
        f"Please use '{new_path}' instead.",
        DeprecationWarning,
        stacklevel=2
    )

# Maintain old import paths temporarily
try:
    from fit.nn.modules.linear import Linear as _Linear
    Linear = _Linear
    _deprecated_import("nn.linear", "fit.nn.modules.linear")
except ImportError:
    pass