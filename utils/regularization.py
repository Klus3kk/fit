from core.tensor import Tensor
import numpy as np


def apply_l2_regularization(model, loss, lambda_reg=1e-5):
    """
    Applies L2 regularization to model parameters and adds to loss
    """
    l2_reg = Tensor(0.0, requires_grad=True)

    for param in model.parameters():
        if param.requires_grad:
            l2_reg = l2_reg + (param * param).sum()

    return loss + lambda_reg * l2_reg