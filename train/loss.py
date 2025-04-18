from core.tensor import Tensor
from nn.layer import Layer
import numpy as np

class MSELoss(Layer):
    def forward(self, prediction: Tensor, target: Tensor):
        diff = prediction - target
        return (diff * diff).mean()
    
class CrossEntropyLoss(Layer):
    def forward(self, logits: Tensor, target: Tensor):
        logits_data = logits.data
        target_data = target.data.astype(int)
        batch_size = logits_data.shape[0]

        shifted_logits = logits_data - np.max(logits_data, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        log_likelihood = -np.log(probs[np.arange(batch_size), target_data])
        loss_value = np.mean(log_likelihood)

        out = Tensor(loss_value)
        out.requires_grad = True

        def _backward():
            if logits.requires_grad:
                grad = probs
                grad[np.arange(batch_size), target_data] -= 1
                grad /= batch_size
                grad = grad * out.grad  # <== THIS IS THE MISSING LINK
                logits.grad = grad if logits.grad is None else logits.grad + grad

        out._backward = _backward
        out._prev = {logits}
        return out



