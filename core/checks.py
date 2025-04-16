import numpy as np

def check_gradients(model, x, y, loss_fn, epsilon=1e-5, threshold=1e-3):
    """
    Compares analytical gradients from backprop to numerical gradients.
    """
    # Forward and backward to compute analytical gradients
    preds = model(x)
    loss = loss_fn(preds, y)
    loss.backward()

    print("Running gradient check...")

    for param in model.parameters():
        if not param.requires_grad:
            continue

        assert param.grad is not None, "Parameter has no gradient"

        grad_analytic = param.grad
        grad_numeric = np.zeros_like(param.data)

        # Compute numerical gradient using central difference
        for idx in np.ndindex(param.data.shape):
            old_value = param.data[idx]

            param.data[idx] = old_value + epsilon
            loss_plus = loss_fn(model(x), y).data

            param.data[idx] = old_value - epsilon
            loss_minus = loss_fn(model(x), y).data

            param.data[idx] = old_value  # reset

            grad_numeric[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        relative_error = np.abs(grad_analytic - grad_numeric).mean() / (
            np.abs(grad_analytic).mean() + np.abs(grad_numeric).mean() + 1e-8
        )

        print(f"  Param shape: {param.data.shape}")
        print(f"  Mean relative error: {relative_error:.6e}")

        assert relative_error < threshold, f"Gradient check failed! Relative error: {relative_error}"
