"""
Example usage of Neural ODEs with the FIT framework.

This example demonstrates how to use Neural ODEs for learning
continuous dynamics, illustrated with a simple spiral dataset.
"""

import matplotlib.pyplot as plt
import numpy as np

from core.tensor import Tensor
from monitor.tracker import TrainingTracker
from nn.activations import ReLU, Tanh
from nn.linear import Linear
from nn.neural_ode import MLPODEFunction, NeuralODE
from nn.sequential import Sequential
from train.loss import MSELoss
from train.optim import Adam
from train.optim_sam import SAM


def generate_spiral_data(n_points=100, noise=0.1):
    """
    Generate a simple 2D spiral dataset.

    Args:
        n_points: Number of data points
        noise: Amount of Gaussian noise to add

    Returns:
        Tuple of (t, x0, x1) where t are time points and (x0, x1) are coordinates
    """
    t = np.linspace(0, 1, n_points)
    x0 = t * np.cos(2 * np.pi * t) + noise * np.random.randn(n_points)
    x1 = t * np.sin(2 * np.pi * t) + noise * np.random.randn(n_points)

    return t, np.stack([x0, x1], axis=1)


def run_node_example():
    """
    Example of using Neural ODEs to model a spiral trajectory.

    This shows how a Neural ODE can learn continuous dynamics from data.
    """
    # Generate spiral data
    t, x = generate_spiral_data(n_points=100, noise=0.05)

    # Create tensors for training
    # We use the initial point as input and subsequent points as targets
    x_tensor = Tensor(x[:-1], requires_grad=True)  # Initial points
    y_tensor = Tensor(x[1:], requires_grad=False)  # Target points

    # Define the ODE function as a neural network
    ode_func_model = Sequential(
        Linear(2, 16), Tanh(), Linear(16, 16), Tanh(), Linear(16, 2)
    )

    # Wrap the model in an ODEFunction
    ode_func = MLPODEFunction(ode_func_model)

    # Create a Neural ODE layer
    node = NeuralODE(
        ode_func,
        t0=0.0,
        t1=1.0 / len(t),  # Integration time is the time step in our data
        solver="rk4",
        nsteps=10,
    )

    # Create a model
    model = Sequential(node)

    # Create a loss function
    loss_fn = MSELoss()

    # Create an optimizer
    base_optimizer = Adam(model.parameters(), lr=0.01)
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)

    # Create a tracker
    tracker = TrainingTracker(experiment_name="neural_ode_spiral")

    # Define loss closure for SAM
    def loss_closure():
        outputs = model(x_tensor)
        loss = loss_fn(outputs, y_tensor)
        loss.backward()
        return loss

    # Train the model
    print("\nTraining Neural ODE on spiral data...")
    for epoch in range(1, 301):
        # Start a new epoch in the tracker
        tracker.start_epoch()

        # Perform optimization step with SAM
        optimizer.first_step(loss_closure)
        loss = optimizer.second_step(loss_closure)

        # Log metrics
        tracker.log(loss=loss.data)

        # Print progress
        if epoch % 25 == 0:
            print(f"Epoch {epoch}/300: Loss = {loss.data:.6f}")

    # Generate predictions for visualization
    # First, prepare evenly spaced time points
    eval_t = np.linspace(0, 1, 100)

    # Start with the initial point (x[0])
    trajectory = [x[0]]

    # Use the trained model to generate the full trajectory
    current_point = Tensor(x[0:1], requires_grad=False)

    for i in range(99):
        # Predict the next point
        next_point = model(current_point)

        # Add to trajectory
        trajectory.append(next_point.data[0])

        # Update current point
        current_point = next_point

    trajectory = np.array(trajectory)

    # Plot the results
    plt.figure(figsize=(10, 8))

    # Plot the original data
    plt.subplot(2, 1, 1)
    plt.scatter(x[:, 0], x[:, 1], c=t, cmap="viridis", label="Data", s=30)
    plt.colorbar(label="Time")
    plt.title("Original Spiral Data")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()

    # Plot the learned trajectory
    plt.subplot(2, 1, 2)
    plt.scatter(
        trajectory[:, 0],
        trajectory[:, 1],
        c=eval_t,
        cmap="viridis",
        label="Neural ODE",
        s=30,
    )
    plt.colorbar(label="Time")
    plt.title("Neural ODE Learned Trajectory")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()

    plt.tight_layout()
    plt.savefig("neural_ode_spiral.png")
    plt.close()

    # Show training summary
    tracker.summary(style="box")

    # Export and plot training metrics
    tracker.export(format="json")
    tracker.plot(save_path="neural_ode_training_plot.png")

    print("\nNeural ODE training completed!")
    print("Check 'neural_ode_spiral.png' for the visualization.")


if __name__ == "__main__":
    run_node_example()
