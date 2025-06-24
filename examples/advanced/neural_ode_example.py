"""
Example usage of Neural ODEs with the FIT framework.

This example demonstrates how to use Neural ODEs for learning
continuous dynamics, illustrated with a simple spiral dataset.
"""

import matplotlib.pyplot as plt
import numpy as np

from fit.core.tensor import Tensor
from fit.monitor.tracker import TrainingTracker
from fit.nn.modules.activation import ReLU, Tanh
from fit.nn.modules.linear import Linear
from fit.nn.utils.neural_ode import MLPODEFunction, NeuralODE
from fit.nn.modules.container import Sequential
from fit.loss.regression import MSELoss
from fit.optim.adam import Adam
from fit.optim.experimental.sam import SAM


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

    print("Training Neural ODE on spiral data...")
    print(f"Data shape: {x.shape}")
    print(f"Time points: {len(t)}")

    # Training loop
    epochs = 200
    losses = []

    for epoch in range(epochs):
        # Zero gradients
        for param in model.parameters():
            param.grad = None

        # Forward pass
        predictions = model(x_tensor)
        loss = loss_fn(predictions, y_tensor)

        # Backward pass
        loss.backward()

        # SAM update
        optimizer.first_step(zero_grad=True)

        # Second forward pass for SAM
        predictions2 = model(x_tensor)
        loss2 = loss_fn(predictions2, y_tensor)
        loss2.backward()

        optimizer.second_step(zero_grad=True)

        # Track loss
        losses.append(loss.data)
        tracker.update(epoch, {"loss": loss.data})

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data:.6f}")

    print("Training completed!")

    # Generate predictions for visualization
    test_t = np.linspace(0, 1, 200)
    test_x = generate_spiral_data(n_points=200, noise=0.0)[1]

    predictions = []
    for i in range(len(test_x) - 1):
        x_test = Tensor([test_x[i]], requires_grad=False)
        pred = model(x_test)
        predictions.append(pred.data[0])

    predictions = np.array(predictions)

    # Plot results
    plot_node_results(t, x, test_t, test_x, predictions, losses)

    return True


def generate_lorenz_data(n_points=1000, dt=0.01):
    """
    Generate Lorenz attractor data.

    Args:
        n_points: Number of time points
        dt: Time step

    Returns:
        Tuple of (t, trajectory) where trajectory has shape (n_points, 3)
    """
    # Lorenz parameters
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    # Initialize
    t = np.arange(n_points) * dt
    trajectory = np.zeros((n_points, 3))
    trajectory[0] = [1.0, 1.0, 1.0]  # Initial condition

    # Generate trajectory using Euler method
    for i in range(1, n_points):
        x, y, z = trajectory[i - 1]

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        trajectory[i] = trajectory[i - 1] + dt * np.array([dx, dy, dz])

    return t, trajectory


def run_lorenz_example():
    """
    Example of using Neural ODEs to learn Lorenz attractor dynamics.
    """
    print("\nRunning Lorenz Attractor Neural ODE Example...")

    # Generate Lorenz data
    t, trajectory = generate_lorenz_data(n_points=500, dt=0.01)

    # Prepare training data
    x_tensor = Tensor(trajectory[:-1], requires_grad=True)
    y_tensor = Tensor(trajectory[1:], requires_grad=False)

    # Define ODE function for 3D Lorenz system
    ode_func_model = Sequential(
        Linear(3, 32), Tanh(), Linear(32, 32), Tanh(), Linear(32, 3)
    )

    ode_func = MLPODEFunction(ode_func_model)

    # Create Neural ODE
    node = NeuralODE(ode_func, t0=0.0, t1=0.01, solver="rk4", nsteps=5)  # dt = 0.01

    model = Sequential(node)

    # Training setup
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    print("Training Neural ODE on Lorenz attractor...")

    # Training loop
    epochs = 100
    losses = []

    for epoch in range(epochs):
        # Zero gradients
        for param in model.parameters():
            param.grad = None

        # Forward pass
        predictions = model(x_tensor)
        loss = loss_fn(predictions, y_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.data)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data:.6f}")

    # Generate longer trajectory for testing
    print("Generating predictions...")

    # Start with initial condition
    current_state = Tensor([trajectory[0]], requires_grad=False)
    predicted_trajectory = [current_state.data[0]]

    # Generate 200 steps
    for _ in range(200):
        next_state = model(current_state)
        predicted_trajectory.append(next_state.data[0])
        current_state = next_state

    predicted_trajectory = np.array(predicted_trajectory)

    # Plot Lorenz results
    plot_lorenz_results(trajectory[:201], predicted_trajectory, losses)

    return True


def plot_node_results(t, true_trajectory, test_t, test_trajectory, predictions, losses):
    """
    Plot Neural ODE results for spiral data.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot training loss
        axes[0, 0].plot(losses)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)

        # Plot true vs predicted trajectory
        axes[0, 1].plot(
            true_trajectory[:, 0], true_trajectory[:, 1], "b-", label="True"
        )
        axes[0, 1].plot(predictions[:, 0], predictions[:, 1], "r--", label="Predicted")
        axes[0, 1].set_title("Spiral Trajectory")
        axes[0, 1].set_xlabel("X")
        axes[0, 1].set_ylabel("Y")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot X component over time
        axes[1, 0].plot(t, true_trajectory[:, 0], "b-", label="True X")
        axes[1, 0].plot(test_t[1:], predictions[:, 0], "r--", label="Predicted X")
        axes[1, 0].set_title("X Component")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("X")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot Y component over time
        axes[1, 1].plot(t, true_trajectory[:, 1], "b-", label="True Y")
        axes[1, 1].plot(test_t[1:], predictions[:, 1], "r--", label="Predicted Y")
        axes[1, 1].set_title("Y Component")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Y")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("neural_ode_spiral.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Spiral results saved as 'neural_ode_spiral.png'")

    except Exception as e:
        print(f"Could not generate spiral plot: {e}")


def plot_lorenz_results(true_trajectory, predicted_trajectory, losses):
    """
    Plot Neural ODE results for Lorenz attractor.
    """
    try:
        fig = plt.figure(figsize=(15, 10))

        # 3D plot of true trajectory
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot(
            true_trajectory[:, 0],
            true_trajectory[:, 1],
            true_trajectory[:, 2],
            "b-",
            alpha=0.7,
        )
        ax1.set_title("True Lorenz Attractor")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # 3D plot of predicted trajectory
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax2.plot(
            predicted_trajectory[:, 0],
            predicted_trajectory[:, 1],
            predicted_trajectory[:, 2],
            "r-",
            alpha=0.7,
        )
        ax2.set_title("Predicted Lorenz Attractor")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        # Training loss
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(losses)
        ax3.set_title("Training Loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.grid(True)

        # Compare X components
        ax4 = fig.add_subplot(2, 2, 4)
        t_plot = np.arange(len(true_trajectory))
        ax4.plot(t_plot, true_trajectory[:, 0], "b-", label="True", alpha=0.7)
        ax4.plot(
            t_plot, predicted_trajectory[:, 0], "r--", label="Predicted", alpha=0.7
        )
        ax4.set_title("X Component Comparison")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("X")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig("neural_ode_lorenz.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Lorenz results saved as 'neural_ode_lorenz.png'")

    except Exception as e:
        print(f"Could not generate Lorenz plot: {e}")


def demonstrate_node_concepts():
    """
    Explain the key concepts behind Neural ODEs.
    """
    print("\nNeural ODE Concepts:")
    print("=" * 40)
    print("1. Continuous Dynamics: Model time as continuous rather than discrete")
    print("2. ODE Solver: Use numerical integration (RK4, Euler, etc.)")
    print("3. Adjoint Method: Efficient backpropagation through ODE solver")
    print("4. Memory Efficiency: Constant memory usage regardless of time steps")
    print("5. Flexible Time: Can evaluate at any time point")
    print("\nKey Parameters:")
    print("- solver: Integration method ('euler', 'rk4', 'dopri5')")
    print("- nsteps: Number of integration steps")
    print("- t0, t1: Start and end times for integration")
    print("- rtol, atol: Relative and absolute tolerance for adaptive solvers")


def compare_node_vs_rnn():
    """
    Simple comparison between Neural ODE and RNN on sequence modeling.
    """
    print("\nNeural ODE vs RNN Comparison:")
    print("=" * 40)

    # Generate simple sine wave data
    t = np.linspace(0, 4 * np.pi, 100)
    y = np.sin(t) + 0.1 * np.random.randn(100)

    print(f"Dataset: Sine wave with {len(t)} points")

    # Simple RNN-style approach (for comparison)
    print("\nRNN Approach:")
    print("- Fixed time steps")
    print("- Discrete hidden states")
    print("- Memory grows with sequence length")

    print("\nNeural ODE Approach:")
    print("- Continuous time modeling")
    print("- Can predict at any time point")
    print("- Constant memory usage")
    print("- Better for irregular time series")

    return True


if __name__ == "__main__":
    print("Neural ODE Examples with FIT Framework")
    print("=" * 50)

    # Explain concepts
    demonstrate_node_concepts()

    # Ask user what to run
    print("\nAvailable examples:")
    print("1. Spiral trajectory modeling")
    print("2. Lorenz attractor dynamics")
    print("3. Neural ODE vs RNN comparison")
    print("4. All examples")

    choice = input("Enter your choice (1-4): ")

    try:
        if choice == "1":
            run_node_example()

        elif choice == "2":
            run_lorenz_example()

        elif choice == "3":
            compare_node_vs_rnn()

        elif choice == "4":
            print("\nRunning all examples...")

            print("\n" + "=" * 30 + " SPIRAL EXAMPLE " + "=" * 30)
            run_node_example()

            print("\n" + "=" * 30 + " LORENZ EXAMPLE " + "=" * 30)
            run_lorenz_example()

            print("\n" + "=" * 30 + " COMPARISON " + "=" * 30)
            compare_node_vs_rnn()

        else:
            print("Invalid choice. Please run the script again.")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()

    print("\nNeural ODE examples completed!")
    print("\nKey takeaways:")
    print("- Neural ODEs model continuous dynamics")
    print("- Great for irregular time series and physical systems")
    print("- Memory efficient for long sequences")
    print("- Can extrapolate to unseen time points")
