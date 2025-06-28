"""
Simplified Neural ODE example that works with current FIT framework.

This example demonstrates the concept of Neural ODEs without requiring
specialized ODE solvers that aren't implemented yet.
"""

import matplotlib.pyplot as plt
import numpy as np

from fit.core.tensor import Tensor
from fit.nn.modules.activation import Tanh
from fit.nn.modules.linear import Linear
from fit.nn.modules.container import Sequential
from fit.loss.regression import MSELoss
from fit.optim.adam import Adam


def generate_spiral_data(n_points=100, noise=0.05):
    """
    Generate a simple 2D spiral dataset.

    Args:
        n_points: Number of data points
        noise: Amount of Gaussian noise to add

    Returns:
        Tuple of (t, trajectory) where t are time points and trajectory is the path
    """
    t = np.linspace(0, 2*np.pi, n_points)
    x0 = np.cos(t) * np.exp(-0.1*t) + noise * np.random.randn(n_points)
    x1 = np.sin(t) * np.exp(-0.1*t) + noise * np.random.randn(n_points)

    trajectory = np.stack([x0, x1], axis=1)
    return t, trajectory


class SimpleNeuralODE:
    """
    A simplified Neural ODE implementation using Euler's method.
    
    This demonstrates the core concept: learning the derivative function
    that governs the dynamics of a system.
    """
    
    def __init__(self, ode_func, dt=0.01):
        """
        Initialize the Neural ODE.
        
        Args:
            ode_func: Neural network that learns dx/dt = f(x, t)
            dt: Time step for numerical integration
        """
        self.ode_func = ode_func
        self.dt = dt
    
    def __call__(self, x0, n_steps):
        """
        Integrate the ODE forward in time.
        
        Args:
            x0: Initial condition (batch_size, state_dim)
            n_steps: Number of integration steps
            
        Returns:
            Trajectory of states
        """
        trajectory = [x0]
        x = x0
        
        for _ in range(n_steps):
            # Compute derivative: dx/dt = f(x)
            dx_dt = self.ode_func(x)
            
            # Euler step: x_{t+1} = x_t + dt * dx/dt
            x_next_data = x.data + self.dt * dx_dt.data
            x_next = Tensor(x_next_data, requires_grad=x.requires_grad)
            
            # Set up backward pass
            if x.requires_grad:
                def _backward(x_curr=x, dx_dt_curr=dx_dt, x_next_curr=x_next):
                    if x_next_curr.grad is not None:
                        # Gradient flows back through Euler step
                        if x_curr.grad is None:
                            x_curr.grad = x_next_curr.grad.copy()
                        else:
                            x_curr.grad += x_next_curr.grad
                        
                        # Gradient w.r.t. derivative
                        dx_dt_grad = self.dt * x_next_curr.grad
                        if dx_dt_curr.grad is None:
                            dx_dt_curr.grad = dx_dt_grad
                        else:
                            dx_dt_curr.grad += dx_dt_grad
                
                x_next._backward = _backward
                x_next._prev = {x, dx_dt}
            
            trajectory.append(x_next)
            x = x_next
        
        return trajectory


def run_neural_ode_example():
    """
    Demonstrate Neural ODE learning on spiral data.
    """
    print("Neural ODE Example: Learning Spiral Dynamics")
    print("=" * 50)
    
    # Generate spiral data
    t, true_trajectory = generate_spiral_data(n_points=50, noise=0.02)
    
    print(f"Generated spiral data with {len(true_trajectory)} points")
    
    # Create Neural ODE function
    # This network learns dx/dt = f(x)
    ode_func = Sequential(
        Linear(2, 16),
        Tanh(),
        Linear(16, 16), 
        Tanh(),
        Linear(16, 2)
    )
    
    # Create Neural ODE solver
    neural_ode = SimpleNeuralODE(ode_func, dt=0.05)
    
    # Prepare training data
    # Use consecutive points: x[i] -> x[i+1]
    X_data = true_trajectory[:-1]  # Current states
    y_data = true_trajectory[1:]   # Next states
    
    X_tensor = Tensor(X_data, requires_grad=True)
    y_tensor = Tensor(y_data, requires_grad=False)
    
    # Loss function and optimizer
    loss_fn = MSELoss()
    optimizer = Adam(ode_func.parameters(), lr=0.001)
    
    # Training loop
    epochs = 500
    losses = []
    
    print("\nTraining Neural ODE...")
    
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass: predict next states
        predicted_trajectory = []
        for i in range(len(X_data)):
            x_current = Tensor(X_data[i:i+1], requires_grad=True)
            trajectory = neural_ode(x_current, n_steps=1)
            predicted_trajectory.append(trajectory[1])  # Next state
        
        # Stack predictions
        predicted_next = Tensor(
            np.array([pred.data[0] for pred in predicted_trajectory]),
            requires_grad=True
        )
        
        # Compute loss
        loss = loss_fn(predicted_next, y_tensor)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        losses.append(loss.data)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data:.6f}")
    
    print("Training completed!")
    
    # Generate predictions for longer trajectory
    print("\nGenerating predictions...")
    
    # Start from initial condition
    x0 = Tensor(true_trajectory[0:1], requires_grad=False)
    predicted_trajectory = neural_ode(x0, n_steps=len(true_trajectory)-1)
    
    # Extract data for plotting
    predicted_data = np.array([state.data[0] for state in predicted_trajectory])
    
    # Plot results
    plot_results(true_trajectory, predicted_data, losses)
    
    return True


def plot_results(true_trajectory, predicted_trajectory, losses):
    """
    Plot the Neural ODE results.
    """
    try:
        plt.figure(figsize=(15, 5))
        
        # Plot trajectories
        plt.subplot(1, 3, 1)
        plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', label='True', linewidth=2)
        plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'r--', label='Predicted', linewidth=2)
        plt.scatter(true_trajectory[0, 0], true_trajectory[0, 1], c='green', s=100, label='Start')
        plt.title('Trajectory Comparison')
        plt.xlabel('X0')
        plt.ylabel('X1')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Plot loss curve
        plt.subplot(1, 3, 2)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        
        # Plot error over time
        plt.subplot(1, 3, 3)
        errors = np.linalg.norm(true_trajectory - predicted_trajectory, axis=1)
        plt.plot(errors)
        plt.title('Prediction Error')
        plt.xlabel('Time Step')
        plt.ylabel('L2 Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('neural_ode_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Results saved as 'neural_ode_results.png'")
        
        # Print final statistics
        final_error = np.mean(errors)
        print(f"\nFinal average prediction error: {final_error:.4f}")
        
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    print("Simplified Neural ODE Example")
    print("=" * 30)
    
    try:
        success = run_neural_ode_example()
        if success:
            print("\n🎉 Neural ODE example completed successfully!")
        else:
            print("\n❌ Neural ODE example failed")
    except Exception as e:
        print(f"\n❌ Error running Neural ODE example: {e}")
        import traceback
        traceback.print_exc()