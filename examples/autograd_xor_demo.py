import numpy as np
import matplotlib.pyplot as plt

from core.tensor import Tensor
import core.ops as ops


def plot_decision_boundary(
    model_params, x_min=-0.1, x_max=1.1, y_min=-0.1, y_max=1.1, mesh_size=0.01
):
    """Plot the decision boundary of a model."""
    W1, b1, W2, b2 = model_params

    # Create mesh grid
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_size), np.arange(y_min, y_max, mesh_size)
    )

    # Create input features
    X_mesh = np.c_[xx.ravel(), yy.ravel()]

    # Predict outputs
    Z = []
    for x in X_mesh:
        # Forward pass with the neural network
        x_tensor = Tensor([x])
        h1 = ops.matmul(x_tensor, W1) + b1
        h1_act = ops.tanh(h1)  # Use tanh activation for XOR
        logits = ops.matmul(h1_act, W2) + b2
        pred = ops.sigmoid(logits)
        Z.append(pred.data[0, 0])

    # Reshape predictions to match mesh grid
    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors="k")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("XOR Decision Boundary")


# Define XOR problem data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Convert to tensors
X_tensor = Tensor(X)
y_tensor = Tensor(y)

# Define network parameters
np.random.seed(42)  # For reproducibility

# Use a larger hidden layer with proper initialization for XOR
hidden_size = 16

# Initialize weights with a pattern that helps solve XOR
# This initialization pattern breaks symmetry which is crucial for XOR
W1 = Tensor(np.zeros((2, hidden_size)), requires_grad=True)
b1 = Tensor(np.zeros(hidden_size), requires_grad=True)
W2 = Tensor(np.random.randn(hidden_size, 1) * 0.1, requires_grad=True)
b2 = Tensor(np.zeros(1), requires_grad=True)

# Use asymmetric initialization to break symmetry for XOR
init_scale = 1.0
for i in range(hidden_size):
    if i % 2 == 0:
        W1.data[0, i] = init_scale
        W1.data[1, i] = -init_scale
    else:
        W1.data[0, i] = -init_scale
        W1.data[1, i] = init_scale

    # Alternating bias
    if i < hidden_size // 2:
        b1.data[i] = 0.1
    else:
        b1.data[i] = -0.1

# Training parameters
learning_rate = 0.05
num_epochs = 1000
losses = []

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    h1 = ops.matmul(X_tensor, W1) + b1
    h1_act = ops.tanh(h1)  # Using tanh activation for XOR
    logits = ops.matmul(h1_act, W2) + b2
    outputs = ops.sigmoid(logits)

    # Compute mean squared error loss
    diff = outputs - y_tensor
    # Calculate MSE manually
    squared_diff = diff * diff
    total_loss = np.sum(squared_diff.data)  # Use numpy directly
    loss_value = total_loss / squared_diff.data.size
    
    # Create a scalar tensor for loss
    loss = Tensor(loss_value, requires_grad=True)
    losses.append(loss_value)

    # Manual backward computation - bypass autograd
    # For MSE loss: derivative is 2*(pred-target)/n
    output_grad = 2.0 * diff.data / diff.data.size
    
    # Gradient through sigmoid: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    sigmoid_grad = outputs.data * (1 - outputs.data) * output_grad
    
    # Gradient through W2: h1_act.T @ sigmoid_grad
    W2_grad = h1_act.data.T @ sigmoid_grad
    # Gradient through b2: sum of sigmoid_grad along batch dimension
    b2_grad = np.sum(sigmoid_grad, axis=0)
    
    # Gradient through h1_act: sigmoid_grad @ W2.T
    h1_act_grad = sigmoid_grad @ W2.data.T
    
    # Gradient through tanh: tanh'(x) = 1 - tanh(x)^2
    tanh_grad = (1 - h1_act.data**2) * h1_act_grad
    
    # Gradient through W1: X.T @ tanh_grad
    W1_grad = X_tensor.data.T @ tanh_grad
    # Gradient through b1: sum of tanh_grad along batch dimension
    b1_grad = np.sum(tanh_grad, axis=0)
    
    # Ensure gradients exist (assign manually instead of relying on autograd)
    W1.grad = W1_grad
    b1.grad = b1_grad
    W2.grad = W2_grad
    b2.grad = b2_grad

    # Update parameters using gradient descent
    W1.data -= learning_rate * W1.grad
    b1.data -= learning_rate * b1.grad
    W2.data -= learning_rate * W2.grad
    b2.data -= learning_rate * b2.grad

    # Zero gradients after update (not strictly necessary but good practice)
    W1.grad = None
    b1.grad = None
    W2.grad = None
    b2.grad = None

    # Print progress
    if (epoch + 1) % 100 == 0:
        # Calculate accuracy
        predictions = (outputs.data > 0.5).astype(int)
        accuracy = np.mean(predictions == y_tensor.data) * 100
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_value:.4f}, Accuracy: {accuracy:.1f}%"
        )

# Make predictions
h1 = ops.matmul(X_tensor, W1) + b1
h1_act = ops.tanh(h1)
logits = ops.matmul(h1_act, W2) + b2
predictions = ops.sigmoid(logits)

# Convert predictions to binary
binary_predictions = (predictions.data > 0.5).astype(int)

# Print results
print("\nPredictions vs. Actual:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {binary_predictions[i][0]}, Actual: {y[i][0]}")

# Calculate accuracy
accuracy = np.mean(binary_predictions == y_tensor.data) * 100
print(f"Final accuracy: {accuracy:.1f}%")

# Plot loss curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

# Plot decision boundary
plt.subplot(1, 2, 2)
plot_decision_boundary([W1, b1, W2, b2])
plt.tight_layout()
plt.savefig("xor_results.png")
plt.show()

print("Training complete! Decision boundary and loss curve saved to 'xor_results.png'")