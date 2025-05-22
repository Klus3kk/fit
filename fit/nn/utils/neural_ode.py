"""
Enhanced Neural Ordinary Differential Equation (ODE) implementation.

This module provides a flexible and numerically stable implementation of Neural ODEs,
allowing the representation of continuous-depth neural networks through differential
equations. This approach offers several advantages:

1. Memory efficiency - only the start and end states need to be stored
2. Continuous time dynamics - models can learn at arbitrary time precision
3. Reversibility - dynamics can run forward or backward in time
4. Well-established numerical tools - leverage ODE solvers with adaptive step sizing

Reference: Neural Ordinary Differential Equations (Chen et al., 2018)
https://arxiv.org/abs/1806.07366
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

from core.tensor import Tensor
from nn.layer import Layer


class ODEFunction:
    """
    Base class for defining the dynamics (vector field) of a Neural ODE.

    This class represents the right-hand side of the ODE: dx/dt = f(t, x).
    Concrete implementations should override the forward method.
    """

    def __init__(self):
        self.nfe = 0  # Number of function evaluations counter

    def forward(self, t: float, x: Tensor) -> Tensor:
        """
        Compute the time derivative at time t and state x.

        Args:
            t: Current time point
            x: Current state (Tensor)

        Returns:
            Time derivative of x (Tensor)
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def __call__(self, t: float, x: Tensor) -> Tensor:
        """Call the forward method and count function evaluations."""
        self.nfe += 1
        return self.forward(t, x)

    def reset_nfe(self):
        """Reset the function evaluation counter."""
        self.nfe = 0


class MLPODEFunction(ODEFunction):
    """
    Neural network implementation of ODEFunction using a multi-layer perceptron.

    This model parameterizes the vector field using a neural network that maps
    state to its derivative, independent of time (autonomous system).
    """

    def __init__(self, model: Layer):
        """
        Initialize the ODE function with a neural network model.

        Args:
            model: Neural network model that defines the dynamics
        """
        super().__init__()
        self.model = model

    def forward(self, t: float, x: Tensor) -> Tensor:
        """
        Compute the derivative using the neural network.

        Args:
            t: Current time point (unused in autonomous systems)
            x: Current state (Tensor)

        Returns:
            Time derivative of x (Tensor)
        """
        # Time argument is ignored for autonomous systems
        return self.model(x)


class TimeVaryingODEFunction(ODEFunction):
    """
    Time-varying ODE function where dynamics explicitly depend on time.

    This model parameterizes non-autonomous systems where the vector
    field depends on both state and time.
    """

    def __init__(self, model: Layer):
        """
        Initialize with a model that takes time and state.

        Args:
            model: Neural network expecting time and state as inputs
        """
        super().__init__()
        self.model = model

    def forward(self, t: float, x: Tensor) -> Tensor:
        """
        Compute the derivative with time-dependent dynamics.

        Args:
            t: Current time point
            x: Current state (Tensor)

        Returns:
            Time derivative of x (Tensor)
        """
        # Create time tensor with the same batch size as x
        batch_size = x.data.shape[0]
        t_tensor = Tensor(np.ones((batch_size, 1)) * t)

        # Concatenate time and state
        tx = Tensor(np.concatenate([t_tensor.data, x.data], axis=1))

        # Pass through model
        return self.model(tx)


class ODESolver:
    """
    Base class for ODE solvers.

    This provides a unified interface for different numerical integration
    methods to solve ODEs.
    """

    def __init__(self, func: ODEFunction):
        """
        Initialize the solver with an ODE function.

        Args:
            func: The ODEFunction defining the dynamics
        """
        self.func = func

    def integrate(
        self, x0: Tensor, t_span: Tuple[float, float], dt: float = 0.1
    ) -> Tensor:
        """
        Integrate the ODE from t_span[0] to t_span[1] with initial condition x0.

        Args:
            x0: Initial state
            t_span: Tuple of (t0, t1) for the integration time range
            dt: Time step size for fixed-step solvers

        Returns:
            Final state at time t_span[1]
        """
        raise NotImplementedError("Subclasses must implement integrate method")


class EulerSolver(ODESolver):
    """
    Euler method for numerically solving ODEs.

    This is the simplest numerical integration scheme, taking steps proportional
    to the derivative at the current point. While not the most accurate, it is
    fast and simple to implement.
    """

    def integrate(
        self, x0: Tensor, t_span: Tuple[float, float], dt: float = 0.1
    ) -> Tensor:
        """
        Integrate using Euler's method.

        Args:
            x0: Initial state
            t_span: Tuple of (t0, t1) for the integration time range
            dt: Time step size

        Returns:
            Final state at time t_span[1]
        """
        t0, t1 = t_span
        t = t0
        x = x0

        # Reset function evaluation counter
        self.func.reset_nfe()

        # Main integration loop
        while t < t1:
            # Ensure we don't go beyond t1
            h = min(dt, t1 - t)

            # Euler step: x_{n+1} = x_n + h * f(t_n, x_n)
            dx = self.func(t, x)
            x = x + h * dx

            # Update time
            t += h

        return x


class RK4Solver(ODESolver):
    """
    Fourth-order Runge-Kutta method for numerically solving ODEs.

    This method provides a good balance of accuracy and computational
    efficiency, making it a popular choice for many applications.
    """

    def integrate(
        self, x0: Tensor, t_span: Tuple[float, float], dt: float = 0.1
    ) -> Tensor:
        """
        Integrate using the fourth-order Runge-Kutta method.

        Args:
            x0: Initial state
            t_span: Tuple of (t0, t1) for the integration time range
            dt: Time step size

        Returns:
            Final state at time t_span[1]
        """
        t0, t1 = t_span
        t = t0
        x = x0

        # Reset function evaluation counter
        self.func.reset_nfe()

        # Main integration loop
        while t < t1:
            # Ensure we don't go beyond t1
            h = min(dt, t1 - t)

            # RK4 stages
            k1 = self.func(t, x)
            k2 = self.func(t + h / 2, x + h / 2 * k1)
            k3 = self.func(t + h / 2, x + h / 2 * k2)
            k4 = self.func(t + h, x + h * k3)

            # Update state: x_{n+1} = x_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Update time
            t += h

        return x


class DormandPrinceSolver(ODESolver):
    """
    Dormand-Prince (DOPRI) method with adaptive step sizing.

    This is an embedded Runge-Kutta method that provides error estimation
    for adaptive step size control, leading to more efficient and accurate
    integration.
    """

    def integrate(
        self,
        x0: Tensor,
        t_span: Tuple[float, float],
        dt: float = 0.1,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_steps: int = 1000,
    ) -> Tensor:
        """
        Integrate using the Dormand-Prince method with adaptive step sizing.

        Args:
            x0: Initial state
            t_span: Tuple of (t0, t1) for the integration time range
            dt: Initial time step size
            rtol: Relative tolerance for step size control
            atol: Absolute tolerance for step size control
            max_steps: Maximum number of steps to prevent infinite loops

        Returns:
            Final state at time t_span[1]
        """
        t0, t1 = t_span
        t = t0
        x = x0
        h = dt

        # Reset function evaluation counter
        self.func.reset_nfe()

        # Safety factor for step size adjustment
        safety = 0.9

        # Dormand-Prince method coefficients
        # Butcher tableau coefficients
        a = [
            [],  # a_1j (empty)
            [1 / 5],  # a_2j
            [3 / 40, 9 / 40],  # a_3j
            [44 / 45, -56 / 15, 32 / 9],  # a_4j
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],  # a_5j
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],  # a_6j
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],  # a_7j
        ]

        # 5th order coefficients
        b5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]

        # 4th order coefficients
        b4 = [
            5179 / 57600,
            0,
            7571 / 16695,
            393 / 640,
            -92097 / 339200,
            187 / 2100,
            1 / 40,
        ]

        # Time increments
        c = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]

        step_count = 0

        # Main integration loop
        while t < t1 and step_count < max_steps:
            step_count += 1

            # Ensure we don't go beyond t1
            if t + h > t1:
                h = t1 - t

            # Calculate stage derivatives
            k = []
            k.append(self.func(t, x))

            for i in range(1, 7):  # 7 stages
                # Compute intermediate state
                stage_x = x.copy()
                for j in range(i):
                    stage_x = stage_x + h * a[i][j] * k[j]

                # Compute stage derivative
                k.append(self.func(t + c[i] * h, stage_x))

            # Compute 5th order solution
            x5 = x.copy()
            for i in range(7):
                x5 = x5 + h * b5[i] * k[i]

            # Compute 4th order solution for error estimation
            x4 = x.copy()
            for i in range(7):
                x4 = x4 + h * b4[i] * k[i]

            # Estimate error
            error = np.max(np.abs(x5.data - x4.data))

            # Compute optimal step size
            tol = atol + rtol * np.max(np.abs(x.data))

            if error < tol:
                # Step accepted
                t += h
                x = x5  # Use higher order solution

                # Update step size for next iteration
                if error > 0:
                    h = safety * h * (tol / error) ** 0.2
                else:
                    h = h * 2  # Double step size if error is very small
            else:
                # Step rejected, reduce step size and retry
                h = safety * h * (tol / error) ** 0.25

                # Ensure minimum step size
                if h < 1e-10:
                    raise ValueError("Step size too small in adaptive solver.")

        if step_count >= max_steps:
            print(
                f"Warning: Maximum number of steps ({max_steps}) reached in adaptive solver."
            )

        return x


class NeuralODE(Layer):
    """
    Neural ODE layer that transforms inputs by solving an ODE.

    This layer integrates the initial state (input) from time t0 to t1
    using the specified ODE function and numerical solver.
    """

    def __init__(
        self,
        ode_func: ODEFunction,
        t0: float = 0.0,
        t1: float = 1.0,
        solver: str = "rk4",
        dt: float = 0.1,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        adjoint: bool = False,
    ):
        """
        Initialize a Neural ODE layer.

        Args:
            ode_func: Function defining the dynamics (ODEFunction instance)
            t0: Initial time (default: 0.0)
            t1: End time (default: 1.0)
            solver: ODE solver to use ("euler", "rk4", or "dopri", default: "rk4")
            dt: Time step size for fixed-step solvers (default: 0.1)
            rtol: Relative tolerance for adaptive solvers (default: 1e-3)
            atol: Absolute tolerance for adaptive solvers (default: 1e-6)
            adjoint: Whether to use adjoint method for backward pass (currently ignored)
        """
        super().__init__()
        self.ode_func = ode_func
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint

        # Create solver based on specified method
        if solver.lower() == "euler":
            self.solver = EulerSolver(ode_func)
        elif solver.lower() == "rk4":
            self.solver = RK4Solver(ode_func)
        elif solver.lower() == "dopri":
            self.solver = DormandPrinceSolver(ode_func)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Add ODE function model as a child if it has parameters
        if hasattr(self.ode_func, "model"):
            self.add_child(self.ode_func.model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform input by numerically solving the ODE.

        Args:
            x: Input tensor (initial state)

        Returns:
            Output tensor (final state)
        """
        # Set up time span
        t_span = (self.t0, self.t1)

        # Solve ODE
        if isinstance(self.solver, DormandPrinceSolver):
            # For adaptive solver, pass tolerances
            result = self.solver.integrate(
                x, t_span, dt=self.dt, rtol=self.rtol, atol=self.atol
            )
        else:
            # For fixed-step solvers
            result = self.solver.integrate(x, t_span, dt=self.dt)

        # Handle gradient if needed
        if x.requires_grad:
            result.requires_grad = True

            def _backward():
                # This is a simplified backward pass
                # A full implementation would use the adjoint method
                if x.requires_grad and result.grad is not None:
                    # Pass the gradient straight through for now
                    # This is a reasonable approximation for small time spans
                    x.grad = (
                        result.grad.copy() if x.grad is None else x.grad + result.grad
                    )

            result._backward = _backward
            result._prev = {x}

        return result

    def parameters(self):
        """Get all parameters of the layer, including those from the ODE function."""
        params = super().parameters()

        # Add parameters from ODE function if available
        if hasattr(self.ode_func, "model") and hasattr(
            self.ode_func.model, "parameters"
        ):
            params.extend(self.ode_func.model.parameters())

        return params


class ContinuousNormalization(Layer):
    """
    Continuous normalization layer using ODEs.

    This implements a normalizing flow using a Neural ODE, allowing
    complex transformations with a simpler implementation than traditional
    normalizing flows.
    """

    def __init__(
        self, dim: int, hidden_dims: List[int] = [64, 64], time_dependent: bool = False
    ):
        """
        Initialize a continuous normalization layer.

        Args:
            dim: Dimensionality of the input/output
            hidden_dims: List of hidden dimensions for the dynamics network
            time_dependent: Whether to use time-dependent dynamics
        """
        super().__init__()

        # Create the ODE function model
        from nn.linear import Linear
        from nn.activations import Tanh

        layers = []

        if time_dependent:
            # Input includes time dimension
            input_dim = dim + 1
        else:
            input_dim = dim

        # Build the dynamics network
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(Linear(prev_dim, h_dim))
            layers.append(Tanh())
            prev_dim = h_dim

        # Output layer with same dimension as input
        layers.append(Linear(prev_dim, dim))

        # Create Sequential model
        from nn.sequential import Sequential

        dynamics_network = Sequential(*layers)

        # Create ODE function
        if time_dependent:
            self.ode_func = TimeVaryingODEFunction(dynamics_network)
        else:
            self.ode_func = MLPODEFunction(dynamics_network)

        # Create Neural ODE layer
        self.node = NeuralODE(self.ode_func, t0=0.0, t1=1.0, solver="dopri", dt=0.1)

        # Add child layer
        self.add_child(self.node)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform input through the continuous normalization.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        return self.node(x)


# Backward compatibility alias
NODE = NeuralODE
