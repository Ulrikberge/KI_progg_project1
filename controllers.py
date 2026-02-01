"""Controller implementations using stateful classes.

Each controller has:
- calculate_control_signal(params, error): Computes control signal U from error E
- error_history: Tracks errors for integral and derivative terms
"""
import jax.numpy as jnp


class Controller:
    """Base class for all controllers."""

    def __init__(self):
        self.error_history = []

    def calculate_control_signal(self, params, error):
        """Calculate control signal U from error E and parameters."""
        raise NotImplementedError


class PIDController(Controller):
    """Classic 3-parameter PID controller.

    params: jnp.array([kp, ki, kd])
    """

    def calculate_control_signal(self, params, error):
        self.error_history.append(error)

        # Proportional term
        proportional = error

        # Derivative term
        if len(self.error_history) > 1:
            derivative = error - self.error_history[-2]
        else:
            derivative = 0.0

        # Integral term (with anti-windup clipping)
        integral = jnp.clip(sum(self.error_history), -50.0, 50.0)

        # PID formula: U = kp*e + ki*integral + kd*derivative
        input_vector = jnp.array([proportional, derivative, integral])
        return params.dot(input_vector)


class NeuralController(Controller):
    """Neural network-based controller.

    params: List of (weight, bias) tuples for each layer
    """

    def __init__(self, activation_functions, output_activation_function):
        super().__init__()
        self.activation_functions = activation_functions
        self.output_activation_function = output_activation_function

    def sigmoid(self, x):
        x = jnp.clip(x, -20, 20)  # Prevent overflow
        return 1.0 / (1.0 + jnp.exp(-x))

    def tanh(self, x):
        return jnp.tanh(x)

    def relu(self, x):
        return jnp.maximum(0.0, x)

    def linear(self, x):
        return x

    def apply_activation(self, x, activation_name):
        """Apply activation function by name."""
        if activation_name == "sigmoid":
            return self.sigmoid(x)
        elif activation_name == "tanh":
            return self.tanh(x)
        elif activation_name == "relu":
            return self.relu(x)
        elif activation_name == "linear":
            return self.linear(x)
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")

    def calculate_control_signal(self, params, error):
        self.error_history.append(error)

        # Proportional term
        proportional = error

        # Derivative term
        if len(self.error_history) > 1:
            derivative = error - self.error_history[-2]
        else:
            derivative = 0.0

        # Integral term (with anti-windup clipping to prevent explosion)
        integral = jnp.clip(sum(self.error_history), -50.0, 50.0)

        # Input vector: [proportional, derivative, integral]
        activation = jnp.array([proportional, derivative, integral])

        # Forward pass through hidden layers
        all_activations = self.activation_functions + [self.output_activation_function]

        for activation_fn_name, (weight, bias) in zip(all_activations, params):
            # Linear transformation
            activation = activation.dot(weight) + bias

            # Apply activation function
            activation = self.apply_activation(activation, activation_fn_name)

        # Squeeze to scalar output
        activation = activation.squeeze()
        return activation
