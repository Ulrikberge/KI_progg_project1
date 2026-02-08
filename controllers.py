# Controller implementeringer for control systemet

import jax.numpy as jnp


# Activation function register
ACTIVATIONS = {
    "sigmoid": lambda x: 1.0 / (1.0 + jnp.exp(-x)),
    "tanh": jnp.tanh,
    "relu": lambda x: jnp.maximum(0.0, x),
    "linear": lambda x: x,
}


class Controller:
    # Base controller med feilsporing

    INTEGRAL_LIMIT = 50.0  # Anti-windup bound

    def __init__(self):
        self._errors = []

    def _compute_error_terms(self, e):
        # Beregn P, I, D fra current error
        self._errors.append(e)

        p_term = e
        d_term = e - self._errors[-2] if len(self._errors) > 1 else 0.0
        i_term = jnp.clip(sum(self._errors), -self.INTEGRAL_LIMIT, self.INTEGRAL_LIMIT)

        return jnp.array([p_term, d_term, i_term])

    def calculate_control_signal(self, params, error):
        raise NotImplementedError


class PIDController(Controller):
    # PID controller: U = Kpc*ce + Kic*cintegral(e) + Kd*de/dt

    def calculate_control_signal(self, params, error):
        pid_input = self._compute_error_terms(error)
        return jnp.dot(params, pid_input)


class NeuralController(Controller):
    # Feedforward neural network controller

    def __init__(self, activation_functions, output_activation_function):
        super().__init__()
        self._hidden_acts = activation_functions
        self._output_act = output_activation_function

    def _forward_pass(self, x, layer_params):
        """Single layer: linear transform + activation."""
        weights, biases = layer_params
        return x @ weights + biases

    def calculate_control_signal(self, params, error):
        x = self._compute_error_terms(error)

        act_names = self._hidden_acts + [self._output_act]

        for act_name, layer in zip(act_names, params):
            x = self._forward_pass(x, layer)
            x = ACTIVATIONS[act_name](x)

        return x.squeeze()
