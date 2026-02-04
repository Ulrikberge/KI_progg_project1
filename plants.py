"""Plant models for control system simulation."""
import numpy as np
import jax.numpy as jnp


class Plant:
    """Base class for all plant models."""

    def __init__(self, noise_range):
        self._noise_range = noise_range

    def _sample_noise(self):
        """Generate random disturbance within configured range."""
        return np.random.uniform(self._noise_range[0], self._noise_range[1])

    def get_output(self):
        raise NotImplementedError

    def timestep(self, control_signal):
        raise NotImplementedError


class BloodGlucosePlant(Plant):
    """Simplified blood glucose regulation dynamics.

    dG/dt = sensitivity * U + disturbance - clearance * (G - G_basal)
    """

    def __init__(self, initial_glucose, basal_glucose, insulin_sensitivity,
                 glucose_clearance, dt, noise_range):
        super().__init__(noise_range)
        self._glucose = initial_glucose
        self._basal = basal_glucose
        self._sensitivity = insulin_sensitivity
        self._clearance = glucose_clearance
        self._dt = dt

    def get_output(self):
        return self._glucose

    def timestep(self, control_signal):
        meal_effect = self._sample_noise()
        control_effect = self._sensitivity * control_signal
        clearance_effect = self._clearance * (self._glucose - self._basal)

        delta = (control_effect + meal_effect - clearance_effect) * self._dt
        self._glucose = jnp.maximum(0.0, self._glucose + delta)


class BathtubPlant(Plant):
    """Water tank with gravity-driven drain."""

    GRAVITY = 9.81

    def __init__(self, area, drain_coefficient, initial_height, gravity, dt, noise_range):
        super().__init__(noise_range)
        self._tank_area = area
        self._drain_area = drain_coefficient
        self._level = initial_height

    def _drain_velocity(self):
        """Torricelli's law: v = sqrt(2gh)"""
        return jnp.sqrt(2.0 * self.GRAVITY * self._level)

    def get_output(self):
        return self._level

    def timestep(self, control_signal):
        outflow = self._drain_area * self._drain_velocity()
        disturbance = self._sample_noise()

        net_flow = control_signal + disturbance - outflow
        self._level = jnp.maximum(0.0, self._level + net_flow / self._tank_area)


class CournotPlant(Plant):
    """Cournot duopoly competition model."""

    QUANTITY_BOUNDS = (0.0, 1.0)

    def __init__(self, max_price, marginal_cost, initial_q1, initial_q2, noise_range):
        super().__init__(noise_range)
        self._price_intercept = max_price
        self._cost = marginal_cost
        self._q1 = initial_q1
        self._q2 = initial_q2

    def _market_price(self):
        """Linear inverse demand: P = a - q1 - q2"""
        return self._price_intercept - self._q1 - self._q2

    def get_output(self):
        """Firm 1's profit: q1 * (P - c)"""
        return self._q1 * (self._market_price() - self._cost)

    def timestep(self, control_signal):
        competitor_move = self._sample_noise()

        lo, hi = self.QUANTITY_BOUNDS
        self._q1 = jnp.clip(self._q1 + control_signal, lo, hi)
        self._q2 = jnp.clip(self._q2 + competitor_move, lo, hi)
