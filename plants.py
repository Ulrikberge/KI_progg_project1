"""Plant implementations using stateful classes.

Each plant has:
- get_output(): Returns the current output Y
- timestep(control_signal): Updates the plant state with control signal U
"""
import numpy as np
import jax.numpy as jnp


class Plant:
    """Base class for all plants."""

    def get_output(self):
        """Return the current plant output Y."""
        raise NotImplementedError

    def timestep(self, control_signal):
        """Update plant state with control signal U and disturbance D."""
        raise NotImplementedError


class BloodGlucosePlant(Plant):
    """Blood glucose regulation model (simplified glucose dynamics).

    Models how blood glucose changes based on:
    - Control signal U - directly adjusts glucose (e.g., insulin/glucagon balance)
    - Meal/activity disturbance - random glucose fluctuations
    - Natural clearance - body tries to return to basal glucose level

    Differential equation:
    dG/dt = sensitivity * U + disturbance - clearance * (G - G_basal)
    """

    def __init__(self, initial_glucose, basal_glucose, insulin_sensitivity,
                 glucose_clearance, dt, noise_range):
        self.glucose = initial_glucose  # Current blood glucose (mg/dL)
        self.basal_glucose = basal_glucose  # Equilibrium glucose level
        self.insulin_sensitivity = insulin_sensitivity  # How much insulin affects glucose
        self.glucose_clearance = glucose_clearance  # Natural uptake rate
        self.dt = dt
        self.noise_range = noise_range

    def get_output(self):
        return self.glucose

    def timestep(self, control_signal):
        # Disturbance represents meal/carbohydrate absorption
        meal_disturbance = np.random.uniform(self.noise_range[0], self.noise_range[1])

        # Control signal effect on glucose:
        # Positive error (glucose < target) → positive U → increase glucose
        # Negative error (glucose > target) → negative U → decrease glucose
        # So control_signal directly adjusts glucose in the right direction
        control_effect = self.insulin_sensitivity * control_signal

        # Natural clearance: body tries to return to basal glucose
        natural_clearance = self.glucose_clearance * (self.glucose - self.basal_glucose)

        # Glucose dynamics: dG/dt = control_effect + meal - clearance
        dG = control_effect + meal_disturbance - natural_clearance

        # Update glucose (cannot go below 0)
        self.glucose = jnp.maximum(0.0, self.glucose + dG * self.dt)


class BathtubPlant(Plant):
    """Bathtub model with drain."""

    def __init__(self, area, drain_coefficient, initial_height, gravity, dt, noise_range):
        self.area = area
        self.drain_area = drain_coefficient  
        self.water_level = initial_height
        self.noise_range = noise_range

    def get_output(self):
        return self.water_level

    def timestep(self, control_signal):
        
        velocity = jnp.sqrt(2.0 * 9.81 * self.water_level)

        # Disturbance
        disturbance = np.random.uniform(self.noise_range[0], self.noise_range[1])

        # Update water level: H = H + (U + D - Q) / A
        # where Q = drain_area * velocity
        self.water_level = jnp.maximum(0.0, self.water_level + (
            control_signal + disturbance - self.drain_area * velocity
        ) / self.area)


class CournotPlant(Plant):
    """Cournot competition duopoly model."""

    def __init__(self, max_price, marginal_cost, initial_q1, initial_q2, noise_range):
        self.max_price = max_price
        self.marginal_cost = marginal_cost
        self.q1 = initial_q1
        self.q2 = initial_q2
        self.noise_range = noise_range

    def price(self):
        return self.max_price - self.q1 - self.q2

    def get_output(self):
        # Output is firm 1's profit
        return self.q1 * (self.price() - self.marginal_cost)

    def timestep(self, control_signal):
        # Control signal is change in q1: U = ∂q1/∂t
        # Disturbance is change in q2: D = ∂q2/∂t
        disturbance = np.random.uniform(self.noise_range[0], self.noise_range[1])

        # Update quantities and enforce constraints as per task requirements
        # q1(t+1) = U + q1(t) as per task specification
        self.q1 = jnp.clip(self.q1 + control_signal, 0.0, 1.0)
        self.q2 = jnp.clip(self.q2 + disturbance, 0.0, 1.0)
