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


class NorwayPopulationPlant(Plant):
    """Norway population model with logistic growth."""

    def __init__(self, initial_population, base_birth_rate, base_death_rate,
                 carrying_capacity, dt, noise_range):
        self.population = initial_population
        self.base_birth_rate = base_birth_rate
        self.base_death_rate = base_death_rate
        self.carrying_capacity = carrying_capacity
        self.dt = dt
        self.noise_range = noise_range

    def get_output(self):
        return self.population

    def timestep(self, control_signal):
        # Disturbance affects death rate
        disturbance = np.random.uniform(self.noise_range[0], self.noise_range[1])

        # Birth and death rates (control adjusts birth rate)
        birth_rate = self.base_birth_rate + control_signal
        death_rate = self.base_death_rate + disturbance

        # Net growth rate
        r = birth_rate - death_rate

        # Logistic growth: dP/dt = r * P * (1 - P/K)
        growth = r * self.population * (1.0 - self.population / self.carrying_capacity)

        # Update population
        self.population = jnp.maximum(0.0, self.population + growth * self.dt)


class BathtubPlant(Plant):
    """Bathtub model with drain."""

    def __init__(self, area, drain_coefficient, initial_height, gravity, dt, noise_range):
        self.area = area
        self.drain_area = drain_coefficient  # This is actually drain area (C)
        self.water_level = initial_height
        self.noise_range = noise_range

    def get_output(self):
        return self.water_level

    def timestep(self, control_signal):
        # Drain velocity: V = sqrt(2*g*H)
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
