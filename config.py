"""Configuration file for the JAX-based control system.

EDIT THIS FILE to configure your experiment.
All pivotal parameters are set here in ONE PLACE.
"""

PLANT = "bathtub"  # "bathtub" | "cournot" | "norway_population"
CONTROLLER = "pid"  # "pid" | "neural"

VERBOSE = True  # Print progress during training

TRAINING_EPOCHS = 200
SIMULATION_TIMESTEPS = 100
LEARNING_RATE = 0.01  # Bathtub: small errors now, same as Cournot

# PID Controller
PID_INITIAL_PARAMS = [0.2, 0.05, 0.01]  # Initial [kp, ki, kd] values

# Neural Network Controller
NEURAL_NETWORK = {
    "neurons_per_hidden_layer": [16],  # Simpler single layer (bathtub is simpler than Cournot)
    "activation_functions": ["tanh"],  # Per layer: "sigmoid" | "tanh" | "relu"
    "output_activation_function": "linear",  # Output layer activation
    "weight_range": [-0.01, 0.01],  # Normal init (errors are small now!)
    "bias_range": [-0.01, 0.01]  # Normal init
}

# Norway Population Plant
NORWAY_POPULATION = {
    "initial_population": 5.5,  # Millions
    "target_population": 6.0,  # Millions
    "base_birth_rate": 0.01,  # ~1% per year
    "base_death_rate": 0.008,  # ~0.8% per year
    "carrying_capacity": 10.0,  # Millions
    "dt": 0.1,  # Timestep in years
    "noise_range": [-0.003, 0.003]  # Death rate disturbance
}

# Bathtub Plant
BATHTUB = {
    "cross_sectional_area": 1.0,  # m^2
    "drain_coefficient": 0.01,  # Drain area coefficient
    "initial_height": 5.0,  # m - Start at target (task: maintain H0)
    "target_height": 5.0,  # m - Goal: maintain this height
    "gravity": 9.8,  # m/s^2
    "dt": 1.0,  # seconds
    "noise_range": [-0.01, 0.01]  # Inflow/outflow disturbance
}

# Cournot Competition Plant
COURNOT = {
    "max_price": 4.0,        # Maximum price when q=0
    "marginal_cost": 0.1,    # Cost per unit (as suggested in task)
    "target_profit": 2,   # Challenging target (requires q1≈0.4-0.5)
    "initial_q1": 0.45,       # Start far below Nash equilibrium
    "initial_q2": 0.5,      # Competitor starts high
    "noise_range": [-0.01, 0.01]  # Task requirement for Cournot model
}

ENABLE_VISUALIZATION = True
PLOT_FREQUENCY = 1  # Plot every N epochs


RANDOM_SEED = 42
