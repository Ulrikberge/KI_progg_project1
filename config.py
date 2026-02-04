"""Experiment configuration with per-plant presets.


Simply change ACTIVE_PRESET to switch between experiments.
"""

ACTIVE_PRESET = "blood_glucose_pid"  # See PRESETS dict below for options

VERBOSE = True
ENABLE_PLOTS = True
RANDOM_SEED = 42

PRESETS = {
    "bathtub_pid": {
        "description": "Bathtub water level control with PID",
        "plant_type": "bathtub",
        "controller_type": "pid",
        "training": {
            "epochs": 200,
            "timesteps": 100,
            "learning_rate": 0.01,
        },
        "pid": {
            "initial_gains": [0.2, 0.05, 0.01],  # [Kp, Ki, Kd]
        },
        "plant": {
            "tank_area": 1.0,
            "outlet_area": 0.01,
            "initial_level": 5.0,
            "target_level": 5.0,
            "disturbance_range": [-0.01, 0.01],
        },
    },
    "bathtub_neural": {
        "description": "Bathtub water level control with neural network",
        "plant_type": "bathtub",
        "controller_type": "neural",
        "training": {
            "epochs": 200,
            "timesteps": 100,
            "learning_rate": 0.0001,
        },
        "neural": {
            "layer_sizes": [16],
            "hidden_activations": ["tanh"],
            "output_activation": "linear",
            "weight_init": [-0.01, 0.01],
            "bias_init": [-0.01, 0.01],
        },
        "plant": {
            "tank_area": 1.0,
            "outlet_area": 0.01,
            "initial_level": 5.0,
            "target_level": 5.0,
            "disturbance_range": [-0.01, 0.01],
        },
    },

    "cournot_pid": {
        "description": "Cournot competition profit maximization with PID",
        "plant_type": "cournot",
        "controller_type": "pid",
        "training": {
            "epochs": 100,
            "timesteps": 100,
            "learning_rate": 0.01,
        },
        "pid": {
            "initial_gains": [0.01, 0.01, 0.01],
        },
        "plant": {
            "price_intercept": 4.0,
            "unit_cost": 0.1,
            "firm1_quantity": 0.45,
            "firm2_quantity": 0.5,
            "target_profit": 2.0,
            "disturbance_range": [-0.01, 0.01],
        },
    },
    "cournot_neural": {
        "description": "Cournot competition profit maximization with neural network",
        "plant_type": "cournot",
        "controller_type": "neural",
        "training": {
            "epochs": 200,
            "timesteps": 100,
            "learning_rate": 0.00005,
        },
        "neural": {
            "layer_sizes": [8, 8],
            "hidden_activations": ["tanh", "tanh"],
            "output_activation": "linear",
            "weight_init": [-0.001, 0.001],
            "bias_init": [-0.001, 0.001],
        },
        "plant": {
            "price_intercept": 4.0,
            "unit_cost": 0.1,
            "firm1_quantity": 0.45,
            "firm2_quantity": 0.5,
            "target_profit": 2.0,
            "disturbance_range": [-0.01, 0.01],
        },
    },

    "blood_glucose_pid": {
        "description": "Blood glucose regulation with PID controller",
        "plant_type": "blood_glucose",
        "controller_type": "pid",
        "training": {
            "epochs": 200,
            "timesteps": 100,
            "learning_rate": 0.0005,
        },
        "pid": {
            "initial_gains": [1.0, 0.05, 0.05],
        },
        "plant": {
            "initial_glucose": 120.0,
            "target_glucose": 100.0,
            "basal_glucose": 90.0,
            "control_sensitivity": 1.0,
            "clearance_rate": 0.01,
            "time_step": 0.1,
            "disturbance_range": [-1.0, 1.0],
        },
    },
    "blood_glucose_neural": {
        "description": "Blood glucose regulation with neural network",
        "plant_type": "blood_glucose",
        "controller_type": "neural",
        "training": {
            "epochs": 300,
            "timesteps": 100,
            "learning_rate": 0.001,
        },
        "neural": {
            "layer_sizes": [16, 8],
            "hidden_activations": ["tanh", "tanh"],
            "output_activation": "linear",
            "weight_init": [-0.05, 0.05],
            "bias_init": [-0.05, 0.05],
        },
        "plant": {
            "initial_glucose": 120.0,
            "target_glucose": 100.0,
            "basal_glucose": 90.0,
            "control_sensitivity": 1.0,
            "clearance_rate": 0.01,
            "time_step": 0.1,
            "disturbance_range": [-1.0, 1.0],
        },
    },
}

def get_preset():
    """Return the currently active preset configuration."""
    if ACTIVE_PRESET not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{ACTIVE_PRESET}'. Available: {available}")
    return PRESETS[ACTIVE_PRESET]


def get_plant_config():
    """Return plant-specific configuration from active preset."""
    return get_preset()["plant"]


def get_training_config():
    """Return training hyperparameters from active preset."""
    return get_preset()["training"]


def get_controller_config():
    """Return controller configuration (pid or neural) from active preset."""
    preset = get_preset()
    ctrl_type = preset["controller_type"]
    return preset.get(ctrl_type, {})
