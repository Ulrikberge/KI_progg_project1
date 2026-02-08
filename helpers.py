# Factory module for å instansiere kompontenter fra config og lignende

import numpy as np
from config import get_preset, get_plant_config, get_controller_config
from plants import BathtubPlant, CournotPlant, BloodGlucosePlant
from controllers import PIDController, NeuralController


def initialize_parameters():
    # Opprett initial controller parametre basert på aktiv preset
    # Returner for PID: numpy array [Kp, Ki, Kd]
    # Returner for Neural: list of (weights, biases) tuples per layer

    preset = get_preset()
    ctrl_type = preset["controller_type"]
    ctrl_config = get_controller_config()

    if ctrl_type == "pid":
        return np.array(ctrl_config["initial_gains"], dtype=np.float32)

    elif ctrl_type == "neural":
        layers = ctrl_config["layer_sizes"]
        w_range = ctrl_config["weight_init"]
        b_range = ctrl_config["bias_init"]

        # Opprett layer-by-layer: input(3) -> hidden layers -> output(1)
        network_params = []
        input_dim = 3  # [proportional, derivative, integral]

        for output_dim in layers + [1]:
            W = np.random.uniform(w_range[0], w_range[1], (input_dim, output_dim))
            b = np.random.uniform(b_range[0], b_range[1], output_dim)
            network_params.append((W, b))
            input_dim = output_dim

        return network_params

    raise ValueError(f"Unknown controller type: {ctrl_type}")


def create_controller():
    # Opprett controller objekt basert på aktiv preset
    # Returner PID eller Neural Network
    
    preset = get_preset()
    ctrl_type = preset["controller_type"]
    ctrl_config = get_controller_config()

    if ctrl_type == "pid":
        return PIDController()

    elif ctrl_type == "neural":
        return NeuralController(
            activation_functions=ctrl_config["hidden_activations"],
            output_activation_function=ctrl_config["output_activation"]
        )

    raise ValueError(f"Unknown controller type: {ctrl_type}")


def create_plant():
    # Opprett plante objekt basert på aktiv preset
    # Returner plante og target_value

    preset = get_preset()
    plant_type = preset["plant_type"]
    cfg = get_plant_config()

    if plant_type == "bathtub":
        plant = BathtubPlant(
            area=cfg["tank_area"],
            drain_coefficient=cfg["outlet_area"],
            initial_height=cfg["initial_level"],
            gravity=cfg["bathtub_gravity"],
            noise_range=cfg["disturbance_range"]
        )
        return plant, cfg["target_level"]

    elif plant_type == "cournot":
        plant = CournotPlant(
            max_price=cfg["price_intercept"],
            marginal_cost=cfg["unit_cost"],
            initial_q1=cfg["firm1_quantity"],
            initial_q2=cfg["firm2_quantity"],
            noise_range=cfg["disturbance_range"]
        )
        return plant, cfg["target_profit"]

    elif plant_type == "blood_glucose":
        plant = BloodGlucosePlant(
            initial_glucose=cfg["initial_glucose"],
            basal_glucose=cfg["basal_glucose"],
            insulin_sensitivity=cfg["control_sensitivity"],
            glucose_clearance=cfg["clearance_rate"],
            dt=cfg["time_step"],
            noise_range=cfg["disturbance_range"]
        )
        return plant, cfg["target_glucose"]

    raise ValueError(f"Unknown plant type: {plant_type}")
