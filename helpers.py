"""Helper functions for creating plants, controllers, and parameters."""
import numpy as np
import config as config
from plants import NorwayPopulationPlant, BathtubPlant, CournotPlant
from controllers import PIDController, NeuralController


def get_params():
    """Initialize controller parameters based on config."""
    if config.CONTROLLER == "pid":
        # PID: 3 parameters [kp, ki, kd] from config
        params = np.array(config.PID_INITIAL_PARAMS)

    elif config.CONTROLLER == "neural":
        # Neural network: list of (weight, bias) tuples
        params = []
        hidden_layers = config.NEURAL_NETWORK["neurons_per_hidden_layer"]

        # Input size is always 3 (proportional, derivative, integral)
        sender = 3

        # Create weight and bias for each layer (including output layer)
        for receiver in hidden_layers + [1]:  # +[1] for output layer
            weights = np.random.uniform(
                config.NEURAL_NETWORK["weight_range"][0],
                config.NEURAL_NETWORK["weight_range"][1],
                (sender, receiver)
            )
            biases = np.random.uniform(
                config.NEURAL_NETWORK["bias_range"][0],
                config.NEURAL_NETWORK["bias_range"][1],
                receiver  # Shape (receiver,) not (1, receiver)
            )
            params.append((weights, biases))
            sender = receiver

    else:
        raise ValueError(f"Invalid controller type: {config.CONTROLLER}")

    return params


def get_controller():
    """Create controller object based on config."""
    if config.CONTROLLER == "pid":
        controller = PIDController()

    elif config.CONTROLLER == "neural":
        controller = NeuralController(
            activation_functions=config.NEURAL_NETWORK["activation_functions"],
            output_activation_function=config.NEURAL_NETWORK["output_activation_function"]
        )

    else:
        raise ValueError(f"Invalid controller type: {config.CONTROLLER}")

    return controller


def get_plant():
    """Create plant object and target value based on config."""
    if config.PLANT == "norway_population":
        plant = NorwayPopulationPlant(
            initial_population=config.NORWAY_POPULATION["initial_population"],
            base_birth_rate=config.NORWAY_POPULATION["base_birth_rate"],
            base_death_rate=config.NORWAY_POPULATION["base_death_rate"],
            carrying_capacity=config.NORWAY_POPULATION["carrying_capacity"],
            dt=config.NORWAY_POPULATION["dt"],
            noise_range=config.NORWAY_POPULATION["noise_range"]
        )
        target = config.NORWAY_POPULATION["target_population"]

    elif config.PLANT == "bathtub":
        plant = BathtubPlant(
            area=config.BATHTUB["cross_sectional_area"],
            drain_coefficient=config.BATHTUB["drain_coefficient"],
            initial_height=config.BATHTUB["initial_height"],
            gravity=config.BATHTUB["gravity"],
            dt=config.BATHTUB["dt"],
            noise_range=config.BATHTUB["noise_range"]
        )
        target = config.BATHTUB["target_height"]

    elif config.PLANT == "cournot":
        plant = CournotPlant(
            max_price=config.COURNOT["max_price"],
            marginal_cost=config.COURNOT["marginal_cost"],
            initial_q1=config.COURNOT["initial_q1"],
            initial_q2=config.COURNOT["initial_q2"],
            noise_range=config.COURNOT["noise_range"]
        )
        target = config.COURNOT["target_profit"]

    else:
        raise ValueError(f"Invalid plant type: {config.PLANT}")

    return plant, target
