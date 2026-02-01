"""Control system (CONSYS) - the glue that runs training.

This module contains the core training loop:
- run_one_timestep: Execute one timestep
- run_one_epoch: Run K timesteps and compute MSE
- train: Run M epochs with gradient descent
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import config as config
from helpers import get_controller, get_params, get_plant


def run_one_timestep(params, plant, controller, target):
    """Run one timestep of the control loop.

    Args:
        params: Controller parameters
        plant: Plant object
        controller: Controller object
        target: Target value

    Returns:
        error: Error at this timestep
    """
    # Get current plant output
    output = plant.get_output()

    # Compute error
    error = target - output

    # Controller computes control signal
    control_signal = controller.calculate_control_signal(params, error)

    # Plant steps forward with control signal
    plant.timestep(control_signal)

    return error


def run_one_epoch(params):
    """Run one epoch (K timesteps) and compute MSE.

    Args:
        params: Controller parameters

    Returns:
        mse: Mean squared error over the epoch
    """
    # Create fresh plant and controller for this epoch
    controller = get_controller()
    plant, target = get_plant()

    # Run K timesteps
    errors = []
    for _ in range(config.SIMULATION_TIMESTEPS):
        error = run_one_timestep(params, plant, controller, target)
        errors.append(error)

    # Compute MSE
    mse = jnp.mean(jnp.square(jnp.array(errors)))
    return mse


def train():
    """Train controller parameters using gradient descent.

    Returns:
        params: Final trained parameters
        mse_history: List of MSE values per epoch
        param_history: List of parameters per epoch (for PID plotting)
    """
    # Initialize parameters
    params = get_params()

    # Track history
    mse_history = []
    param_history = [params] if config.CONTROLLER == "pid" else []

    # Create gradient function
    grad_func = jax.value_and_grad(run_one_epoch)

    # Training loop
    print("=" * 70)
    print(f"Training {config.CONTROLLER.upper()} controller on {config.PLANT.upper()} plant")
    print(f"Epochs: {config.TRAINING_EPOCHS}, Timesteps: {config.SIMULATION_TIMESTEPS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print("=" * 70)

    for epoch in range(config.TRAINING_EPOCHS):
        # Compute MSE and gradients
        mse, gradients = grad_func(params)

        # Clip gradients to prevent explosion (for neural networks)
        if isinstance(params, list):
            gradients = [
                tuple(jnp.clip(g, -1.0, 1.0) for g in grad_layer)
                for grad_layer in gradients
            ]

        # Update parameters using gradient descent
        if isinstance(params, list):
            # Neural network: params is list of (weight, bias) tuples
            params = [
                tuple(p - g * config.LEARNING_RATE for p, g in zip(param_layer, grad_layer))
                for param_layer, grad_layer in zip(params, gradients)
            ]
        else:
            # PID: params is array [kp, ki, kd]
            params = params - gradients * config.LEARNING_RATE
            param_history.append(params)

        # Record MSE
        mse_history.append(mse.item())

        # Print progress every epoch
        if config.VERBOSE:
            print(f"Epoch {epoch + 1}/{config.TRAINING_EPOCHS}, MSE: {mse_history[-1]}")
            if config.CONTROLLER == "pid":
                print(f"  Params: kp={params[0]:.4f}, ki={params[1]:.4f}, kd={params[2]:.4f}")

    print("=" * 70)
    print("Training complete!")
    print(f"Final MSE: {mse_history[-1]:.6f}")
    print(f"Initial MSE: {mse_history[0]:.6f}")
    if mse_history[0] > 0:
        improvement = ((mse_history[0] - mse_history[-1]) / mse_history[0] * 100)
        print(f"Improvement: {improvement:.2f}%")
    print("=" * 70)

    return params, mse_history, param_history


def plot_results(mse_history, param_history=None):
    """Plot training results.

    Args:
        mse_history: List of MSE values
        param_history: List of PID parameters (only for PID controller)
    """
    import os

    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Generate experiment name based on plant and controller
    experiment_name = f"{config.PLANT}_{config.CONTROLLER}"

    # Plot MSE history
    plt.figure(figsize=(10, 6))
    plt.plot(mse_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Training Progress: {config.PLANT.upper()} + {config.CONTROLLER.upper()}")
    plt.grid(True, alpha=0.3)

    # Save MSE plot
    mse_path = f"results/{experiment_name}_mse.png"
    plt.savefig(mse_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {mse_path}")
    plt.show()

    # Plot PID parameters if available
    if param_history and len(param_history) > 1:
        param_array = jnp.array(param_history)
        plt.figure(figsize=(10, 6))
        plt.plot(param_array[:, 0], label="kp (Proportional)", linewidth=2)
        plt.plot(param_array[:, 1], label="ki (Integral)", linewidth=2)
        plt.plot(param_array[:, 2], label="kd (Derivative)", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Parameter Value")
        plt.title("PID Parameter Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save PID parameters plot
        pid_path = f"results/{experiment_name}_pid_params.png"
        plt.savefig(pid_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {pid_path}")
        plt.show()
    else:
        print(f"DEBUG: Skipping PID plot - param_history empty or too short")
