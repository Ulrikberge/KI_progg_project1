"""Control System Training Module (CONSYS).
"""
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import config
from config import get_preset, get_training_config
from helpers import initialize_parameters, create_controller, create_plant


def simulate_timestep(params, plant, controller, setpoint):
    """Execute one control loop iteration.

    Args:
        params: Controller parameters (array for PID, list for neural)
        plant: Plant instance
        controller: Controller instance
        setpoint: Target value to track

    Returns:
        Tracking error for this timestep
    """
    measurement = plant.get_output()
    error = setpoint - measurement
    control_action = controller.calculate_control_signal(params, error)
    plant.timestep(control_action)
    return error


def evaluate_epoch(params):
    """Run one complete simulation epoch and compute loss.

    Creates fresh plant/controller instances to ensure independent trials.

    Args:
        params: Controller parameters

    Returns:
        Mean squared error over all timesteps
    """
    controller = create_controller()
    plant, setpoint = create_plant()
    training_cfg = get_training_config()

    errors = []
    for _ in range(training_cfg["timesteps"]):
        err = simulate_timestep(params, plant, controller, setpoint)
        errors.append(err)

    return jnp.mean(jnp.square(jnp.array(errors)))


def train():
    """Train controller using gradient descent on MSE loss.

    Returns:
        Tuple of (final_params, mse_history, param_history)
    """
    preset = get_preset()
    training_cfg = get_training_config()

    # Extract settings
    n_epochs = training_cfg["epochs"]
    n_timesteps = training_cfg["timesteps"]
    lr = training_cfg["learning_rate"]
    ctrl_type = preset["controller_type"]
    plant_type = preset["plant_type"]


    params = initialize_parameters()
    mse_history = []
    param_history = [params] if ctrl_type == "pid" else []

    # JAX gradient function
    loss_and_grad = jax.value_and_grad(evaluate_epoch)

    # Print header
    print("=" * 70)
    print(f"Training {ctrl_type.upper()} on {plant_type.upper()}")
    print(f"Epochs: {n_epochs}, Timesteps: {n_timesteps}, LR: {lr}")
    print(f"Preset: {config.ACTIVE_PRESET}")
    print("=" * 70)

    # Training loop
    for epoch in range(n_epochs):
        loss, grads = loss_and_grad(params)

        # Gradient clipping for neural networks
        if isinstance(params, list):
            grads = [
                tuple(jnp.clip(g, -1.0, 1.0) for g in layer_grads)
                for layer_grads in grads
            ]
            # Update neural network params
            params = [
                tuple(p - lr * g for p, g in zip(layer_p, layer_g))
                for layer_p, layer_g in zip(params, grads)
            ]
        else:
            # Update PID params
            params = params - lr * grads
            param_history.append(params)

        mse_history.append(loss.item())

        if config.VERBOSE:
            print(f"Epoch {epoch + 1}/{n_epochs}, MSE: {mse_history[-1]:.6f}")
            if ctrl_type == "pid":
                print(f"  Kp={params[0]:.4f}, Ki={params[1]:.4f}, Kd={params[2]:.4f}")

    print("=" * 70)
    print("Training complete!")
    print(f"Final MSE: {mse_history[-1]:.6f}")
    print(f"Initial MSE: {mse_history[0]:.6f}")
    if mse_history[0] > 0:
        improvement = (mse_history[0] - mse_history[-1]) / mse_history[0] * 100
        print(f"Improvement: {improvement:.2f}%")
    print("=" * 70)

    return params, mse_history, param_history


def plot_results(mse_history, param_history=None):
    """Generate and save training visualization plots."""
    os.makedirs("results", exist_ok=True)

    preset = get_preset()
    experiment_name = config.ACTIVE_PRESET

    # MSE convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(mse_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Training Progress: {preset['description']}")
    plt.grid(True, alpha=0.3)

    mse_path = f"results/{experiment_name}_mse.png"
    plt.savefig(mse_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {mse_path}")
    plt.show()

    # PID parameter evolution plot
    if param_history and len(param_history) > 1:
        param_array = jnp.array(param_history)
        plt.figure(figsize=(10, 6))
        plt.plot(param_array[:, 0], label="Kp (Proportional)", linewidth=2)
        plt.plot(param_array[:, 1], label="Ki (Integral)", linewidth=2)
        plt.plot(param_array[:, 2], label="Kd (Derivative)", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Parameter Value")
        plt.title("PID Gain Evolution During Training")
        plt.legend()
        plt.grid(True, alpha=0.3)

        pid_path = f"results/{experiment_name}_pid.png"
        plt.savefig(pid_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {pid_path}")
        plt.show()
