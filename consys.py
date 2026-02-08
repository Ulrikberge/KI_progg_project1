# Control System Training Module (CONSYS)

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import config
from config import get_preset, get_training_config
from helpers import initialize_parameters, create_controller, create_plant


class ControlSystemTrainer:
    # Klasse for å håndtere trening av controllere ved hjelp av gradients
    # GRAD_CLIP er grensen for gradient clipping for neural networks
    GRAD_CLIP = 1.0 

    def __init__(self):
        self._preset = get_preset()
        self._training_cfg = get_training_config()
        self._mse_history = []
        self._param_history = []

    @property
    def controller_type(self):
        return self._preset["controller_type"]

    @property
    def plant_type(self):
        return self._preset["plant_type"]

    def _compute_loss(self, params):
        #Beregn MSE loss 
        controller = create_controller()
        plant, target = create_plant()
        n_steps = self._training_cfg["timesteps"]

        errors = []
        for _ in range(n_steps):
            error = target - plant.get_output()
            action = controller.calculate_control_signal(params, error)
            plant.timestep(action)
            errors.append(error)

        return jnp.mean(jnp.square(jnp.array(errors)))

    def _apply_gradients(self, params, grads):
        #Bruk gradient descent for å oppdatere parametrene
        # Ikke endre det under uten å tenke seg godt om
        lr = self._training_cfg["learning_rate"]

        if isinstance(params, list):
            # Neura nett: klipp gradients og oppdater hver layer
            clipped = [
                tuple(jnp.clip(g, -self.GRAD_CLIP, self.GRAD_CLIP) for g in layer)
                for layer in grads
            ]
            return [
                tuple(p - lr * g for p, g in zip(layer_p, layer_g))
                for layer_p, layer_g in zip(params, clipped)
            ]
        else:
            # PID: enkel gradient descent
            self._param_history.append(params)
            return params - lr * grads

    def _log_epoch(self, epoch_num, total_epochs, params):
        if not config.VERBOSE:
            return

        mse = self._mse_history[-1]
        print(f"Epoch {epoch_num}/{total_epochs}, MSE: {mse:.6f}")

        if self.controller_type == "pid":
            print(f"  Kp={params[0]:.4f}, Ki={params[1]:.4f}, Kd={params[2]:.4f}")

    def _print_summary(self):
        initial = self._mse_history[0]
        final = self._mse_history[-1]

        print("=" * 70)
        print("Training complete!")
        print(f"Final MSE: {final:.6f}")
        print(f"Initial MSE: {initial:.6f}")
        if initial > 0:
            improvement = (initial - final) / initial * 100
            print(f"Improvement: {improvement:.2f}%")
        print("=" * 70)

    def run(self):
        #Kjør treningen
        #Returner (final_params, mse_history, param_history)
        
        n_epochs = self._training_cfg["epochs"]
        lr = self._training_cfg["learning_rate"]

        # Initialize
        params = initialize_parameters()
        grad_fn = jax.value_and_grad(self._compute_loss)

        # Header
        print("=" * 70)
        print(f"Training {self.controller_type.upper()} on {self.plant_type.upper()}")
        print(f"Epochs: {n_epochs}, Timesteps: {self._training_cfg['timesteps']}, LR: {lr}")
        print(f"Preset: {config.ACTIVE_PRESET}")
        print("=" * 70)

        # Training loop
        for epoch in range(n_epochs):
            loss, grads = grad_fn(params)
            params = self._apply_gradients(params, grads)
            self._mse_history.append(loss.item())
            self._log_epoch(epoch + 1, n_epochs, params)

        self._print_summary()
        return params, self._mse_history, self._param_history


def train():
    # Tren controllerw ved å bruke gradient descent på MSE loss
    # Returner (final_params, mse_history, param_history)
   
    trainer = ControlSystemTrainer()
    return trainer.run()


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
