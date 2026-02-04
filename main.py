"""Main entry point for the JAX-based control system.

TO CONFIGURE: Edit config.py
TO RUN: python main.py
"""
import numpy as np
import config as config
from consys import train, plot_results


def main():
    """Run training experiment."""
    # Set random seed for reproducibility
    np.random.seed(config.RANDOM_SEED)

    # Train controller
    params, mse_history, param_history = train()

    # Plot results
    if config.ENABLE_VISUALIZATION:
        plot_results(mse_history, param_history)


if __name__ == "__main__":
    main()
