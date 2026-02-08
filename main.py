# Kjør main
# For å gjøre konfigendringer: config.py

import numpy as np
import config as config
from consys import train, plot_results


def main():
    # Kjør main
    np.random.seed(config.RANDOM_SEED)

    # Tren controller
    params, mse_history, param_history = train()

    # Plot resultatene
    if config.ENABLE_PLOTS:
        plot_results(mse_history, param_history)


if __name__ == "__main__":
    main()
