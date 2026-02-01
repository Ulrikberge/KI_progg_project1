# JAX Control System - Simplified Version

## Quick Start

**1. Edit [`simple_config.py`](simple_config.py) - Change these two lines:**

```python
PLANT = "norway_population"  # "bathtub" | "cournot" | "norway_population"
CONTROLLER = "pid"  # "pid" | "neural"
```

**2. Run:**

```bash
python main.py
```

That's it!

---

## File Structure (MUCH SIMPLER!)

```
KI_progg_project1/
├── simple_config.py       # ← EDIT THIS to configure experiments
├── main.py         # Main entry point
├── consys.py              # System runner (train, run_epoch, run_timestep)
├── helpers.py             # Factory functions (get_plant, get_controller, get_params)
├── simple_plants.py       # Plant classes
└── simple_controllers.py  # Controller classes
```

---

## How It Works

### config.py
All parameters in ONE flat file. No complex nested structures.

```python
PLANT = "bathtub"
CONTROLLER = "neural"
TRAINING_EPOCHS = 100
SIMULATION_TIMESTEPS = 100
LEARNING_RATE = 0.01

BATHTUB = {
    "cross_sectional_area": 1.0,
    "target_height": 5.0,
    "noise_range": [-0.01, 0.01]
}

NEURAL_NETWORK = {
    "neurons_per_hidden_layer": [20, 16, 8],
    "activation_functions": ["tanh", "tanh", "relu"],
    ...
}
```

### simple_plants.py
Stateful plant classes:

```python
class NorwayPopulationPlant(Plant):
    def get_output(self):
        return self.population

    def timestep(self, control_signal):
        # Update self.population
        ...
```

### simple_controllers.py
Stateful controller classes:

```python
class PIDController(Controller):
    def __init__(self):
        self.error_history = []

    def calculate_control_signal(self, params, error):
        # PID formula
        return params.dot([error, derivative, integral])
```

### consys.py
Simple training functions:

```python
def run_one_timestep(params, plant, controller, target):
    output = plant.get_output()
    error = target - output
    control = controller.calculate_control_signal(params, error)
    plant.timestep(control)
    return error

def run_one_epoch(params):
    controller = get_controller()
    plant, target = get_plant()
    errors = [run_one_timestep(params, plant, controller, target)
              for _ in range(TIMESTEPS)]
    return mean_squared_error(errors)

def train():
    params = get_params()
    grad_func = jax.grad(run_one_epoch)

    for epoch in range(EPOCHS):
        mse, gradients = grad_func(params)
        params = params - LEARNING_RATE * gradients

    return params, mse_history
```

---

## Examples

### Run Norway Population with Classic PID

Edit `simple_config.py`:
```python
PLANT = "norway_population"
CONTROLLER = "pid"
```

Run:
```bash
python main.py
```

### Run Bathtub with Neural Network

Edit `config.py`:
```python
PLANT = "bathtub"
CONTROLLER = "neural"
```

Run:
```bash
python main.py
```

### Change Neural Network Architecture

Edit `config.py`:
```python
NEURAL_NETWORK = {
    "neurons_per_hidden_layer": [32, 16],  # 2 layers instead of 3
    "activation_functions": ["relu", "tanh"],  # Different activations
    ...
}
```

### Change Noise Level

Edit `config.py`:
```python
NORWAY_POPULATION = {
    ...
    "noise_range": [-0.005, 0.005]  # More noise!
}
```

---

## Key Differences from Original Code

| Aspect | Original | Simplified |
|--------|----------|------------|
| **Config** | Nested dataclasses in multiple files | Single flat dict in one file |
| **Plants** | Static functions with Protocol | Stateful classes with methods |
| **Controllers** | Static functions with Protocol | Stateful classes with methods |
| **State Management** | Explicit state passing | State stored in objects |
| **Complexity** | 15+ files, 2000+ lines | 6 files, ~500 lines |
| **Setup** | Complex imports and registries | Simple imports |

---

## Benefits of This Approach

1. **Simpler**: Much less code, easier to understand
2. **Cleaner config**: One file, flat structure, easy to edit
3. **More Pythonic**: Uses classes and methods naturally
4. **Easier to debug**: State is visible in objects
5. **Faster to modify**: Change config.py and run

---

## Running All Combinations

Want to test all 6 combinations (3 plants × 2 controllers)?

Create a script:

```python
import config
from consys import train, plot_results

plants = ["norway_population", "bathtub", "cournot"]
controllers = ["pid", "neural"]

for plant in plants:
    for controller in controllers:
        config.PLANT = plant
        config.CONTROLLER = controller
        config.ENABLE_VISUALIZATION = False

        print(f"\n{'='*70}")
        print(f"Running: {plant} + {controller}")
        print(f"{'='*70}")

        params, mse_history, param_history = train()
```

---

## Comparison with Original

### Original Way (Complex):
1. Edit `config/scenarios.py` (100+ lines)
2. Nested dataclasses: `Config.norway_population.neural_net.neurons_per_layer`
3. Protocol interfaces, static methods
4. Multiple registries and factory patterns
5. Run `python main.py`

### New Way (Simple):
1. Edit `config.py` (change 2 lines)
2. Flat dict: `NORWAY_POPULATION["target_population"]`
3. Simple classes with methods
4. One helper module
5. Run `python main.py`

---

## Which Version to Use?

**Use the SIMPLE version** (`main.py`) if you want:
- Quick experiments
- Easy configuration
- Clear code structure
- Less complexity

**Use the ORIGINAL version** (`main.py`) if you want:
- More formal software engineering patterns
- Type safety with Protocols
- Explicit state management
- Educational value for learning design patterns

Both implementations are functionally equivalent and satisfy all project requirements!
