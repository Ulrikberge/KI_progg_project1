# CONSYS - Control System with JAX

Gradient-based control system trainer using JAX autodiff.

## Quick Start

```bash
# Edit config.py to select preset
ACTIVE_PRESET = "bathtub_pid"

# Run
python main.py
```

## Presets

| Preset | Plant | Controller |
|--------|-------|------------|
| `bathtub_pid` | Water tank | PID |
| `bathtub_neural` | Water tank | Neural Net |
| `cournot_pid` | Duopoly market | PID |
| `cournot_neural` | Duopoly market | Neural Net |
| `blood_glucose_pid` | Glucose regulation | PID |
| `blood_glucose_neural` | Glucose regulation | Neural Net |

## Files

- `config.py` - Experiment presets
- `plants.py` - Plant models
- `controllers.py` - PID & Neural controllers
- `consys.py` - Training loop
- `main.py` - Entry point

## Requirements

JAX, NumPy, Matplotlib
