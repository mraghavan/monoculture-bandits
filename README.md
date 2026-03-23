# monoculture-bandits

Simulations comparing monoculture vs. polyculture strategies across several settings.

## Setup

1. **Install `uv`:**
    ```bash
    pip install uv
    ```

2. **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    uv pip install -r requirements.txt
    ```

## Simulations

### Failure Rates (`simulations/failure_rates/`)

Bayesian bandit simulation comparing failure rates of monoculture vs. polyculture agents across varying initial sample sizes.

```bash
cd simulations/failure_rates
uv run python run.py    # runs simulation, saves results to results/
uv run python plot.py   # generates plots to plots/
```

### Hiring Bandit (`simulations/hiring_bandit/`)

Bandit simulation in a hiring context. Compares monoculture, poly_fixed, poly_random, and ensemble conditions on regret and misclassified arms.

```bash
cd simulations/hiring_bandit
uv run python run.py
```

### Sequential Hiring (`simulations/sequential_hiring/`)

Sequential hiring simulation where agents pick candidates one at a time. Reports fraction of available value captured for polyculture, monoculture, and ensemble.

```bash
cd simulations/sequential_hiring
uv run python run.py
```

### Simultaneous Hiring (`simulations/simultaneous_hiring/`)

Hiring simulation using deferred acceptance matching. Reports fraction of available value captured for polyculture, monoculture, and ensemble.

```bash
cd simulations/simultaneous_hiring
uv run python run.py
```
