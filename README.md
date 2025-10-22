# monoculture-bandits

This project simulates a 2-armed Bayesian bandit problem to analyze the failure rate of a greedy (myopic) agent. The simulation compares the failure rate between two conditions: a single agent and multiple agents who learn independently.

## Setup and Usage

### Using `uv`

1.  **Install `uv`:**
    ```bash
    pip install uv
    ```

2.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **Run the simulation:**
    ```bash
    python main.py
    ```

    This will run the simulations and generate a plot of the failure rates named `failure_rates.png`.

6.  **Run tests:**
    ```bash
    python -m unittest discover tests
    ```
