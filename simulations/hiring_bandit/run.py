import numpy as np
import time
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool

def simulate_setting(setting, n, k, t, p_true, k_initial):
    """
    Simulates a single setting for a given set of arms and initial samples.
    
    setting: 'monoculture', 'poly_fixed', 'poly_random', 'ensemble'
    n: number of agents
    k: number of arms
    t: number of rounds
    p_true: true means of the k arms (k,)
    k_initial: initial successes for each agent and arm (n, k)
    """
    n_0 = 5
    
    # S[i] = successes for arm i in simulation rounds
    # N[i] = pulls for arm i in simulation rounds
    S = np.zeros(k)
    N = np.zeros(k)
    
    # Pre-calculate initial successes for settings
    if setting == 'monoculture':
        # All agents share the first agent's initial samples
        k_init_shared = k_initial[0]
        n_samples_init = n_0
    else:
        # For ensemble and polyculture, all initial samples are potentially available to an observer
        # (Though polyculture agents only use their own for picking)
        k_init_shared = np.sum(k_initial, axis=0)
        n_samples_init = n * n_0
    
    # For polyculture, determine the order
    # For fixed polyculture, shuffle once before all rounds
    # For random polyculture, shuffle each round
    if setting == 'poly_fixed':
        order = np.arange(n)
        np.random.shuffle(order)
    
    total_expected_reward = 0
    
    for r in range(t):
        # For random polyculture, shuffle order each round
        if setting == 'poly_random':
            order = np.random.permutation(n)
        
        pulled_this_round = []
        available = np.ones(k, dtype=bool)
        
        if setting in ['monoculture', 'ensemble']:
            # All agents have the same belief, but pick sequentially
            # Calculate shared scores for all arms
            scores = (2 + k_init_shared + S) / (4 + n_samples_init + N)
            # Add tie-breaker
            scores += np.random.uniform(0, 1e-10, size=k)
            
            # Each agent picks best available arm sequentially
            for _ in range(n):
                # Mask unavailable arms
                available_scores = scores.copy()
                available_scores[~available] = -1.0
                arm = np.argmax(available_scores)
                pulled_this_round.append(arm)
                available[arm] = False
                
        else:  # Polyculture: agents pick sequentially with different beliefs
            # Calculate scores for all agents
            # scores[j, i] = expected reward for agent j on arm i
            num = 2 + k_initial + S
            den = 4 + n_0 + N
            scores = num / den
            # Add tie-breaker for each agent-arm pair
            scores += np.random.uniform(0, 1e-10, size=(n, k))
            
            # Agents pick in the predetermined order
            for agent_idx in order:
                # Find best available arm for this agent
                agent_scores = scores[agent_idx].copy()
                agent_scores[~available] = -1.0
                arm = np.argmax(agent_scores)
                pulled_this_round.append(arm)
                available[arm] = False
        
        # Pull arms and observe rewards
        # Note: All agents observe all rewards
        for arm in pulled_this_round:
            reward = 1 if np.random.rand() < p_true[arm] else 0
            S[arm] += reward
            N[arm] += 1
            total_expected_reward += p_true[arm]

    # Calculate Statistics
    # 1. Total Bayesian Regret
    # Expected reward if we knew the best n arms
    top_n_indices = np.argsort(p_true)[-n:]
    max_expected_reward_per_round = np.sum(p_true[top_n_indices])
    total_regret = (t * max_expected_reward_per_round) - total_expected_reward
    
    # 2. Misclassified Arms
    # Impartial observer starts with Beta(2,2) and observes initial samples + simulation rewards
    obs_scores = (2 + k_init_shared + S) / (4 + n_samples_init + N)
    obs_scores += np.random.uniform(0, 1e-10, size=k)
    obs_top_n = np.argsort(obs_scores)[-n:]
    
    true_top_n_set = set(top_n_indices)
    obs_top_n_set = set(obs_top_n)
    misclassified = len(obs_top_n_set - true_top_n_set)
    
    return total_regret, misclassified

def run_single_experiment(params):
    n, k, t, seed = params
    np.random.seed(seed)
    # (1) Construct k arms. Mean p ~ Beta(2,2)
    p_true = np.random.beta(2, 2, size=k)
    
    # (2) Construct initial samples for n agents
    # For each arm and agent, n_0 = 5 samples from Bern(p_true)
    n_0 = 5
    # k_initial[j, i] is number of successes for agent j on arm i
    # This can be drawn from Binomial(n_0, p_true[i])
    k_initial = np.random.binomial(n_0, p_true, size=(n, k))
    
    results = {}
    for setting in ['monoculture', 'poly_fixed', 'poly_random', 'ensemble']:
        results[setting] = simulate_setting(setting, n, k, t, p_true, k_initial)
        
    return results

def run_simulations(n_list, k, t, n_runs):
    all_results = {}
    
    for n in n_list:
        print(f"Running n={n}...")
        start_time = time.time()
        
        # Parallelize runs
        with Pool() as pool:
            # Prepare params for each run with a unique seed for each
            # Using n and the run index to ensure unique seeds
            params = [(n, k, t, np.random.randint(0, 2**32 - 1) + i) for i in range(n_runs)]
            batch_results = pool.map(run_single_experiment, params)
        
        # Aggregate results
        agg = {
            'monoculture': {'regret': [], 'mis': []},
            'poly_fixed': {'regret': [], 'mis': []},
            'poly_random': {'regret': [], 'mis': []},
            'ensemble': {'regret': [], 'mis': []}
        }
        
        for res in batch_results:
            for setting in agg:
                agg[setting]['regret'].append(res[setting][0])
                agg[setting]['mis'].append(res[setting][1])
        
        # Calculate averages
        final_n_res = {}
        for setting in agg:
            final_n_res[setting] = {
                'avg_regret': np.mean(agg[setting]['regret']),
                'avg_mis': np.mean(agg[setting]['mis'])
            }
        
        all_results[n] = final_n_res
        end_time = time.time()
        print(f"n={n} took {end_time - start_time:.2f} seconds")
        
    return all_results

def generate_plots(n_values, results, plots_dir='plots'):
    os.makedirs(plots_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })

    conditions = ['monoculture', 'poly_fixed', 'poly_random', 'ensemble']
    labels = ['Monoculture', 'Poly Fixed', 'Poly Random', 'Ensemble Monoculture']
    line_styles_markers = [('-', 'o'), ('--', 's'), ('-.', '^'), ((0, (4, 1.5, 1, 1.5)), 'D')]
    cmap = plt.get_cmap('cubehelix')
    num_series = len(conditions)
    series_colors = [
        cmap(0.15 + (0.75 - 0.15) * i / (num_series - 1))
        for i in range(num_series)
    ]

    for metric, ylabel, filename, title in [
        ('avg_regret', 'Average Regret', 'hiring_bandit_regret.png', 'Hiring bandit: average regret'),
        ('avg_mis', 'Average Misclassified Arms', 'hiring_bandit_misclassified.png', 'Hiring bandit: average misclassified arms'),
    ]:
        fig, ax = plt.subplots(figsize=(12, 7))

        for condition, label, (linestyle, marker), color in zip(
            conditions, labels, line_styles_markers, series_colors
        ):
            values = [results[n][condition][metric] for n in n_values]
            ax.plot(n_values, values, marker=marker, linestyle=linestyle, label=label,
                    color=color, markerfacecolor=color, markeredgecolor='black',
                    markeredgewidth=1.6, markersize=7, linewidth=2.3)

        ax.set_xlabel('$n$ (number of agents)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=600)
        plt.close(fig)
        print(f"Generated plot: {os.path.join(plots_dir, filename)}")


if __name__ == "__main__":
    n_values = [10, 30, 50, 70, 90]
    k = 100
    t = 200
    n_runs = 1000

    # Set global seed for reproducibility of seeds passed to workers
    np.random.seed(42)

    # Actual run
    results = run_simulations(n_values, k, t, n_runs)

    # Print results in a nice table
    print("\nResults Table (Average over 1000 runs):")
    print(f"{'n':<4} | {'Setting':<15} | {'Avg Regret':<12} | {'Avg Misclassified':<18}")
    print("-" * 55)
    for n in n_values:
        for setting in ['monoculture', 'poly_fixed', 'poly_random', 'ensemble']:
            res = results[n][setting]
            print(f"{n:<4} | {setting:<15} | {res['avg_regret']:<12.2f} | {res['avg_mis']:<18.2f}")
        print("-" * 55)

    generate_plots(n_values, results)
