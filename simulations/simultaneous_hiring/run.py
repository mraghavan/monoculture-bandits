import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import os

def deferred_acceptance(candidate_prefs, firm_prefs_rank, firm_capacities):
    """
    Candidate-proposing deferred acceptance algorithm for many-to-one matching.
    
    candidate_prefs: (k, n) array, where each row is a list of firms in order of preference.
    firm_prefs_rank: (n, k) array, where firm_prefs_rank[f, c] is the rank of candidate c for firm f.
    firm_capacities: list of length n.
    """
    k, n = candidate_prefs.shape
    candidate_next_proposal = np.zeros(k, dtype=int)
    firm_matches = [[] for _ in range(n)]
    
    free_candidates = list(range(k))
    
    while free_candidates:
        c = free_candidates.pop(0)
        if candidate_next_proposal[c] >= n:
            continue
            
        f = candidate_prefs[c, candidate_next_proposal[c]]
        candidate_next_proposal[c] += 1
        
        firm_matches[f].append(c)
        firm_matches[f].sort(key=lambda cand: firm_prefs_rank[f, cand])
        
        if len(firm_matches[f]) > firm_capacities[f]:
            rejected = firm_matches[f].pop()
            free_candidates.append(rejected)
            
    matched_candidates = []
    for f_match in firm_matches:
        matched_candidates.extend(f_match)
    return matched_candidates

def run_single_sim(args):
    n_firms, seed, k_candidates, capacity = args
    rng = np.random.default_rng(seed)
    
    # 1. Construct candidates: true values drawn from N(0, 1)
    v = rng.normal(0, 1, k_candidates)
    
    # 2. Construct preferences: candidates rank firms randomly
    candidate_prefs = np.array([rng.permutation(n_firms) for _ in range(k_candidates)])
    
    # 3. Define hiring systems
    
    # Polyculture: noise ~ N(0, 0.5) drawn independently per firm per candidate
    noise_poly = rng.normal(0, 0.5, (n_firms, k_candidates))
    E_poly = v + noise_poly
    
    # Monoculture: noise ~ N(0, 0.5) drawn once per candidate, shared across all firms
    noise_mono_single = rng.normal(0, 0.5, k_candidates)
    E_mono = v + noise_mono_single
    
    # Ensemble Monoculture: noise is mean of Polyculture noises across firms
    noise_ensemble_single = np.mean(noise_poly, axis=0)
    E_ensemble = v + noise_ensemble_single
    
    firm_capacities = [capacity] * n_firms
    n_hired = n_firms * capacity

    # Baselines: average true value of the best and worst n_hired candidates
    sorted_v = np.sort(v)
    perf_best  = np.mean(sorted_v[-n_hired:])
    perf_worst = np.mean(sorted_v[:n_hired])
    
    # --- Polyculture Matching ---
    firm_prefs_rank_poly = np.argsort(np.argsort(-E_poly, axis=1), axis=1)
    matched_poly = deferred_acceptance(candidate_prefs, firm_prefs_rank_poly, firm_capacities)
    perf_poly = np.mean(v[matched_poly])
    
    # --- Monoculture Matching ---
    matched_mono_idx = np.argsort(-E_mono)[:n_hired]
    perf_mono = np.mean(v[matched_mono_idx])
    
    # --- Ensemble Monoculture Matching ---
    matched_ensemble_idx = np.argsort(-E_ensemble)[:n_hired]
    perf_ensemble = np.mean(v[matched_ensemble_idx])
    
    # Metric: fraction of available value captured = (actual - worst) / (best - worst)
    # Higher is better; 1 = perfect hiring, 0 = worst possible hiring
    denom = perf_best - perf_worst
    metric_poly     = (perf_poly     - perf_worst) / denom
    metric_mono     = (perf_mono     - perf_worst) / denom
    metric_ensemble = (perf_ensemble - perf_worst) / denom

    return metric_poly, metric_mono, metric_ensemble

def generate_line_plot(firm_counts, all_results, plots_dir='plots'):
    os.makedirs(plots_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })

    conditions = ['Monoculture', 'Polyculture', 'Ensemble Monoculture']
    line_styles_markers = [('-', 'o'), ('--', 's'), ('-.', '^')]
    cmap = plt.get_cmap('cubehelix')
    num_series = len(conditions)
    series_colors = [
        cmap(0.15 + (0.75 - 0.15) * i / (num_series - 1))
        for i in range(num_series)
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (condition, (linestyle, marker), color) in enumerate(
        zip(conditions, line_styles_markers, series_colors)
    ):
        values = [all_results[n][i] for n in firm_counts]
        ax.plot(firm_counts, values, marker=marker, linestyle=linestyle, label=condition,
                color=color, markerfacecolor=color, markeredgecolor='black',
                markeredgewidth=1.6, markersize=7, linewidth=2.3)

    ax.set_xlabel('$n$ (number of firms)')
    ax.set_ylabel('Fraction of available value captured')
    ax.set_title('Simultaneous hiring: monoculture vs. polyculture vs. ensemble')
    ax.set_ylim(0.93, 1.01)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'simultaneous_hiring_line.png'), dpi=600)
    plt.close(fig)
    print(f"Generated plot: {os.path.join(plots_dir, 'simultaneous_hiring_line.png')}")


def main():
    k_candidates = 1000
    capacity = 10
    num_runs = 1000
    firm_counts = [10, 30, 50, 70, 90]

    print(f"Starting simulation with k={k_candidates} candidates, {num_runs} runs per n_firms.")
    print(f"Performance metric: fraction of available value captured, higher is better")
    print(f"{'n':>2} | {'Polyculture':>12} | {'Monoculture':>12} | {'Ensemble':>12}")
    print("-" * 50)

    all_results = {}
    for n in firm_counts:
        seeds = range(n * 10000, n * 10000 + num_runs)
        args = [(n, seed, k_candidates, capacity) for seed in seeds]

        with multiprocessing.Pool() as pool:
            results = pool.map(run_single_sim, args)

        avg_poly, avg_mono, avg_ensemble = np.mean(results, axis=0)
        all_results[n] = (avg_mono, avg_poly, avg_ensemble)
        print(f"{n:>2} | {avg_poly:12.4f} | {avg_mono:12.4f} | {avg_ensemble:12.4f}")

    generate_line_plot(firm_counts, all_results)

if __name__ == "__main__":
    main()
