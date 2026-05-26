import numpy as np
import multiprocessing

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

    # Candidates who are not currently matched and have not exhausted their preferences
    free_candidates = list(range(k))

    while free_candidates:
        c = free_candidates.pop(0)
        # If candidate has proposed to all firms, they stay unmatched
        if candidate_next_proposal[c] >= n:
            continue

        # Candidate proposes to their next most preferred firm
        f = candidate_prefs[c, candidate_next_proposal[c]]
        candidate_next_proposal[c] += 1

        # Firm f receives the proposal
        firm_matches[f].append(c)
        # Sort firm matches by firm's preference (lower rank is better)
        firm_matches[f].sort(key=lambda cand: firm_prefs_rank[f, cand])

        # If firm exceeds capacity, it rejects the least preferred candidate
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

    # Polyculture: noise epsilon_ij ~ N(0, 0.25) -> std dev = 0.5
    noise_poly = rng.normal(0, 0.5, (n_firms, k_candidates))
    E_poly = v + noise_poly

    # Monoculture: noise epsilon_i ~ N(0, 0.25) same for all firms
    noise_mono_single = rng.normal(0, 0.5, k_candidates)
    E_mono = v + noise_mono_single

    # Ensemble Monoculture: noise is mean of Polyculture noises
    noise_ensemble_single = np.mean(noise_poly, axis=0)
    E_ensemble = v + noise_ensemble_single

    firm_capacities = [capacity] * n_firms

    # --- Polyculture Matching ---
    # Calculate ranks for firm preferences: lower rank is better (higher E)
    firm_prefs_rank_poly = np.argsort(np.argsort(-E_poly, axis=1), axis=1)
    matched_poly = deferred_acceptance(candidate_prefs, firm_prefs_rank_poly, firm_capacities)
    perf_poly = np.mean(v[matched_poly])

    # --- Monoculture Matching ---
    # Since all firms have identical preferences, the matched candidates
    # will be the top (n_firms * capacity) candidates according to E_mono.
    matched_mono_idx = np.argsort(-E_mono)[:n_firms * capacity]
    perf_mono = np.mean(v[matched_mono_idx])

    # --- Ensemble Monoculture Matching ---
    # Similarly, matched candidates are the top (n_firms * capacity) according to E_ensemble.
    matched_ensemble_idx = np.argsort(-E_ensemble)[:n_firms * capacity]
    perf_ensemble = np.mean(v[matched_ensemble_idx])

    return perf_poly, perf_mono, perf_ensemble

def main():
    k_candidates = 1000
    capacity = 10
    num_runs = 1000
    firm_counts = [10, 25, 50, 75, 90]

    print(f"Starting simulation with k={k_candidates} candidates, {num_runs} runs per n_firms.")
    print(f"{'n':>2} | {'Polyculture':>12} | {'Monoculture':>12} | {'Ensemble':>12}")
    print("-" * 50)

    for n in firm_counts:
        # Use a distinct range of seeds for each n to ensure reproducibility and independence
        seeds = range(n * 10000, n * 10000 + num_runs)
        args = [(n, seed, k_candidates, capacity) for seed in seeds]

        with multiprocessing.Pool() as pool:
            results = pool.map(run_single_sim, args)

        avg_poly, avg_mono, avg_ensemble = np.mean(results, axis=0)
        print(f"{n:>2} | {avg_poly:12.4f} | {avg_mono:12.4f} | {avg_ensemble:12.4f}")

if __name__ == "__main__":
    main()
