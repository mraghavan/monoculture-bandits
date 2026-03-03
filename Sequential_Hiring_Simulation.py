import numpy as np
import multiprocessing
import time

def run_single_simulation(n, k=1000, seed=None):
    rng = np.random.default_rng(seed)

    v_true = rng.normal(0, 1, k)

    noise_sd = 0.5

    # 1. Polyculture
    noise_poly = rng.normal(0, noise_sd, (n, k))
    X_poly = v_true + noise_poly

    available_poly = np.ones(k, dtype=bool)
    chosen_poly = []
    for i in range(n):
        estimates = X_poly[i]
        masked_estimates = np.where(available_poly, estimates, -np.inf)
        best_idx = np.argmax(masked_estimates)
        chosen_poly.append(best_idx)
        available_poly[best_idx] = False
    perf_poly = np.mean(v_true[chosen_poly])

    # 2. Monoculture
    noise_mono_single = rng.normal(0, noise_sd, k)
    X_mono = v_true + noise_mono_single
    top_n_mono = np.argsort(X_mono)[-n:]
    perf_mono = np.mean(v_true[top_n_mono])

    # 3. Ensemble Monoculture
    noise_ensemble = np.mean(noise_poly, axis=0)
    X_ensemble = v_true + noise_ensemble
    top_n_ensemble = np.argsort(X_ensemble)[-n:]
    perf_ensemble = np.mean(v_true[top_n_ensemble])

    # Baseline: average true value of the objectively best n candidates
    perf_best = np.mean(np.sort(v_true)[-n:])

    # Return (best - actual) for each system; lower is better
    return perf_best - perf_poly, perf_best - perf_mono, perf_best - perf_ensemble

def run_simulation_set(n, num_runs=1000):
    seeds = [np.random.randint(0, 2**32 - 1) for _ in range(num_runs)]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(run_single_simulation, [(n, 1000, seed) for seed in seeds])

    results = np.array(results)
    mean_results = np.mean(results, axis=0)
    return mean_results

if __name__ == "__main__":
    ns = [100, 250, 500, 750, 900]
    num_runs = 1000

    print(f"Performance metric: (best - actual), lower is better")
    print(f"{'n':<5} | {'Polyculture':<15} | {'Monoculture':<15} | {'Ensemble':<15}")
    print("-" * 55)

    start_time = time.time()
    for n in ns:
        perf_poly, perf_mono, perf_ensemble = run_simulation_set(n, num_runs)
        print(f"{n:<5} | {perf_poly:<15.4f} | {perf_mono:<15.4f} | {perf_ensemble:<15.4f}")

    end_time = time.time()
    # print(f"\nTotal time: {end_time - start_time:.2f} seconds")
