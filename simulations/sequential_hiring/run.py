import numpy as np
import multiprocessing
import time

def run_single_simulation(n, k=1000, seed=None):
    # Ensure independent random numbers for each worker
    rng = np.random.default_rng(seed)
    
    # Construct candidates
    v_true = rng.normal(0, 1, k)
    
    # Noise standard deviation
    noise_sd = 0.5
    
    # 1. Polyculture
    # Each agent's noise value for a given candidate is drawn independently from N(0, 0.5)
    noise_poly = rng.normal(0, noise_sd, (n, k))
    X_poly = v_true + noise_poly
    
    # Sequential choice for Polyculture
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
    # Each agent's noise value for a given candidate is drawn from N(0, 0.5), shared across agents
    noise_mono_single = rng.normal(0, noise_sd, k)
    X_mono = v_true + noise_mono_single
    top_n_mono = np.argsort(X_mono)[-n:]
    perf_mono = np.mean(v_true[top_n_mono])
    
    # 3. Ensemble Monoculture
    # Each agent's noise value for a given candidate is equal to the mean
    # of the noise values for that candidate of all of the agents in Polyculture.
    noise_ensemble = np.mean(noise_poly, axis=0)
    X_ensemble = v_true + noise_ensemble
    top_n_ensemble = np.argsort(X_ensemble)[-n:]
    perf_ensemble = np.mean(v_true[top_n_ensemble])

    # Baselines: average true value of the best and worst n candidates
    sorted_v = np.sort(v_true)
    perf_best = np.mean(sorted_v[-n:])
    perf_worst = np.mean(sorted_v[:n])

    # Metric: fraction of available value captured = (actual - worst) / (best - worst)
    # Higher is better; 1 = perfect hiring, 0 = worst possible hiring
    denom = perf_best - perf_worst
    metric_poly     = (perf_poly     - perf_worst) / denom
    metric_mono     = (perf_mono     - perf_worst) / denom
    metric_ensemble = (perf_ensemble - perf_worst) / denom

    return metric_poly, metric_mono, metric_ensemble

def run_simulation_set(n, num_runs=1000):
    seeds = [np.random.randint(0, 2**32 - 1) for _ in range(num_runs)]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(run_single_simulation, [(n, 1000, seed) for seed in seeds])
    
    results = np.array(results)
    mean_results = np.mean(results, axis=0)
    return mean_results

if __name__ == "__main__":
    ns = [100, 300, 500, 700, 900]
    num_runs = 1000
    
    print(f"Performance metric: fraction of available value captured, higher is better")
    print(f"{'n':<5} | {'Polyculture':<15} | {'Monoculture':<15} | {'Ensemble':<15}")
    print("-" * 55)
    
    start_time = time.time()
    for n in ns:
        metric_poly, metric_mono, metric_ensemble = run_simulation_set(n, num_runs)
        print(f"{n:<5} | {metric_poly:<15.4f} | {metric_mono:<15.4f} | {metric_ensemble:<15.4f}")
    
    end_time = time.time()
    # print(f"\nTotal time: {end_time - start_time:.2f} seconds")
