import numpy as np

def deferred_acceptance(firm_prefs, candidate_prefs, firm_capacities):
    """
    Implements the Gale-Shapley deferred acceptance algorithm.
    This version is candidate-proposing.
    """
    num_firms = len(firm_prefs)
    num_candidates = len(candidate_prefs)

    # Invert firm_prefs for quick lookup: firm_pref_rank[firm_id][candidate_id] = rank
    firm_pref_rank = {f: {c: i for i, c in enumerate(prefs)} for f, prefs in firm_prefs.items()}

    # Track which firm each candidate should propose to next
    candidate_next_proposal = {c: 0 for c in range(num_candidates)}

    # List of free candidates, initially all candidates
    free_candidates = list(range(num_candidates))

    # Matches: firm_id -> list of candidate_ids
    matches = {f: [] for f in range(num_firms)}

    while free_candidates:
        candidate_id = free_candidates.pop(0)

        # If candidate has proposed to all firms on their list, they remain unmatched
        if candidate_next_proposal[candidate_id] >= len(candidate_prefs[candidate_id]):
            continue

        firm_id = candidate_prefs[candidate_id][candidate_next_proposal[candidate_id]]
        candidate_next_proposal[candidate_id] += 1

        # If firm has an open slot
        if len(matches[firm_id]) < firm_capacities[firm_id]:
            matches[firm_id].append(candidate_id)
        else:
            # Firm is full, check if this candidate is preferred
            # Find the worst candidate currently matched to the firm
            current_matched_candidates = matches[firm_id]
            # Higher rank number is worse
            worst_candidate_rank = -1
            worst_candidate_id = -1
            for c in current_matched_candidates:
                rank = firm_pref_rank[firm_id].get(c, float('inf'))
                if rank > worst_candidate_rank:
                    worst_candidate_rank = rank
                    worst_candidate_id = c

            # If the new candidate is better than the worst one
            new_candidate_rank = firm_pref_rank[firm_id].get(candidate_id, float('inf'))
            if new_candidate_rank < worst_candidate_rank:
                # Bump the worst candidate
                matches[firm_id].remove(worst_candidate_id)
                free_candidates.append(worst_candidate_id)
                # Accept the new candidate
                matches[firm_id].append(candidate_id)
            else:
                # Candidate is rejected, remains free
                free_candidates.append(candidate_id)

    return matches

def run_simulation(n_firms, k_candidates, num_runs):
    """
    Runs the full simulation for a given number of firms and candidates.
    """
    firm_capacity = 10

    # Performance storage
    polyculture_performance = []
    monoculture_performance = []
    monoculture_ensembled_performance = []

    for _ in range(num_runs):
        # 1. Construct candidates: true values and preferences for firms
        candidate_true_values = np.random.normal(0, 1, k_candidates)
        candidate_prefs = {i: np.random.permutation(n_firms).tolist() for i in range(k_candidates)}
        firm_capacities = {i: firm_capacity for i in range(n_firms)}

        # --- Polyculture ---
        polyculture_noises = np.random.normal(0, 0.5, (n_firms, k_candidates))
        polyculture_estimated_values = candidate_true_values + polyculture_noises
        polyculture_firm_prefs = {i: np.argsort(-polyculture_estimated_values[i]).tolist() for i in range(n_firms)}

        polyculture_matches = deferred_acceptance(polyculture_firm_prefs, candidate_prefs, firm_capacities)

        matched_candidates_poly = [candidate for firm_matches in polyculture_matches.values() for candidate in firm_matches]
        if matched_candidates_poly:
            polyculture_performance.append(np.mean(candidate_true_values[matched_candidates_poly]))
        else:
            polyculture_performance.append(0)

        # --- Monoculture ---
        monoculture_noise = np.random.normal(0, 0.5, k_candidates)
        monoculture_estimated_values = candidate_true_values + monoculture_noise
        mono_pref_list = np.argsort(-monoculture_estimated_values).tolist()
        monoculture_firm_prefs = {i: mono_pref_list for i in range(n_firms)}

        monoculture_matches = deferred_acceptance(monoculture_firm_prefs, candidate_prefs, firm_capacities)

        matched_candidates_mono = [candidate for firm_matches in monoculture_matches.values() for candidate in firm_matches]
        if matched_candidates_mono:
            monoculture_performance.append(np.mean(candidate_true_values[matched_candidates_mono]))
        else:
            monoculture_performance.append(0)

        # --- Monoculture_Ensembled ---
        ensembled_noise = np.mean(polyculture_noises, axis=0)
        ensembled_estimated_values = candidate_true_values + ensembled_noise
        ensem_pref_list = np.argsort(-ensembled_estimated_values).tolist()
        monoculture_ensembled_firm_prefs = {i: ensem_pref_list for i in range(n_firms)}

        monoculture_ensembled_matches = deferred_acceptance(monoculture_ensembled_firm_prefs, candidate_prefs, firm_capacities)

        matched_candidates_ensem = [candidate for firm_matches in monoculture_ensembled_matches.values() for candidate in firm_matches]
        if matched_candidates_ensem:
            monoculture_ensembled_performance.append(np.mean(candidate_true_values[matched_candidates_ensem]))
        else:
            monoculture_ensembled_performance.append(0)

    return {
        "Polyculture": np.mean(polyculture_performance),
        "Monoculture": np.mean(monoculture_performance),
        "Monoculture_Ensembled": np.mean(monoculture_ensembled_performance)
    }

def main():
    k_candidates = 1000
    num_runs = 100
    firm_ns = [10, 25, 50, 75, 90]

    print("Running hiring simulation...")
    print(f"Parameters: {k_candidates} candidates, {num_runs} runs per scenario.")

    for n in firm_ns:
        print(f"\n--- Simulating for {n} firms ---")
        results = run_simulation(n, k_candidates, num_runs)
        print(f"  Polyculture Performance: {results['Polyculture']:.4f}")
        print(f"  Monoculture Performance: {results['Monoculture']:.4f}")
        print(f"  Monoculture_Ensembled Performance: {results['Monoculture_Ensembled']:.4f}")

if __name__ == "__main__":
    main()
