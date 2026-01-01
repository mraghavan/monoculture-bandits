import numpy as np

def deferred_acceptance_firm_proposing(firm_prefs, candidate_prefs, firm_capacities):
    """
    Implements the Gale-Shapley deferred acceptance algorithm.
    This version is firm-proposing.
    """
    # Track which candidate each firm should propose to next
    firm_next_proposal = {f: 0 for f in firm_prefs.keys()}

    # Final matches for firms
    firm_matches = {f: [] for f in firm_prefs.keys()}

    # Tentative matches for candidates (candidate_id -> firm_id)
    candidate_matches = {}

    # Invert candidate_prefs for quick lookup of ranks
    candidate_pref_rank = {c: {f: i for i, f in enumerate(prefs)} for c, prefs in candidate_prefs.items()}

    # Initially, all firms have open slots and are "free" to make proposals
    free_firms = list(firm_prefs.keys())

    while free_firms:
        firm_id = free_firms.pop(0)

        # Get the list of candidates this firm can still propose to
        proposals_to_make = firm_prefs[firm_id]
        proposal_index = firm_next_proposal[firm_id]

        if proposal_index < len(proposals_to_make):
            candidate_id = proposals_to_make[proposal_index]
            firm_next_proposal[firm_id] += 1

            current_match = candidate_matches.get(candidate_id)

            if current_match is None:
                # Candidate is free and accepts the proposal
                candidate_matches[candidate_id] = firm_id
                firm_matches[firm_id].append(candidate_id)
                # If the firm is not full, it needs to propose again
                if len(firm_matches[firm_id]) < firm_capacities[firm_id]:
                    free_firms.append(firm_id)
            else:
                # Candidate is already matched, check preferences
                current_firm_rank = candidate_pref_rank[candidate_id].get(current_match, float('inf'))
                new_firm_rank = candidate_pref_rank[candidate_id].get(firm_id, float('inf'))

                if new_firm_rank < current_firm_rank:
                    # Candidate prefers the new firm
                    # The old firm is now rejected and has a free slot
                    firm_matches[current_match].remove(candidate_id)
                    free_firms.append(current_match)

                    # Candidate's new match is this firm
                    candidate_matches[candidate_id] = firm_id
                    firm_matches[firm_id].append(candidate_id)

                    # If the new firm is not full, it must continue proposing
                    if len(firm_matches[firm_id]) < firm_capacities[firm_id]:
                        free_firms.append(firm_id)
                else:
                    # Candidate rejects the new proposal, firm must propose to next
                    free_firms.append(firm_id)

    return firm_matches

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

        polyculture_matches = deferred_acceptance_firm_proposing(polyculture_firm_prefs, candidate_prefs, firm_capacities)

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

        monoculture_matches = deferred_acceptance_firm_proposing(monoculture_firm_prefs, candidate_prefs, firm_capacities)

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

        monoculture_ensembled_matches = deferred_acceptance_firm_proposing(monoculture_ensembled_firm_prefs, candidate_prefs, firm_capacities)

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
