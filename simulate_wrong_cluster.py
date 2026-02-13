"""
Simulation: Wrong Answer Cluster Phenomenon

This simulates the scenario where:
1. Most agents give the SAME wrong answer
2. Minority agents give the correct answer
3. Comparisons are based on confidence/length, not correctness

Goal: Show that CCRR will select the wrong cluster as "hammers"
"""

import numpy as np
from src.algorithm.ccrr import CCRRanking


def simulate_wrong_cluster_scenario(
    N: int = 15,
    n_correct: int = 3,
    wrong_agreement: float = 0.8,
    correct_agreement: float = 0.6,
    cross_agreement: float = 0.3,
    seed: int = 42
):
    """
    Simulate wrong answer cluster scenario.

    Args:
        N: Total agents
        n_correct: Number of agents with correct answer
        wrong_agreement: Agreement probability within wrong cluster
        correct_agreement: Agreement probability within correct group
        cross_agreement: Agreement probability between correct and wrong
        seed: Random seed

    Returns:
        Comparison matrix R
    """
    np.random.seed(seed)

    n_wrong = N - n_correct

    print("="*70)
    print("SIMULATING WRONG ANSWER CLUSTER")
    print("="*70)
    print(f"\nSetup:")
    print(f"  Total agents: {N}")
    print(f"  Correct agents: {n_correct} (minority)")
    print(f"  Wrong agents: {n_wrong} (majority, same wrong answer)")
    print(f"  Wrong cluster agreement: {wrong_agreement}")
    print(f"  Correct group agreement: {correct_agreement}")
    print(f"  Cross-group agreement: {cross_agreement}")
    print()

    # Agent types: first n_correct are correct, rest are wrong
    correct_agents = list(range(n_correct))
    wrong_agents = list(range(n_correct, N))

    # Generate comparison matrix
    R = np.ones((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            if i == j:
                R[i, j] = 1
                continue

            # Determine agreement probability
            if i in correct_agents and j in correct_agents:
                # Both correct: high agreement
                p_agree = correct_agreement
            elif i in wrong_agents and j in wrong_agents:
                # Both wrong (same answer): very high agreement
                p_agree = wrong_agreement
            else:
                # One correct, one wrong: low agreement
                p_agree = cross_agreement

            # R[i,j] = 1 means agent i thinks i is better than j
            # If they agree on answers, they might still disagree on which is better
            # For simplicity: agreement means similar comparison

            if np.random.random() < p_agree:
                # They agree: both say the same side wins
                # Randomly pick a side
                R[i, j] = np.random.choice([1, -1])
            else:
                # They disagree
                R[i, j] = -R[j, i] if j < i else np.random.choice([1, -1])

    return R, correct_agents, wrong_agents


def analyze_simulation_result(R, correct_agents, wrong_agents):
    """Analyze what the algorithm does."""
    N = R.shape[0]

    print("="*70)
    print("ANALYSIS")
    print("="*70)

    # Apply CCRR
    ccrr = CCRRanking(beta=5.0, epsilon=0.1, T=5)
    selected_idx, scores, weights = ccrr.select_best(R, return_details=True)

    print(f"\nCCRR selected: Agent {selected_idx}")
    print(f"  Is correct? {'YES ✓' if selected_idx in correct_agents else 'NO ✗'}")
    print()

    # Analyze weights
    correct_weights = weights[correct_agents]
    wrong_weights = weights[wrong_agents]

    print(f"Weights analysis:")
    print(f"  Correct agents avg weight: {correct_weights.mean():.3f}")
    print(f"  Wrong agents avg weight:   {wrong_weights.mean():.3f}")
    print(f"  Difference: {wrong_weights.mean() - correct_weights.mean():+.3f}")
    print()

    if wrong_weights.mean() > correct_weights.mean():
        print("  ❌ WRONG CLUSTER HAS HIGHER WEIGHTS!")
        print("     Algorithm thinks the wrong cluster are 'hammers'")
    else:
        print("  ✓ Correct agents have higher weights")

    # Analyze scores
    correct_scores = scores[correct_agents]
    wrong_scores = scores[wrong_agents]

    print(f"\nScores analysis:")
    print(f"  Correct agents avg score: {correct_scores.mean():+.3f}")
    print(f"  Wrong agents avg score:   {wrong_scores.mean():+.3f}")
    print()

    # Compute row-norms
    M = R.astype(float).copy()
    np.fill_diagonal(M, 0.0)
    G = M @ M.T / (N - 2) if N > 2 else M @ M.T
    np.fill_diagonal(G, 0.0)
    rho = np.sum(G ** 2, axis=1)

    correct_rho = rho[correct_agents]
    wrong_rho = rho[wrong_agents]

    print(f"Row-norm (ρ) analysis:")
    print(f"  Correct agents avg ρ: {correct_rho.mean():.3f}")
    print(f"  Wrong agents avg ρ:   {wrong_rho.mean():.3f}")
    print()

    tau = np.median(rho) / 2.0
    print(f"  Threshold τ = {tau:.3f}")

    correct_above = sum(1 for r in correct_rho if r > tau)
    wrong_above = sum(1 for r in wrong_rho if r > tau)

    print(f"  Correct agents above τ: {correct_above}/{len(correct_agents)}")
    print(f"  Wrong agents above τ:   {wrong_above}/{len(wrong_agents)}")
    print()

    if wrong_above > correct_above:
        print("  ❌ MORE WRONG AGENTS CLASSIFIED AS HAMMERS!")

    # Agreement matrix analysis
    print(f"Agreement analysis:")

    # Within wrong cluster
    wrong_agreements = []
    for i in wrong_agents:
        for j in wrong_agents:
            if i < j:
                agreement = (R[i, :] @ R[j, :]) / N
                wrong_agreements.append(agreement)

    # Within correct group
    correct_agreements = []
    for i in correct_agents:
        for j in correct_agents:
            if i < j:
                agreement = (R[i, :] @ R[j, :]) / N
                correct_agreements.append(agreement)

    # Cross-group
    cross_agreements = []
    for i in correct_agents:
        for j in wrong_agents:
            agreement = (R[i, :] @ R[j, :]) / N
            cross_agreements.append(agreement)

    if wrong_agreements:
        print(f"  Wrong cluster internal agreement: {np.mean(wrong_agreements):.3f}")
    if correct_agreements:
        print(f"  Correct group internal agreement: {np.mean(correct_agreements):.3f}")
    if cross_agreements:
        print(f"  Cross-group agreement: {np.mean(cross_agreements):.3f}")

    print()
    print("="*70)
    print("CONCLUSION")
    print("="*70)

    if selected_idx in wrong_agents:
        print("\n❌ ALGORITHM SELECTED FROM WRONG CLUSTER!")
        print("\nWhy this happened:")
        print("1. Wrong agents form a coherent cluster (high agreement)")
        print("2. High agreement → high row-norms → classified as 'hammers'")
        print("3. Correct agents are minority → low agreement → 'spammers'")
        print("4. Algorithm selects from wrong 'hammer' cluster")
        print("\nThis is the FUNDAMENTAL FLAW when:")
        print("  - Wrong answer is more popular")
        print("  - Comparisons based on confidence, not correctness")
        print("  - Agreement ≠ Correctness")
    else:
        print("\n✓ Algorithm happened to select correctly")
        print("  (But analysis shows the structural bias)")

    # Compare with majority voting
    print("\n" + "="*70)
    print("MAJORITY VOTING")
    print("="*70)

    # Simulate answer distribution
    # Assume correct agents give answer "45", wrong agents give "315"
    answers = ["315"] * N
    for i in correct_agents:
        answers[i] = "45"

    from collections import Counter
    answer_counts = Counter(answers)
    mv_answer = answer_counts.most_common(1)[0][0]

    print(f"\nAnswer distribution:")
    for ans, count in answer_counts.most_common():
        is_correct = (ans == "45")
        marker = "✓" if is_correct else "✗"
        print(f"  {marker} '{ans}': {count} agents")

    print(f"\nMajority voting selects: '{mv_answer}'")
    print(f"  Is correct? {'YES ✓' if mv_answer == '45' else 'NO ✗'}")

    if mv_answer == "315":
        print("\n  ⚠️  Even majority voting fails when wrong answer is majority!")
    else:
        print("\n  ✓ Majority voting would get it right")

    print()
    print("="*70)


def run_scenarios():
    """Run multiple scenarios."""
    print("\n" + "="*70)
    print("SCENARIO 1: Wrong cluster is majority (typical failure)")
    print("="*70)
    R, correct, wrong = simulate_wrong_cluster_scenario(
        N=15,
        n_correct=3,  # Only 3 correct
        wrong_agreement=0.8,
        correct_agreement=0.6,
        cross_agreement=0.3
    )
    analyze_simulation_result(R, correct, wrong)

    print("\n\n" + "="*70)
    print("SCENARIO 2: Balanced (edge case)")
    print("="*70)
    R, correct, wrong = simulate_wrong_cluster_scenario(
        N=15,
        n_correct=7,  # Almost balanced
        wrong_agreement=0.8,
        correct_agreement=0.6,
        cross_agreement=0.3,
        seed=43
    )
    analyze_simulation_result(R, correct, wrong)


if __name__ == "__main__":
    run_scenarios()
