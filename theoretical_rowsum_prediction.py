"""
Theoretical Analysis: What correlation should we expect?

Under Bradley-Terry model with hammer-spammer:
  P(R_ij = 1 | z_i=H) = σ(β(s_i - s_j))
  P(R_ij = 1 | z_i=S) = 0.5

If this holds, what correlation between row_sum and correctness
should we observe?
"""

import numpy as np
from scipy.stats import pearsonr


def simulate_ideal_bradley_terry(
    N: int = 15,
    num_questions: int = 50,
    beta: float = 5.0,
    epsilon: float = 0.1,
    seed: int = 42
):
    """
    Simulate ideal Bradley-Terry scenario.

    In this scenario:
    - Agents have true quality scores
    - Correct answer → high score
    - Comparisons follow Bradley-Terry exactly
    - No bias, no wrong clusters
    """
    np.random.seed(seed)

    all_row_sums = []
    all_correct = []

    for q in range(num_questions):
        # True quality scores
        s_true = np.random.uniform(0, 1, N)

        # True best answer (highest score = correct)
        true_best = np.argmax(s_true)

        # Agent types
        z = np.random.binomial(1, 1 - epsilon, N)  # 1=hammer, 0=spammer

        # Generate R according to Bradley-Terry
        R = np.ones((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                if z[i] == 1:  # Hammer
                    prob = 1.0 / (1.0 + np.exp(-beta * (s_true[i] - s_true[j])))
                    R[i, j] = 1 if np.random.random() < prob else -1
                else:  # Spammer
                    R[i, j] = 1 if np.random.random() < 0.5 else -1

        # Compute row sums and correctness
        for i in range(N):
            row_sum = (R[i, :].sum() - 1) / (N - 1)
            is_correct = (i == true_best)

            all_row_sums.append(row_sum)
            all_correct.append(is_correct)

    # Compute correlation
    all_row_sums = np.array(all_row_sums)
    all_correct = np.array(all_correct, dtype=float)

    r, p = pearsonr(all_row_sums, all_correct)

    return r, p, all_row_sums, all_correct


def simulate_wrong_cluster_scenario(
    N: int = 15,
    num_questions: int = 50,
    wrong_fraction: float = 0.6,
    seed: int = 42
):
    """
    Simulate wrong cluster scenario.

    In this scenario:
    - Majority (60%) give same wrong answer
    - They agree with each other (high row sum)
    - Minority (40%) give correct answer
    - They disagree (low row sum)
    """
    np.random.seed(seed)

    all_row_sums = []
    all_correct = []

    for q in range(num_questions):
        n_wrong = int(N * wrong_fraction)
        n_correct = N - n_wrong

        # Generate R where wrong cluster agrees
        R = np.ones((N, N), dtype=int)

        wrong_agents = list(range(n_wrong))
        correct_agents = list(range(n_wrong, N))

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                # Wrong agents agree with each other
                if i in wrong_agents and j in wrong_agents:
                    R[i, j] = np.random.choice([1, -1], p=[0.75, 0.25])
                # Correct agents agree somewhat
                elif i in correct_agents and j in correct_agents:
                    R[i, j] = np.random.choice([1, -1], p=[0.65, 0.35])
                # Cross: disagree
                else:
                    R[i, j] = np.random.choice([1, -1], p=[0.35, 0.65])

        # Compute row sums
        for i in range(N):
            row_sum = (R[i, :].sum() - 1) / (N - 1)
            is_correct = (i in correct_agents)

            all_row_sums.append(row_sum)
            all_correct.append(is_correct)

    # Compute correlation
    all_row_sums = np.array(all_row_sums)
    all_correct = np.array(all_correct, dtype=float)

    r, p = pearsonr(all_row_sums, all_correct)

    return r, p, all_row_sums, all_correct


def main():
    """Compare theoretical scenarios."""
    print("="*70)
    print("THEORETICAL PREDICTION: Row Sum vs Correctness Correlation")
    print("="*70)
    print()

    # Scenario 1: Ideal Bradley-Terry
    print("SCENARIO 1: Ideal Bradley-Terry Model")
    print("-"*70)
    print("Assumptions:")
    print("  - Comparisons based on true quality")
    print("  - Correct answer has highest quality")
    print("  - No bias, no wrong clusters")
    print()

    r_ideal, p_ideal, _, _ = simulate_ideal_bradley_terry(
        N=15, num_questions=50, beta=5.0, epsilon=0.1
    )

    print(f"Expected correlation: r = {r_ideal:+.4f} (p = {p_ideal:.4e})")
    print()

    if r_ideal > 0.3:
        print("✓ Strong positive correlation expected")
        print("  → Algorithm should work well")
    else:
        print("⚠️  Weaker than expected (noise from spammers)")

    # Scenario 2: Wrong cluster
    print()
    print("="*70)
    print("SCENARIO 2: Wrong Answer Cluster")
    print("-"*70)
    print("Assumptions:")
    print("  - 60% agents give same wrong answer")
    print("  - Wrong cluster has high internal agreement")
    print("  - 40% correct agents are minority")
    print()

    r_wrong, p_wrong, row_sums_wrong, correct_wrong = simulate_wrong_cluster_scenario(
        N=15, num_questions=50, wrong_fraction=0.6
    )

    print(f"Expected correlation: r = {r_wrong:+.4f} (p = {p_wrong:.4e})")
    print()

    if r_wrong < -0.1:
        print("❌ NEGATIVE correlation!")
        print("  → Wrong agents have HIGHER row sums")
        print("  → Algorithm will select wrong cluster")
    elif abs(r_wrong) < 0.1:
        print("⚠️  NO correlation")
        print("  → Row sum is uninformative")
    else:
        print("~ Weak positive (but much worse than ideal)")

    # Compare row sum distributions
    wrong_correct = row_sums_wrong[correct_wrong == 1]
    wrong_incorrect = row_sums_wrong[correct_wrong == 0]

    print()
    print(f"Row sum distribution:")
    print(f"  Correct answers:   mean = {wrong_correct.mean():.3f}")
    print(f"  Incorrect answers: mean = {wrong_incorrect.mean():.3f}")
    print(f"  Difference:        {wrong_correct.mean() - wrong_incorrect.mean():+.3f}")

    # Comparison
    print()
    print("="*70)
    print("COMPARISON")
    print("="*70)
    print()
    print(f"{'Scenario':<30} {'Correlation':<15} {'Algorithm Result'}")
    print("-"*70)
    print(f"{'Ideal Bradley-Terry':<30} r = {r_ideal:+.3f}       Works well")
    print(f"{'Wrong Cluster':<30} r = {r_wrong:+.3f}       Fails badly")
    print()

    print("What to check in your data:")
    print(f"  If r > +0.3:  ✓ Algorithm should work")
    print(f"  If r ≈ 0:     ⚠️  Algorithm = Majority Voting")
    print(f"  If r < -0.2:  ❌ Algorithm worse than MV")
    print()

    print("="*70)
    print("TO RUN ON YOUR DATA:")
    print("="*70)
    print()
    print("  python analyze_rowsum_correctness.py results/your_results.json")
    print()


if __name__ == "__main__":
    main()
