"""
Diversity-Aware CCRR: Penalize majority clusters.

Key idea: If a large cluster agrees, it might be wrong!
- Encourage diversity in final selection
- Penalize overly coherent clusters
- Balance agreement with diversity
"""

import numpy as np
from typing import Tuple
from collections import Counter


class DiversityAwareCCRR:
    """
    CCRR modified to avoid wrong answer clusters.

    Adds diversity penalty: clusters that are too large and too coherent
    may be wrong, so we penalize them.
    """

    def __init__(self, beta: float = 5.0, epsilon: float = 0.1, T: int = 5,
                 diversity_weight: float = 0.3):
        """
        Initialize diversity-aware CCRR.

        Args:
            beta: Bradley-Terry parameter
            epsilon: Spammer prior
            T: EM iterations
            diversity_weight: Weight for diversity penalty (0-1)
        """
        self.beta = beta
        self.epsilon = epsilon
        self.T = T
        self.diversity_weight = diversity_weight

    def select_best(self, R: np.ndarray, answers: list = None,
                   return_details: bool = False):
        """
        Select best with diversity awareness.

        Args:
            R: Comparison matrix
            answers: List of answer strings (for clustering)
            return_details: Return extra info

        Returns:
            Selected index (or tuple)
        """
        # Run standard CCRR first
        from src.algorithm.ccrr import CCRRanking

        ccrr = CCRRanking(self.beta, self.epsilon, self.T)
        _, scores, weights = ccrr.select_best(R, return_details=True)

        # If answers provided, apply diversity penalty
        if answers is not None:
            N = len(answers)

            # Group answers
            answer_groups = {}
            for i, ans in enumerate(answers):
                ans_key = ans.strip().lower()
                if ans_key not in answer_groups:
                    answer_groups[ans_key] = []
                answer_groups[ans_key].append(i)

            # Find largest cluster
            largest_cluster_size = max(len(indices) for indices in answer_groups.values())

            # Apply diversity penalty to agents in large clusters
            diversity_penalty = np.zeros(N)

            for ans_key, indices in answer_groups.items():
                cluster_size = len(indices)
                if cluster_size > N / 3:  # If cluster is > 33% of agents
                    # Compute coherence: do they agree with each other?
                    if len(indices) >= 2:
                        agreements = []
                        for i in indices:
                            for j in indices:
                                if i < j:
                                    agreement = (R[i, :] @ R[j, :]) / N
                                    agreements.append(agreement)

                        avg_agreement = np.mean(agreements) if agreements else 0

                        # High agreement + large cluster = suspicious
                        if avg_agreement > 0.5:
                            penalty = (cluster_size / N) * avg_agreement
                            for i in indices:
                                diversity_penalty[i] = penalty

            # Adjust scores
            adjusted_scores = scores - self.diversity_weight * diversity_penalty

        else:
            adjusted_scores = scores

        best_idx = int(np.argmax(adjusted_scores))

        if return_details:
            return best_idx, adjusted_scores, weights
        else:
            return best_idx


if __name__ == "__main__":
    print("Testing Diversity-Aware CCRR...")

    # Simulate wrong cluster scenario
    np.random.seed(42)
    N = 15

    # 3 agents answer correctly, 12 answer wrongly (same wrong answer)
    answers = ["315"] * 12 + ["45"] * 3

    # Generate R where wrong cluster agrees heavily
    R = np.ones((N, N), dtype=int)
    wrong_indices = list(range(12))
    correct_indices = list(range(12, 15))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # Wrong agents agree with each other
            if i in wrong_indices and j in wrong_indices:
                R[i, j] = np.random.choice([1, -1], p=[0.8, 0.2])
            # Correct agents agree with each other
            elif i in correct_indices and j in correct_indices:
                R[i, j] = np.random.choice([1, -1], p=[0.7, 0.3])
            # Cross-group: disagree
            else:
                R[i, j] = np.random.choice([1, -1], p=[0.3, 0.7])

    print(f"\nSetup:")
    print(f"  Agents 0-11: answer '315' (wrong)")
    print(f"  Agents 12-14: answer '45' (correct)")
    print()

    # Standard CCRR
    from src.algorithm.ccrr import CCRRanking
    ccrr = CCRRanking(beta=5.0, epsilon=0.1, T=5)
    standard_idx = ccrr.select_best(R)
    standard_answer = answers[standard_idx]

    print(f"Standard CCRR:")
    print(f"  Selected: Agent {standard_idx} → '{standard_answer}'")
    print(f"  Correct? {'YES ✓' if standard_answer == '45' else 'NO ✗'}")
    print()

    # Diversity-aware CCRR
    da_ccrr = DiversityAwareCCRR(beta=5.0, epsilon=0.1, T=5, diversity_weight=0.3)
    diverse_idx = da_ccrr.select_best(R, answers=answers)
    diverse_answer = answers[diverse_idx]

    print(f"Diversity-Aware CCRR:")
    print(f"  Selected: Agent {diverse_idx} → '{diverse_answer}'")
    print(f"  Correct? {'YES ✓' if diverse_answer == '45' else 'NO ✗'}")
    print()

    if diverse_answer == '45' and standard_answer == '315':
        print("✅ Diversity awareness FIXED the wrong cluster problem!")
    elif diverse_answer == standard_answer:
        print("⚠️  Both methods selected same answer")
    else:
        print("❌ Both methods failed (wrong cluster too strong)")
