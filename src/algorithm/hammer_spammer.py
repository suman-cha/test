"""
Hammer-Spammer Model Algorithm (Problem 1).

This module implements the SVD-based ranking algorithm for selecting
the best answer from pairwise comparisons in a hammer-spammer setting.

Algorithm Overview:
1. Given comparison matrix R ∈ {±1}^(N×N)
2. Approximate R ≈ σ₁ u₁ v₁ᵀ (rank-1 SVD)
3. Compute quality scores q = (u₁ + v₁) / 2
4. Select answer with highest quality score
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.linalg import svd


class HammerSpammerRanking:
    """
    SVD-based ranking algorithm for Hammer-Spammer model.

    The algorithm uses low-rank matrix approximation to identify
    high-quality agents (hammers) and low-quality agents (spammers),
    then selects the best answer based on agent quality scores.
    """

    def __init__(self, beta: float = 5.0, epsilon: float = 0.1):
        """
        Initialize Hammer-Spammer ranking algorithm.

        Args:
            beta: Bradley-Terry discrimination parameter in [1, 10]
            epsilon: Prior probability of an agent being a spammer
        """
        self.beta = beta
        self.epsilon = epsilon

        # Statistics
        self.last_singular_values = None
        self.last_quality_scores = None
        self.last_left_singular = None
        self.last_right_singular = None

    def select_best(self, R: np.ndarray, return_scores: bool = False) -> int:
        """
        Select the best answer from comparison matrix.

        Args:
            R: Comparison matrix (N, N) with values in {-1, 1}
            return_scores: If True, return (index, scores), else just index

        Returns:
            Index of best answer (or tuple of (index, scores) if return_scores=True)
        """
        # Compute quality scores via SVD
        quality_scores = self.compute_quality_scores(R)

        # Select answer with highest quality score
        best_idx = int(np.argmax(quality_scores))

        if return_scores:
            return best_idx, quality_scores
        else:
            return best_idx

    def compute_quality_scores(self, R: np.ndarray) -> np.ndarray:
        """
        Compute quality scores for all agents using SVD.

        The quality score for agent i is computed as:
            q_i = (u₁[i] + v₁[i]) / 2

        where u₁ and v₁ are the left and right singular vectors
        corresponding to the largest singular value.

        Args:
            R: Comparison matrix (N, N)

        Returns:
            Quality scores (N,)
        """
        N = R.shape[0]

        # Verify matrix properties
        assert R.shape == (N, N), f"R must be square, got {R.shape}"
        assert np.all(np.isin(R, [-1, 1])), "R must contain only ±1"

        # Perform SVD: R ≈ U Σ Vᵀ
        U, S, Vh = svd(R, full_matrices=False)

        # Store statistics
        self.last_singular_values = S
        self.last_left_singular = U[:, 0]
        self.last_right_singular = Vh[0, :]

        # Extract first singular vectors
        u1 = U[:, 0]  # Left singular vector (N,)
        v1 = Vh[0, :]  # Right singular vector (N,)

        # Ensure consistent sign (u1 and v1 should point in same direction)
        # We want both to have positive correlation with row sums
        row_sums = R.sum(axis=1)
        if np.dot(u1, row_sums) < 0:
            u1 = -u1
        if np.dot(v1, row_sums) < 0:
            v1 = -v1

        # Compute quality scores as average of left and right singular vectors
        quality_scores = (u1 + v1) / 2.0

        self.last_quality_scores = quality_scores

        return quality_scores

    def rank_all_answers(self, R: np.ndarray) -> np.ndarray:
        """
        Rank all answers from best to worst.

        Args:
            R: Comparison matrix (N, N)

        Returns:
            Indices sorted by quality (best first)
        """
        quality_scores = self.compute_quality_scores(R)
        return np.argsort(quality_scores)[::-1]  # Descending order

    def identify_hammers_spammers(self, R: np.ndarray, threshold: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Identify hammers (high-quality agents) and spammers (low-quality agents).

        Args:
            R: Comparison matrix (N, N)
            threshold: Quality threshold (default: median)

        Returns:
            Dictionary with:
                - hammers: indices of high-quality agents
                - spammers: indices of low-quality agents
                - quality_scores: all quality scores
        """
        quality_scores = self.compute_quality_scores(R)

        if threshold is None:
            threshold = np.median(quality_scores)

        hammers = np.where(quality_scores >= threshold)[0]
        spammers = np.where(quality_scores < threshold)[0]

        return {
            'hammers': hammers,
            'spammers': spammers,
            'quality_scores': quality_scores,
            'threshold': threshold
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the last computation.

        Returns:
            Dictionary with diagnostic information
        """
        if self.last_quality_scores is None:
            return {'error': 'No computation performed yet'}

        N = len(self.last_quality_scores)

        return {
            'num_agents': N,
            'singular_values': self.last_singular_values.tolist() if self.last_singular_values is not None else None,
            'rank_1_ratio': (self.last_singular_values[0] / self.last_singular_values.sum()
                           if self.last_singular_values is not None else None),
            'quality_scores': self.last_quality_scores.tolist(),
            'quality_mean': float(np.mean(self.last_quality_scores)),
            'quality_std': float(np.std(self.last_quality_scores)),
            'quality_min': float(np.min(self.last_quality_scores)),
            'quality_max': float(np.max(self.last_quality_scores)),
        }

    def print_diagnostics(self):
        """Print diagnostic information."""
        diag = self.get_diagnostics()

        if 'error' in diag:
            print(f"Error: {diag['error']}")
            return

        print(f"\n{'='*60}")
        print(f"Hammer-Spammer Algorithm Diagnostics")
        print(f"{'='*60}\n")

        print(f"Number of agents: {diag['num_agents']}")
        print(f"\nSingular values (top 5): {diag['singular_values'][:5]}")
        print(f"Rank-1 approximation ratio: {diag['rank_1_ratio']:.4f}")

        print(f"\nQuality scores:")
        print(f"  Mean: {diag['quality_mean']:.4f}")
        print(f"  Std: {diag['quality_std']:.4f}")
        print(f"  Range: [{diag['quality_min']:.4f}, {diag['quality_max']:.4f}]")

        print(f"\nTop 5 agents by quality:")
        sorted_indices = np.argsort(self.last_quality_scores)[::-1][:5]
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"  {rank}. Agent {idx}: {self.last_quality_scores[idx]:.4f}")

    def __repr__(self) -> str:
        """String representation."""
        return f"HammerSpammerRanking(beta={self.beta}, epsilon={self.epsilon})"


# Additional utility functions

def simulate_comparison_matrix(N: int, beta: float = 5.0,
                              epsilon: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a comparison matrix following the Hammer-Spammer model
    as specified in the problem statement.

    Model:
        - s_i ~ U(0, 1) for each agent i
        - z_i ∈ {H, S} with P(z_i = S) = epsilon
        - Hammer (z_i = H): P(R_ij = 1 | s, z_i = H) = sigma(beta * (s_i - s_j))
        - Spammer (z_i = S): R_ij ~ uniform({-1, 1})
        - R_ii = 1 for all i

    Args:
        N: Number of agents
        beta: Bradley-Terry discrimination parameter (in [1, 10])
        epsilon: Probability of being a spammer

    Returns:
        Tuple of (comparison_matrix, true_scores, agent_types)
        where agent_types[i] = 1 for hammer, 0 for spammer
    """
    # Generate true latent scores: s_i ~ U(0, 1)
    scores = np.random.uniform(0, 1, N)

    # Generate agent types: z_i ∈ {H, S}
    agent_types = (np.random.random(N) >= epsilon).astype(int)  # 1=hammer, 0=spammer

    # Generate comparison matrix
    R = np.ones((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            if i == j:
                R[i, j] = 1  # Diagonal
            elif agent_types[i] == 1:
                # Hammer: P(R_ij = 1) = sigma(beta * (s_i - s_j))
                prob = 1.0 / (1.0 + np.exp(-beta * (scores[i] - scores[j])))
                R[i, j] = 1 if np.random.random() < prob else -1
            else:
                # Spammer: R_ij ~ uniform({-1, 1})
                R[i, j] = 1 if np.random.random() < 0.5 else -1

    return R, scores, agent_types


if __name__ == "__main__":
    # Test the algorithm
    print("Testing Hammer-Spammer Ranking Algorithm...")

    np.random.seed(42)

    # Simulate a comparison matrix using the correct model
    print("\n=== Simulating Comparison Matrix ===")
    N = 15
    beta = 5.0
    epsilon = 0.3  # 30% spammer rate

    R, true_scores, agent_types = simulate_comparison_matrix(N, beta=beta, epsilon=epsilon)

    num_hammers = np.sum(agent_types == 1)
    num_spammers = np.sum(agent_types == 0)
    print(f"Generated {N} agents: {num_hammers} hammers, {num_spammers} spammers")
    print(f"True scores (first 5): {true_scores[:5]}")
    print(f"Agent types (first 5): {['H' if t else 'S' for t in agent_types[:5]]}")
    print(f"\nComparison matrix R (shape {R.shape}):")
    print(f"  Positive: {np.sum(R == 1)} ({100*np.sum(R == 1)/R.size:.1f}%)")
    print(f"  Negative: {np.sum(R == -1)} ({100*np.sum(R == -1)/R.size:.1f}%)")

    # Apply algorithm
    print("\n\n=== Applying Hammer-Spammer Algorithm ===")
    algorithm = HammerSpammerRanking(beta=beta, epsilon=epsilon)

    best_idx, quality_scores = algorithm.select_best(R, return_scores=True)

    true_best = np.argmax(true_scores)
    print(f"\nSelected best answer: Agent {best_idx}")
    print(f"Quality score: {quality_scores[best_idx]:.4f}")
    print(f"True score: {true_scores[best_idx]:.4f}")
    print(f"True best agent: {true_best} (score: {true_scores[true_best]:.4f})")
    print(f"Correct selection: {best_idx == true_best}")

    # Identify hammers and spammers
    print("\n\n=== Identifying Hammers and Spammers ===")
    classification = algorithm.identify_hammers_spammers(R)

    print(f"Detected {len(classification['hammers'])} hammers:")
    print(f"  Indices: {classification['hammers']}")
    print(f"  Avg quality: {np.mean(quality_scores[classification['hammers']]):.4f}")

    print(f"\nDetected {len(classification['spammers'])} spammers:")
    print(f"  Indices: {classification['spammers']}")
    print(f"  Avg quality: {np.mean(quality_scores[classification['spammers']]):.4f}")

    # Print diagnostics
    algorithm.print_diagnostics()

    # Compare with true scores
    print("\n\n=== Comparison with True Scores ===")
    print(f"Correlation: {np.corrcoef(quality_scores, true_scores)[0,1]:.4f}")

    # Test ranking
    print("\n\n=== Ranking All Answers ===")
    ranking = algorithm.rank_all_answers(R)
    print(f"Top 5 agents by estimated quality: {ranking[:5]}")
    true_ranking = np.argsort(true_scores)[::-1]
    print(f"Top 5 agents by true score: {true_ranking[:5]}")
