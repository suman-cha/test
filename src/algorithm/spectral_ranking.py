"""
Spectral Ranking Algorithm for Hammer-Spammer Model (Problem 1)

Uses SVD to automatically estimate agent reliability and produce
weighted aggregation of comparisons.

Key insight:
- Hammer agents produce rows that share common patterns (true score order)
- Spammer agents produce random noise rows
- Top SVD component captures the shared hammer pattern
- u₁² provides reliability weights for each agent
"""

import numpy as np
from typing import Tuple, Dict, Any


class SpectralRanking:
    """
    Spectral Ranking algorithm using SVD-based agent weighting.

    This is a simpler, more direct algorithm compared to CCRRanking.
    It focuses on the core spectral insight: use SVD to find agent
    reliability weights, then do weighted column aggregation.
    """

    def __init__(self, beta: float = 5.0):
        """
        Initialize Spectral Ranking.

        Args:
            beta: Sigmoid sharpness parameter (not used in this algorithm,
                  kept for interface consistency)
        """
        self.beta = beta
        self.weights = None
        self.singular_values = None

    def select_best(self, R: np.ndarray, return_details: bool = False):
        """
        Select the best answer using spectral ranking.

        Args:
            R: N×N comparison matrix with values in {-1, +1}
            return_details: If True, return (index, scores, weights, details)
                          If False, return only index

        Returns:
            If return_details=False: best answer index
            If return_details=True: (best_idx, scores, weights, details_dict)
        """
        N = R.shape[0]

        # Step 1: Remove diagonal (self-comparisons are trivial)
        R_tilde = R.copy().astype(float)
        np.fill_diagonal(R_tilde, 0.0)

        # Step 2: SVD to estimate reliability
        # Use top singular component to find shared hammer pattern
        U, s, Vt = np.linalg.svd(R_tilde, full_matrices=False)

        # u₁: left singular vector (agent patterns)
        u1 = U[:, 0]

        # Reliability weights: square of u₁ components
        # (sign doesn't matter, only magnitude)
        weights = u1 ** 2

        # Normalize weights to sum to N (for interpretability)
        weights = weights * (N / weights.sum())

        # Store for diagnostics
        self.weights = weights
        self.singular_values = s

        # Step 3: Compute weighted scores
        # score_j = -Σᵢ wᵢ · Rᵢⱼ
        # Negative sign: R_ij = -1 means "agent i thinks j is better"
        # Higher score = better answer
        scores = np.zeros(N)
        for j in range(N):
            # Weighted sum of column j (excluding diagonal)
            scores[j] = -np.sum(weights * R_tilde[:, j])

        # Step 4: Select best
        best_idx = int(np.argmax(scores))

        if return_details:
            details = {
                'spectral_gap': float(s[0] - s[1]) if len(s) > 1 else float(s[0]),
                'top_singular_value': float(s[0]),
                'weight_std': float(np.std(weights)),
                'weight_range': (float(np.min(weights)), float(np.max(weights))),
            }
            return best_idx, scores, weights, details
        else:
            return best_idx

    def rank_all_answers(self, R: np.ndarray) -> np.ndarray:
        """
        Rank all answers by quality.

        Args:
            R: N×N comparison matrix

        Returns:
            Ranking array (indices sorted by quality, best first)
        """
        _, scores, _, _ = self.select_best(R, return_details=True)
        return np.argsort(-scores)  # Descending order

    def identify_hammers_spammers(self, R: np.ndarray,
                                   threshold: float = None) -> Dict[str, Any]:
        """
        Classify agents as hammers or spammers based on weights.

        Args:
            R: N×N comparison matrix
            threshold: Weight threshold (if None, use median)

        Returns:
            Dictionary with hammer/spammer indices and statistics
        """
        if self.weights is None:
            self.select_best(R, return_details=False)

        N = len(self.weights)

        # Default threshold: median
        if threshold is None:
            threshold = np.median(self.weights)

        hammer_mask = self.weights >= threshold
        spammer_mask = ~hammer_mask

        return {
            'hammer_indices': np.where(hammer_mask)[0].tolist(),
            'spammer_indices': np.where(spammer_mask)[0].tolist(),
            'num_hammers': int(np.sum(hammer_mask)),
            'num_spammers': int(np.sum(spammer_mask)),
            'threshold': float(threshold),
            'weights': self.weights.tolist(),
            'hammer_weight_mean': float(np.mean(self.weights[hammer_mask])) if np.any(hammer_mask) else 0.0,
            'spammer_weight_mean': float(np.mean(self.weights[spammer_mask])) if np.any(spammer_mask) else 0.0,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the last run."""
        if self.weights is None:
            return {'error': 'No weights computed yet'}

        return {
            'algorithm': 'SpectralRanking',
            'beta': self.beta,
            'weights': self.weights.tolist(),
            'weight_mean': float(np.mean(self.weights)),
            'weight_std': float(np.std(self.weights)),
            'singular_values': self.singular_values.tolist() if self.singular_values is not None else None,
            'spectral_gap': float(self.singular_values[0] - self.singular_values[1]) if len(self.singular_values) > 1 else None,
        }

    def __repr__(self) -> str:
        return f"SpectralRanking(beta={self.beta})"


def test_spectral_ranking():
    """Test the spectral ranking algorithm."""
    print("Testing Spectral Ranking Algorithm\n")

    # Test case: 5 agents, where agents 0-2 are hammers (agree on order)
    # and agents 3-4 are spammers (random)
    N = 5

    # True quality: agent 0 > 1 > 2 > 3 > 4
    # Hammers (0,1,2) compare correctly most of the time
    # Spammers (3,4) compare randomly

    R = np.array([
        [ 1,  1,  1,  1,  1],  # Agent 0: best, beats everyone
        [-1,  1,  1,  1,  1],  # Agent 1: second best
        [-1, -1,  1,  1,  1],  # Agent 2: third best
        [ 1, -1, -1,  1, -1],  # Agent 3: spammer (random)
        [-1,  1, -1,  1,  1],  # Agent 4: spammer (random)
    ])

    print("Comparison matrix R:")
    print(R)
    print()

    # Run algorithm
    sr = SpectralRanking(beta=5.0)
    best_idx, scores, weights, details = sr.select_best(R, return_details=True)

    print("Results:")
    print(f"  Best answer: {best_idx} (expected: 0)")
    print(f"  Scores: {scores}")
    print(f"  Weights: {weights}")
    print(f"  Spectral gap: {details['spectral_gap']:.3f}")
    print()

    # Identify hammers/spammers
    classification = sr.identify_hammers_spammers(R)
    print("Agent Classification:")
    print(f"  Hammers: {classification['hammer_indices']} (expected: [0, 1, 2])")
    print(f"  Spammers: {classification['spammer_indices']} (expected: [3, 4])")
    print(f"  Hammer avg weight: {classification['hammer_weight_mean']:.3f}")
    print(f"  Spammer avg weight: {classification['spammer_weight_mean']:.3f}")
    print()

    # Full ranking
    ranking = sr.rank_all_answers(R)
    print(f"Full ranking (best to worst): {ranking}")
    print(f"  Expected: [0, 1, 2, ...] (hammers first)")


if __name__ == "__main__":
    test_spectral_ranking()
