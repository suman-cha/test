"""
Spectral Ranking Algorithm for Hammer-Spammer Model (Karger's Theorem II.3)

Uses SVD to identify reliable judges and weight their comparisons.

Karger's Theorem II.3 shows the fundamental gap:
- Majority voting: Δ_MV ~ 1/(q²) log(1/ε)    [treats all judges equally]
- Spectral method: Δ_spectral ~ 1/q log(1/ε)  [weights by reliability]
  → Quadratic improvement in quality gap q

Key mechanism:
1. Right singular vector v₁ of comparison matrix R identifies judge reliability
2. Hammers (reliable judges) get high weight |v₁|, spammers get low weight
3. Weighted aggregation uses only reliable judges' comparisons
4. This separation produces dramatic improvement over Borda count

Direct analog in our setting:
- v₁ identifies which agents are reliable judges (hammer vs spammer)
- Use v₁-weighted comparisons to score each agent's answer quality
- High reliability judges dominate the final ranking
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
        Select the best answer using spectral ranking (Karger's Theorem II.3).

        Key mechanism (per Karger):
        1. Right singular vector v₁ identifies judge reliability (hammers vs spammers)
        2. Weight comparisons by judge reliability (not treating all judges equally)
        3. This achieves Δ ~ 1/q (vs Δ ~ 1/q² for majority voting)
           → Quadratic improvement in quality gap q

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

        # Step 2: SVD to identify reliable judges
        # R = U Σ V^T
        # Right singular vector v₁ captures judge reliability patterns
        U, s, Vt = np.linalg.svd(R_tilde, full_matrices=False)

        # v₁: RIGHT singular vector (judge reliability)
        # Vt is V^T, so v₁ = Vt[0, :]
        v1 = Vt[0, :]

        # Judge reliability weights from v₁
        # Square to get positive weights (magnitude matters, not sign)
        weights = v1 ** 2

        # Normalize to sum to N (for interpretability)
        weights = weights * (N / weights.sum())

        # Store for diagnostics
        self.weights = weights
        self.singular_values = s

        # Step 3: Weighted aggregation using reliable judges
        # score_j = -Σᵢ wᵢ · R[i,j]
        #   = -sum of reliable judges' comparisons about agent j
        # R[i,j] = -1 means judge i thinks j is better than themselves
        # So negative sign makes higher score = better quality
        scores = -weights @ R_tilde  # Matrix multiplication: weights^T @ R_tilde

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
