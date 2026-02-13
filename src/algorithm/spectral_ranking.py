"""
Spectral Ranking Algorithm for Hammer-Spammer Model

Based on Karger's mechanism (Theorem II.3), adapted to self-comparison setting.

═══════════════════════════════════════════════════════════════════════════
COMPLETE LOGICAL CHAIN: Why SVD Beats Majority Voting
═══════════════════════════════════════════════════════════════════════════

Karger's Key Discovery:
  Majority voting: Δ_MV ~ 1/(q²) log(1/ε)    [treats all judges equally]
  Spectral method: Δ_spectral ~ 1/q log(1/ε)  [weights by reliability]
  → Quadratic improvement in quality gap q

The Mechanism (Karger):
  In bipartite setting (m tasks × n workers):
  - E[A] has structure where spammer COLUMNS are zero vectors
  - SVD's RIGHT singular vector v has v_j = 0 for spammer workers
  - Weighting by v removes spammer noise → lower variance → better accuracy

═══════════════════════════════════════════════════════════════════════════
OUR SETTING: Same Mechanism, Transposed Roles
═══════════════════════════════════════════════════════════════════════════

Step 1: Structural Property (Exact, No Approximation)
  In our N×N self-comparison matrix R̃ (diagonal removed):
  - Hammer i: E[R̃_ij] = tanh(β(s_i - s_j)/2)  [nonzero, informative]
  - Spammer i: E[R̃_ij] = 0  [R_ij ∈ {±1} with equal probability]

  ⟹ Spammer ROWS in E[R̃] are zero vectors (exact from model definition)

Step 2: SVD Consequence (Linear Algebra Fact)
  If row i of matrix M is zero, then for any eigenvector x with λ ≠ 0:
    Mx = λx ⟹ (Mx)_i = 0 = λx_i ⟹ x_i = 0

  ⟹ In E[R̃]'s LEFT singular vectors, spammer positions have value exactly 0
  ⟹ By perturbation bound (‖Z‖₂ = O(√N), σ₁(E[R̃]) = Ω(√N)):
      Actual u₁ converges to E[R̃]'s u₁, so spammer weights w_i = u₁ᵢ² → 0

Step 3: Variance Comparison (Why SVD Beats MV)
  For score estimation of agent j:
  - MV: spammer variance contribution = ε/N > 0 (spammers vote equally)
  - SVD: spammer variance contribution → 0 (spammer weights → 0)

  ⟹ Same signal (expectation), strictly lower noise (variance)
  ⟹ Higher probability of selecting correct answer
  ⟹ SVD > MV (exact inequality for ε > 0)

═══════════════════════════════════════════════════════════════════════════
KEY DIFFERENCE FROM KARGER: Transposed Structure
═══════════════════════════════════════════════════════════════════════════

Karger (bipartite m×n):
  - Spammer COLUMNS are zero
  - Use RIGHT singular vector v for worker reliability
  - Use LEFT singular vector u for task answers

Ours (self-comparison N×N):
  - Spammer ROWS are zero
  - Use LEFT singular vector u for agent reliability  ← TRANSPOSED!
  - Use weighted column aggregation for answer scores

The mechanism is identical, just transposed. This is not a heuristic—it's
the direct mathematical consequence of "spammer contribution = 0 in expectation."
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

        Implements Karger's mechanism adapted to self-comparison setting.

        Complete algorithm (rigorous derivation in module docstring):
          1. Remove diagonal: R̃ = R with R̃_ii = 0
          2. SVD: Extract top LEFT singular vector u₁
          3. Weights: w_i = u₁ᵢ² (spammer weights → 0)
          4. Scores: score_j = -Σᵢ w_i · R̃_ij (weighted column aggregation)
          5. Select: argmax_j score_j

        Why LEFT singular vector (not RIGHT like Karger)?
          - Spammer ROWS are zero in E[R̃] (not columns)
          - u₁ (left) identifies judge reliability (transposed from Karger)
          - This is exact from model definition, not a heuristic

        Args:
            R: N×N comparison matrix with values in {-1, +1}
            return_details: If True, return (index, scores, weights, details)
                          If False, return only index

        Returns:
            If return_details=False: best answer index
            If return_details=True: (best_idx, scores, weights, details_dict)
        """
        N = R.shape[0]

        # ═════════════════════════════════════════════════════════════════
        # Step 1: Remove diagonal (R_ii = 1 is constant, no information)
        # ═════════════════════════════════════════════════════════════════
        R_tilde = R.copy().astype(float)
        np.fill_diagonal(R_tilde, 0.0)

        # ═════════════════════════════════════════════════════════════════
        # Step 2: SVD to identify reliable judges
        # ═════════════════════════════════════════════════════════════════
        # R̃ = U Σ V^T
        # - U: left singular vectors (ROW patterns → judge reliability)
        # - V: right singular vectors (COLUMN patterns → answer quality)
        U, s, Vt = np.linalg.svd(R_tilde, full_matrices=False)

        # u₁: LEFT singular vector (judge reliability)
        # In E[R̃], spammer rows are zero ⟹ u₁[spammer] = 0
        u1 = U[:, 0]

        # ═════════════════════════════════════════════════════════════════
        # Step 3: Reliability weights from u₁
        # ═════════════════════════════════════════════════════════════════
        # Square to remove sign ambiguity (|u₁| = -|u₁| are equivalent)
        # Result: hammers get high weight, spammers get weight ≈ 0
        weights = u1 ** 2

        # Normalize to sum to N (makes weights interpretable as "effective vote count")
        weights = weights * (N / weights.sum())

        # Store for diagnostics
        self.weights = weights
        self.singular_values = s

        # ═════════════════════════════════════════════════════════════════
        # Step 4: Weighted score aggregation
        # ═════════════════════════════════════════════════════════════════
        # score_j = -Σᵢ w_i · R̃_ij
        #
        # Interpretation:
        #   - R̃_ij = +1: judge i prefers themselves over j → decreases score_j
        #   - R̃_ij = -1: judge i prefers j over themselves → increases score_j
        #   - Negative sign converts to: higher score = better quality
        #
        # Key property:
        #   - Hammers (w_i high) dominate the sum
        #   - Spammers (w_i → 0) contribute negligible noise
        #   - Variance: O(1/q) vs O(1/q²) for majority voting
        scores = -weights @ R_tilde  # Equivalent to: -np.sum(weights * R_tilde, axis=0)

        # ═════════════════════════════════════════════════════════════════
        # Step 5: Select best answer
        # ═════════════════════════════════════════════════════════════════
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
