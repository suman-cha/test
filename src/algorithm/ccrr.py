"""
CCRR (Cross-Consistency Robust Ranking) Algorithm.

This module implements the CCRR algorithm as specified in CCRR_Algorithm_Spec.md.
CCRR uses cross-consistency analysis and EM-based iterative refinement to identify
hammers (high-quality agents) and spammers (low-quality agents), then selects the
best answer.

Algorithm Phases:
1. Cross-Consistency Spammer Detection: Compute cross-consistency scores
2. Weighted Score Estimation: Initial score estimation with weights
3. Iterative Refinement: EM algorithm to refine weights and scores
4. Final Selection: Select answer with highest score
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    Args:
        x: Input array

    Returns:
        Sigmoid of x
    """
    x = np.asarray(x)
    pos = x >= 0
    result = np.empty_like(x, dtype=float)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable log-sigmoid function.

    Args:
        x: Input array

    Returns:
        Log of sigmoid of x
    """
    x = np.asarray(x)
    result = np.empty_like(x, dtype=float)
    pos = x >= 0
    result[pos] = -np.log1p(np.exp(-x[pos]))
    result[~pos] = x[~pos] - np.log1p(np.exp(x[~pos]))
    return result


def mad(arr: np.ndarray) -> float:
    """
    Median Absolute Deviation.

    Args:
        arr: Input array

    Returns:
        MAD of array
    """
    med = np.median(arr)
    return np.median(np.abs(arr - med))


class CCRRanking:
    """
    Cross-Consistency Robust Ranking algorithm.

    This algorithm uses cross-consistency analysis to detect spammers
    and EM-based iterative refinement to estimate agent quality scores.
    """

    def __init__(self, beta: float = 5.0, epsilon: float = 0.1, T: int = 5):
        """
        Initialize CCRR algorithm.

        Args:
            beta: Bradley-Terry discrimination parameter (default: 5.0)
            epsilon: Prior probability of spammer (default: 0.1)
            T: Number of iterative refinement rounds (default: 5)
        """
        self.beta = beta
        self.epsilon = epsilon
        self.T = T

        # Statistics (for diagnostics)
        self.last_weights = None
        self.last_scores = None
        self.last_cross_consistency = None

    def select_best(self, R: np.ndarray, return_details: bool = False) -> int:
        """
        Select the best answer from comparison matrix.

        Args:
            R: Comparison matrix (N, N) with values in {-1, 1}
            return_details: If True, return (index, scores, weights), else just index

        Returns:
            Index of best answer (or tuple if return_details=True)
        """
        best_idx, s_hat, w = self._ccrr(R)

        if return_details:
            return best_idx, s_hat, w
        else:
            return best_idx

    def _ccrr(self, R: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Core CCRR algorithm implementation.

        Args:
            R: Comparison matrix (N, N)

        Returns:
            Tuple of (best_idx, scores, weights)
        """
        N = R.shape[0]

        # Verify matrix properties
        assert R.shape == (N, N), f"R must be square, got {R.shape}"
        assert np.all(np.isin(R, [-1, 1])), "R must contain only ±1"
        assert np.all(np.diag(R) == 1), "Diagonal must be all 1s"

        # ===== Phase 1: Cross-Consistency Spammer Detection =====
        # C[i] = sum_{j != i} R[i][j] * R[j][i]
        cross = R * R.T  # element-wise product
        C = np.sum(cross, axis=1) - np.diag(cross)  # subtract diagonal

        self.last_cross_consistency = C

        C_med = np.median(C)
        C_mad = mad(C)
        if C_mad == 0:
            C_mad = 1.0
        gamma = 2.0 / C_mad

        w = sigmoid(-gamma * (C - C_med))

        # ===== Phase 2: Initial Weighted Score Estimation =====
        # row_sum[i] = sum_{j != i} R[i][j]
        row_sums = np.sum(R, axis=1) - 1.0  # subtract diagonal R[i][i] = 1

        # col_sum[i] = sum_{j != i} w[j] * R[j][i]
        col_sums = R.T @ w - w  # R.T @ w - w[i]*R[i][i]

        s_hat = w * row_sums - col_sums

        # Normalize
        s_std = np.std(s_hat)
        if s_std == 0:
            s_std = 1.0
        s_hat = (s_hat - np.mean(s_hat)) / s_std

        # ===== Phase 3: Iterative Refinement =====
        for t in range(self.T):

            # --- 3a. Update weights (E-step) ---
            # Compute beta * (s_hat[i] - s_hat[j]) for all i, j
            diff_matrix = self.beta * (s_hat[:, None] - s_hat[None, :])  # shape (N, N)

            # log_sigmoid(R[i][j] * diff_matrix[i][j])
            log_sig_values = log_sigmoid(R * diff_matrix)

            # Sum over j != i (exclude diagonal)
            np.fill_diagonal(log_sig_values, 0.0)
            L_H = np.sum(log_sig_values, axis=1)

            # Clamp L_H to avoid -inf issues
            L_H = np.maximum(L_H, -500.0)

            L_S = (N - 1) * np.log(0.5)

            log_pH = np.log(1 - self.epsilon) + L_H
            log_pS = np.log(self.epsilon) + L_S

            # Numerically stable softmax
            max_log = np.maximum(log_pH, log_pS)
            w = np.exp(log_pH - max_log) / (np.exp(log_pH - max_log) + np.exp(log_pS - max_log))

            # Handle NaN (if both log_pH and log_pS are -inf)
            w = np.where(np.isnan(w), 0.5, w)

            # --- 3b. Update scores (M-step) ---
            row_sums = np.sum(R, axis=1) - 1.0
            col_sums = R.T @ w - w
            s_hat = w * row_sums - col_sums

            # --- 3c. Normalize scores ---
            s_std = np.std(s_hat)
            if s_std == 0:
                s_std = 1.0
            s_hat = (s_hat - np.mean(s_hat)) / s_std

        # ===== Phase 4: Selection =====
        best_idx = int(np.argmax(s_hat))

        # Store for diagnostics
        self.last_scores = s_hat
        self.last_weights = w

        return best_idx, s_hat, w

    def rank_all_answers(self, R: np.ndarray) -> np.ndarray:
        """
        Rank all answers from best to worst.

        Args:
            R: Comparison matrix (N, N)

        Returns:
            Indices sorted by quality (best first)
        """
        _, s_hat, _ = self._ccrr(R)
        return np.argsort(s_hat)[::-1]  # Descending order

    def identify_hammers_spammers(self, R: np.ndarray, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Identify hammers (high-quality agents) and spammers (low-quality agents).

        Args:
            R: Comparison matrix (N, N)
            threshold: Weight threshold for hammer classification (default: 0.5)

        Returns:
            Dictionary with:
                - hammers: indices of high-quality agents (w >= threshold)
                - spammers: indices of low-quality agents (w < threshold)
                - weights: all hammer weights
                - scores: all quality scores
        """
        _, s_hat, w = self._ccrr(R)

        hammers = np.where(w >= threshold)[0]
        spammers = np.where(w < threshold)[0]

        return {
            'hammers': hammers,
            'spammers': spammers,
            'weights': w,
            'scores': s_hat,
            'threshold': threshold
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the last computation.

        Returns:
            Dictionary with diagnostic information
        """
        if self.last_scores is None:
            return {'error': 'No computation performed yet'}

        N = len(self.last_scores)

        return {
            'algorithm': 'CCRR',
            'num_agents': N,
            'beta': self.beta,
            'epsilon': self.epsilon,
            'iterations': self.T,
            'scores': self.last_scores.tolist(),
            'weights': self.last_weights.tolist(),
            'cross_consistency': self.last_cross_consistency.tolist() if self.last_cross_consistency is not None else None,
            'num_hammers': int(np.sum(self.last_weights >= 0.5)),
            'num_spammers': int(np.sum(self.last_weights < 0.5)),
            'score_mean': float(np.mean(self.last_scores)),
            'score_std': float(np.std(self.last_scores)),
            'weight_mean': float(np.mean(self.last_weights)),
            'weight_std': float(np.std(self.last_weights)),
        }

    def print_diagnostics(self):
        """Print diagnostic information."""
        diag = self.get_diagnostics()

        if 'error' in diag:
            print(f"Error: {diag['error']}")
            return

        print(f"\n{'='*60}")
        print(f"CCRR Algorithm Diagnostics")
        print(f"{'='*60}\n")

        print(f"Number of agents: {diag['num_agents']}")
        print(f"Parameters: beta={diag['beta']}, epsilon={diag['epsilon']}, T={diag['iterations']}")

        print(f"\nHammer/Spammer Classification:")
        print(f"  Hammers: {diag['num_hammers']} (weight >= 0.5)")
        print(f"  Spammers: {diag['num_spammers']} (weight < 0.5)")

        print(f"\nWeights:")
        print(f"  Mean: {diag['weight_mean']:.4f}")
        print(f"  Std: {diag['weight_std']:.4f}")

        print(f"\nScores:")
        print(f"  Mean: {diag['score_mean']:.4f}")
        print(f"  Std: {diag['score_std']:.4f}")

        print(f"\nTop 5 agents by score:")
        sorted_indices = np.argsort(self.last_scores)[::-1][:5]
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"  {rank}. Agent {idx}: score={self.last_scores[idx]:.4f}, weight={self.last_weights[idx]:.4f}")

    def __repr__(self) -> str:
        """String representation."""
        return f"CCRRanking(beta={self.beta}, epsilon={self.epsilon}, T={self.T})"


if __name__ == "__main__":
    # Test the CCRR algorithm
    print("Testing CCRR Algorithm...")

    # Generate a simple test case
    print("\n=== Generating Test Data ===")
    N = 10
    np.random.seed(42)

    # Generate true scores
    s_true = np.random.uniform(0, 1, size=N)

    # Generate types (70% hammers, 30% spammers)
    z = np.where(np.random.random(N) < 0.3, 'S', 'H')

    # Generate comparison matrix
    R = np.ones((N, N), dtype=int)
    beta = 5.0

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if z[i] == 'H':
                prob = 1.0 / (1.0 + np.exp(-beta * (s_true[i] - s_true[j])))
                R[i, j] = 1 if np.random.random() < prob else -1
            else:
                R[i, j] = 1 if np.random.random() < 0.5 else -1

    print(f"Generated {N} agents")
    print(f"True hammers: {np.sum(z == 'H')}, Spammers: {np.sum(z == 'S')}")
    print(f"True best agent: {np.argmax(s_true)} (score: {s_true[np.argmax(s_true)]:.4f})")

    # Apply CCRR
    print("\n=== Applying CCRR Algorithm ===")
    ccrr = CCRRanking(beta=5.0, epsilon=0.1, T=5)

    best_idx, scores, weights = ccrr.select_best(R, return_details=True)

    print(f"\nCCRR selected: Agent {best_idx}")
    print(f"  Estimated score: {scores[best_idx]:.4f}")
    print(f"  Hammer weight: {weights[best_idx]:.4f}")
    print(f"  True score: {s_true[best_idx]:.4f}")
    print(f"  True type: {z[best_idx]}")

    # Classification
    classification = ccrr.identify_hammers_spammers(R)
    print(f"\n=== Hammer/Spammer Classification ===")
    print(f"Detected hammers: {len(classification['hammers'])} (true: {np.sum(z == 'H')})")
    print(f"Detected spammers: {len(classification['spammers'])} (true: {np.sum(z == 'S')})")

    # Check accuracy
    true_hammers = np.where(z == 'H')[0]
    detected_hammers = classification['hammers']
    precision = len(np.intersect1d(detected_hammers, true_hammers)) / len(detected_hammers) if len(detected_hammers) > 0 else 0
    recall = len(np.intersect1d(detected_hammers, true_hammers)) / len(true_hammers) if len(true_hammers) > 0 else 0

    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")

    # Diagnostics
    ccrr.print_diagnostics()

    # Compare scores
    print(f"\n=== Score Correlation ===")
    print(f"Correlation with true scores: {np.corrcoef(scores, s_true)[0,1]:.4f}")
