"""
Bias-Corrected CCRR: Account for self-preference bias α.

Modified Bradley-Terry model:
  P(R_ij = 1 | s, z_i = H) = σ(β(s_i - s_j) + α_i)

where α_i is agent i's self-preference bias (estimated from data).
"""

import numpy as np
from typing import Tuple, Dict
from scipy.special import expit as sigmoid


class BiasCorrectedCCRR:
    """
    CCRR with bias correction for anti-self-preference.

    Estimates α_i for each agent from row sums, then adjusts
    the Bradley-Terry likelihood accordingly.
    """

    def __init__(self, beta: float = 5.0, epsilon: float = 0.1, T: int = 5):
        """
        Initialize bias-corrected CCRR.

        Args:
            beta: Bradley-Terry discrimination
            epsilon: Spammer prior
            T: EM iterations
        """
        self.beta = beta
        self.epsilon = epsilon
        self.T = T

        self.last_alpha = None
        self.last_weights = None
        self.last_scores = None

    def estimate_bias(self, R: np.ndarray) -> np.ndarray:
        """
        Estimate self-preference bias α_i for each agent.

        Uses row sums as a proxy:
          row_sum_i = (1/(N-1)) Σ_j R_ij
          E[row_sum_i] ≈ σ(α_i)  (when scores are balanced)

        Args:
            R: Comparison matrix

        Returns:
            Array of α estimates
        """
        N = R.shape[0]

        # Convert {-1, 1} to {0, 1}
        R_01 = (R + 1) / 2

        # Row sums (excluding diagonal)
        row_sums = np.zeros(N)
        for i in range(N):
            row_sums[i] = (R_01[i, :].sum() - R_01[i, i]) / (N - 1)

        # Estimate α from logit
        # Clip to avoid infinities
        row_sums = np.clip(row_sums, 0.01, 0.99)
        alpha = np.log(row_sums / (1 - row_sums))

        return alpha

    def select_best(self, R: np.ndarray, return_details: bool = False):
        """
        Select best answer with bias correction.

        Args:
            R: Comparison matrix
            return_details: Return (idx, scores, weights, alpha)

        Returns:
            Best index (or tuple if return_details)
        """
        N = R.shape[0]

        # Estimate bias
        alpha = self.estimate_bias(R)
        self.last_alpha = alpha

        # Correct R by subtracting bias
        # R_corrected[i,j] ≈ sign(β(s_i - s_j)) if we remove α_i
        R_corrected = R.copy()

        # For each row i, adjust by removing its bias
        # This is approximate: we shift the logit by -α_i
        # In practice, we can't directly "remove" α from {-1,1} values,
        # so we use it in the EM algorithm instead

        # Initialize weights (hammer detection)
        # Use variance in row sums: low variance → uniform bias → likely hammers
        row_sum_variance = np.var(alpha)

        if row_sum_variance < 0.2:
            # Uniform bias → trust everyone initially
            w = np.ones(N) * (1 - self.epsilon)
        else:
            # Varied bias → use row-norm detection
            from src.algorithm.ccrr import CCRRanking
            temp_ccrr = CCRRanking(self.beta, self.epsilon, 1)
            _, _, w = temp_ccrr.select_best(R, return_details=True)

        # Modified EM with bias correction
        R_float = R.astype(float)

        # Initialize scores (bias-corrected row sums)
        M = R_float.copy()
        np.fill_diagonal(M, 0.0)

        # Adjust row sums by subtracting estimated bias effect
        # row_sum_i ≈ Σ_j σ(β(s_i - s_j) + α_i)
        # For small α, this is approximately: row_sum + (N-1)·α_i/β
        # So: s_i ∝ (row_sum_i - (N-1)·α_i/β)

        raw_row_sums = M.sum(axis=1)
        bias_correction = (N - 1) * alpha / self.beta
        corrected_row_sums = raw_row_sums - bias_correction

        # Initialize scores from corrected row sums
        s_hat = corrected_row_sums
        s_std = np.std(s_hat)
        if s_std > 1e-6:
            s_hat = (s_hat - s_hat.mean()) / s_std

        # EM iterations (same as regular CCRR but with bias awareness)
        for t in range(self.T):
            # E-step: Update weights (same as before)
            diff_matrix = self.beta * (s_hat[:, None] - s_hat[None, :])

            # Here we COULD incorporate α, but it's complex
            # For simplicity, keep standard E-step
            # (The main correction was in initialization)

            # Log-likelihood under hammer model
            log_sig = np.log(sigmoid(R_float * diff_matrix) + 1e-12)
            np.fill_diagonal(log_sig, 0.0)
            L_H = log_sig.sum(axis=1)
            L_H = np.maximum(L_H, -500.0)

            L_S = (N - 1) * np.log(0.5)

            log_pH = np.log(1 - self.epsilon) + L_H
            log_pS = np.log(self.epsilon) + L_S

            max_log = np.maximum(log_pH, log_pS)
            w = np.exp(log_pH - max_log) / (
                np.exp(log_pH - max_log) + np.exp(log_pS - max_log)
            )
            w = np.where(np.isnan(w), 0.5, w)

            # M-step: Update scores (bias-corrected)
            weighted_row = w * M.sum(axis=1)
            weighted_col = M.T @ w

            # Apply bias correction to weighted row sums
            bias_term = w * bias_correction
            s_hat = weighted_row - bias_term - weighted_col

            # Normalize
            s_std = np.std(s_hat)
            if s_std > 1e-6:
                s_hat = (s_hat - s_hat.mean()) / s_std

        # Select best
        best_idx = int(np.argmax(s_hat))

        self.last_weights = w
        self.last_scores = s_hat

        if return_details:
            return best_idx, s_hat, w, alpha
        else:
            return best_idx

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        if self.last_scores is None:
            return {'error': 'No computation performed yet'}

        return {
            'algorithm': 'BiasCorrectedCCRR',
            'beta': self.beta,
            'epsilon': self.epsilon,
            'iterations': self.T,
            'scores': self.last_scores.tolist(),
            'weights': self.last_weights.tolist(),
            'alpha_estimates': self.last_alpha.tolist() if self.last_alpha is not None else None,
            'mean_alpha': float(np.mean(self.last_alpha)) if self.last_alpha is not None else None,
            'alpha_std': float(np.std(self.last_alpha)) if self.last_alpha is not None else None,
        }


if __name__ == "__main__":
    print("Testing Bias-Corrected CCRR...")

    # Generate test data WITH bias
    np.random.seed(42)
    N = 10
    beta = 5.0

    # True scores
    s_true = np.random.uniform(0, 1, N)

    # Agent-specific biases
    alpha = np.random.uniform(-0.5, 0.2, N)  # Mostly anti-self-preference

    # Generate R with bias
    R = np.ones((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            logit = beta * (s_true[i] - s_true[j]) + alpha[i]
            prob = 1.0 / (1.0 + np.exp(-logit))
            R[i, j] = 1 if np.random.random() < prob else -1

    print(f"\nGenerated data:")
    print(f"  True scores: {s_true}")
    print(f"  True alphas: {alpha}")
    print(f"  Mean alpha: {alpha.mean():.3f}")

    # Apply bias-corrected CCRR
    bc_ccrr = BiasCorrectedCCRR(beta=beta, epsilon=0.1, T=5)
    best_idx, scores, weights, alpha_est = bc_ccrr.select_best(R, return_details=True)

    print(f"\nBias-Corrected CCRR results:")
    print(f"  Selected: agent {best_idx} (true score: {s_true[best_idx]:.3f})")
    print(f"  True best: agent {np.argmax(s_true)} (score: {s_true.max():.3f})")
    print(f"\nEstimated alphas: {alpha_est}")
    print(f"True alphas:      {alpha}")
    print(f"Correlation: {np.corrcoef(alpha_est, alpha)[0,1]:.3f}")

    # Compare with regular CCRR
    from src.algorithm.ccrr import CCRRanking
    regular_ccrr = CCRRanking(beta=beta, epsilon=0.1, T=5)
    regular_idx = regular_ccrr.select_best(R)

    print(f"\nComparison:")
    print(f"  Bias-Corrected: Selected agent {best_idx} (correct: {best_idx == np.argmax(s_true)})")
    print(f"  Regular CCRR:   Selected agent {regular_idx} (correct: {regular_idx == np.argmax(s_true)})")
