"""
CCRR (Cross-Consistency Robust Ranking) Algorithm.

Implements the algorithm from the analysis report:
  Phase 1: Row-norm spammer detection via agreement matrix G
  Phase 2: EM refinement (Bradley-Terry MLE + type re-estimation)
  Phase 3: Selection of best answer

References:
  - Karger, Oh, Shah (2011): Budget-optimal crowdsourcing
  - Bradley & Terry (1952): Paired comparison model
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Any


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    result = np.empty_like(x, dtype=float)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-sigmoid: log(sigma(x))."""
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x, dtype=float)
    pos = x >= 0
    result[pos] = -np.log1p(np.exp(-x[pos]))
    result[~pos] = x[~pos] - np.log1p(np.exp(x[~pos]))
    return result


class CCRRanking:
    """
    Cross-Consistency Robust Ranking algorithm.

    Algorithm (from the analysis report):
      Phase 1: Compute agreement matrix G = (1/(N-2)) * M @ M^T (M = R with zeroed diagonal).
               Compute row-norm rho_i = sum_k G_ik^2.
               Hammers have rho_i = Theta(N); spammers have rho_i = O(1).
               Threshold to initialize hammer weights.
      Phase 2: EM refinement alternating:
               M-step: Bradley-Terry MLE for scores using current weights.
               E-step: Update weights via likelihood ratio (hammer vs spammer).
      Phase 3: Select i* = argmax_i s_hat_i.
    """

    def __init__(self, beta: float = 5.0, epsilon: float = 0.1, T: int = 100):
        """
        Args:
            beta: Bradley-Terry discrimination parameter (recommended: 3-5)
            epsilon: Prior probability of being a spammer
            T: Number of EM iterations
        """
        self.beta = beta
        self.epsilon = epsilon
        self.T = T

        # Diagnostics
        self.last_weights = None
        self.last_scores = None
        self.last_rho = None

    def select_best(self, R: np.ndarray, return_details: bool = False):
        """
        Select the best answer from comparison matrix R.

        Args:
            R: Comparison matrix (N, N) with values in {-1, 1}, R_ii = 1.
            return_details: If True, return (index, scores, weights).

        Returns:
            Best answer index (or tuple if return_details=True).
        """
        best_idx, s_hat, w = self._run(R)
        if return_details:
            return best_idx, s_hat, w
        return best_idx

    # ================================================================
    # Core algorithm
    # ================================================================

    def _run(self, R: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        N = R.shape[0]
        assert R.shape == (N, N)
        assert np.all(np.diag(R) == 1)

        # ---- Phase 1: Row-norm spammer detection ----
        w = self._phase1_rownorm_detection(R, N)

        # ---- Phase 2: EM refinement ----
        s_hat = self._init_scores(R, w, N)

        for t in range(self.T):
            # M-step: estimate scores via Bradley-Terry MLE
            s_hat = self._bt_mle(R, w, s_hat, N)

            # E-step: update weights via likelihood ratio
            w = self._update_weights(R, s_hat, N)

        # ---- Phase 3: Selection ----
        best_idx = int(np.argmax(s_hat))

        self.last_scores = s_hat
        self.last_weights = w

        return best_idx, s_hat, w

    # ================================================================
    # Phase 1: Row-norm spammer detection
    # ================================================================

    def _phase1_rownorm_detection(self, R: np.ndarray, N: int) -> np.ndarray:
        """
        Compute agreement matrix G and row-norms rho_i.

        G_ik = (1/(N-2)) * sum_{j != i, j != k} R_ij * R_kj
             = (1/(N-2)) * [M @ M^T]_ik  (approximately, after zeroing i,k contributions)

        For efficiency, compute G = (1/(N-2)) * M @ M^T, then zero diagonal.
        Row-norm: rho_i = sum_{k != i} G_ik^2.

        Hammers: rho_i = Theta(N).
        Spammers: rho_i = O(1).
        """
        # M = R with zeroed diagonal
        M = R.astype(float).copy()
        np.fill_diagonal(M, 0.0)

        # G = (1/(N-2)) * M @ M^T
        G = M @ M.T / (N - 2)
        np.fill_diagonal(G, 0.0)

        # Row-norms
        rho = np.sum(G ** 2, axis=1)
        self.last_rho = rho

        # Adaptive threshold: median / 2
        # Hammers have rho = Theta(N), spammers have rho = O(1).
        # If majority are hammers (epsilon < 0.5), median is among hammers,
        # so median/2 separates well.
        tau = np.median(rho) / 2.0

        # Initialize weights: 1 for hammer, 0 for spammer
        w = np.where(rho > tau, 1.0, 0.0)

        return w

    # ================================================================
    # Phase 2: Bradley-Terry MLE (M-step)
    # ================================================================

    def _init_scores(self, R: np.ndarray, w: np.ndarray, N: int) -> np.ndarray:
        """Initialize scores using weighted row sums."""
        M = R.astype(float).copy()
        np.fill_diagonal(M, 0.0)

        # Weighted row sum: weight by agent reliability
        row_sums = M @ w  # sum_j R_ij * w_j for each i
        s_init = row_sums / (np.sum(w) + 1e-10)

        return s_init

    def _bt_mle(self, R: np.ndarray, w: np.ndarray, s_init: np.ndarray, N: int) -> np.ndarray:
        """
        Bradley-Terry MLE for score estimation.

        Maximize:
          L(s) = sum_i w_i * sum_{j != i} [
              1(R_ij=1) * log sigma(beta*(s_i - s_j))
            + 1(R_ij=-1) * log(1 - sigma(beta*(s_i - s_j)))
          ]

        subject to sum(s) = 0 (identifiability).
        """
        beta = self.beta

        # Precompute indicators
        M = R.astype(float).copy()
        np.fill_diagonal(M, 0.0)
        # R_plus[i,j] = 1 if R_ij = 1 and i != j, else 0
        R_plus = ((M + 1) / 2).clip(0, 1)   # 1 where R_ij = 1
        R_minus = ((1 - M) / 2).clip(0, 1)   # 1 where R_ij = -1

        def neg_log_likelihood(s):
            # Enforce sum = 0
            s = s - np.mean(s)
            diff = beta * (s[:, None] - s[None, :])  # (N, N)

            log_sig_pos = log_sigmoid(diff)    # log sigma(beta(s_i - s_j))
            log_sig_neg = log_sigmoid(-diff)   # log(1 - sigma(beta(s_i - s_j)))

            # Weighted log-likelihood
            ll_per_agent = np.sum(R_plus * log_sig_pos + R_minus * log_sig_neg, axis=1)
            ll = np.sum(w * ll_per_agent)

            return -ll

        def neg_gradient(s):
            s = s - np.mean(s)
            diff = beta * (s[:, None] - s[None, :])
            sig = sigmoid(diff)

            # Gradient: d/ds_i of log-lik contribution from row i
            # = beta * sum_j [R_plus_ij * (1 - sig_ij) - R_minus_ij * sig_ij]
            # = beta * sum_j [R_plus_ij - sig_ij] (for j != i)
            residual = R_plus - sig  # (N, N)
            np.fill_diagonal(residual, 0.0)

            # Row contribution (agent i as judge)
            grad_row = beta * np.sum(residual, axis=1)

            # Column contribution (agent i as subject being judged)
            grad_col = -beta * np.sum((w[:, None] * residual), axis=0)

            grad = w * grad_row + grad_col

            # Project out mean (identifiability constraint)
            grad = grad - np.mean(grad)

            return -grad

        # Optimize
        result = minimize(
            neg_log_likelihood,
            s_init,
            jac=neg_gradient,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-8}
        )

        s_hat = result.x
        s_hat = s_hat - np.mean(s_hat)  # enforce identifiability

        return s_hat

    # ================================================================
    # Phase 2: Weight update (E-step)
    # ================================================================

    def _update_weights(self, R: np.ndarray, s_hat: np.ndarray, N: int) -> np.ndarray:
        """
        E-step: compute P(z_i = H | R, s_hat) for each agent i.

        L_i^H = prod_{j != i} sigma(beta*(s_i - s_j))^{1(R_ij=1)}
                               * (1-sigma(...))^{1(R_ij=-1)}
        L_i^S = 2^{-(N-1)}

        w_i = (1-epsilon)*L_i^H / ((1-epsilon)*L_i^H + epsilon*L_i^S)
        """
        beta = self.beta
        diff = beta * (s_hat[:, None] - s_hat[None, :])  # (N, N)

        # log L_i^H = sum_{j != i} log sigma(R_ij * beta * (s_i - s_j))
        M = R.astype(float).copy()
        np.fill_diagonal(M, 0.0)

        log_lik = log_sigmoid(M * diff)
        np.fill_diagonal(log_lik, 0.0)
        L_H_log = np.sum(log_lik, axis=1)  # (N,)

        # log L_i^S = (N-1) * log(0.5)
        L_S_log = (N - 1) * np.log(0.5)

        # log posterior
        log_pH = np.log(1.0 - self.epsilon) + L_H_log
        log_pS = np.log(self.epsilon) + L_S_log

        # Numerically stable softmax
        max_log = np.maximum(log_pH, log_pS)
        w = np.exp(log_pH - max_log) / (np.exp(log_pH - max_log) + np.exp(log_pS - max_log))

        # Handle NaN
        w = np.where(np.isnan(w), 0.5, w)

        return w

    # ================================================================
    # Utilities
    # ================================================================

    def rank_all_answers(self, R: np.ndarray) -> np.ndarray:
        """Return indices sorted by quality (best first)."""
        _, s_hat, _ = self._run(R)
        return np.argsort(s_hat)[::-1]

    def identify_hammers_spammers(self, R: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Classify agents into hammers and spammers."""
        _, s_hat, w = self._run(R)
        return {
            'hammers': np.where(w >= threshold)[0],
            'spammers': np.where(w < threshold)[0],
            'weights': w,
            'scores': s_hat,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics from last run."""
        if self.last_scores is None:
            return {'error': 'No computation performed yet'}
        return {
            'algorithm': 'CCRR (Row-Norm + BT-MLE)',
            'num_agents': len(self.last_scores),
            'beta': self.beta,
            'epsilon': self.epsilon,
            'iterations': self.T,
            'scores': self.last_scores.tolist(),
            'weights': self.last_weights.tolist(),
            'rho': self.last_rho.tolist() if self.last_rho is not None else None,
            'num_hammers': int(np.sum(self.last_weights >= 0.5)),
            'num_spammers': int(np.sum(self.last_weights < 0.5)),
        }

    def __repr__(self) -> str:
        return f"CCRRanking(beta={self.beta}, epsilon={self.epsilon}, T={self.T})"


# ================================================================
# Test
# ================================================================

if __name__ == "__main__":
    print("Testing CCRR Algorithm (Row-Norm + BT-MLE)...")
    print("=" * 60)

    N = 15
    np.random.seed(42)
    beta_true = 5.0
    epsilon_true = 0.2  # 20% spammers

    # Generate true scores
    s_true = np.random.uniform(0, 1, size=N)

    # Generate types
    z = np.array(['S' if np.random.random() < epsilon_true else 'H' for _ in range(N)])

    # Generate comparison matrix
    R = np.ones((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if z[i] == 'H':
                prob = 1.0 / (1.0 + np.exp(-beta_true * (s_true[i] - s_true[j])))
                R[i, j] = 1 if np.random.random() < prob else -1
            else:
                R[i, j] = 1 if np.random.random() < 0.5 else -1

    true_best = np.argmax(s_true)
    true_hammers = set(np.where(z == 'H')[0])
    true_spammers = set(np.where(z == 'S')[0])

    print(f"N = {N}, epsilon = {epsilon_true}")
    print(f"True hammers: {sorted(true_hammers)}")
    print(f"True spammers: {sorted(true_spammers)}")
    print(f"True best agent: {true_best} (score: {s_true[true_best]:.4f})")

    # ---- CCRR ----
    print(f"\n{'='*60}")
    print("CCRR Algorithm")
    print(f"{'='*60}")
    ccrr = CCRRanking(beta=5.0, epsilon=0.1, T=5)
    best_idx, scores, weights = ccrr.select_best(R, return_details=True)

    print(f"Selected: Agent {best_idx} (true score: {s_true[best_idx]:.4f})")
    print(f"True best: Agent {true_best} (true score: {s_true[true_best]:.4f})")
    print(f"Regret: {s_true[true_best] - s_true[best_idx]:.4f}")

    # Hammer/spammer detection
    detected_hammers = set(np.where(weights >= 0.5)[0])
    detected_spammers = set(np.where(weights < 0.5)[0])
    print(f"\nDetected hammers: {sorted(detected_hammers)}")
    print(f"Detected spammers: {sorted(detected_spammers)}")
    print(f"Hammer precision: {len(detected_hammers & true_hammers) / max(len(detected_hammers), 1):.2%}")
    print(f"Hammer recall: {len(detected_hammers & true_hammers) / max(len(true_hammers), 1):.2%}")

    # Row-norms
    print(f"\nRow-norms (rho):")
    rho = ccrr.last_rho
    for i in range(N):
        marker = "H" if z[i] == 'H' else "S"
        det = "H" if weights[i] >= 0.5 else "S"
        print(f"  Agent {i:2d} [{marker}→{det}]: rho={rho[i]:8.2f}, w={weights[i]:.4f}, s_hat={scores[i]:.4f}, s_true={s_true[i]:.4f}")

    # Score correlation
    corr = np.corrcoef(scores, s_true)[0, 1]
    print(f"\nScore correlation (estimated vs true): {corr:.4f}")

    # ---- Majority Voting Baseline ----
    print(f"\n{'='*60}")
    print("Majority Voting Baseline")
    print(f"{'='*60}")
    col_sums = np.sum(R, axis=0)  # how many agents rate answer j as worse
    row_sums = np.sum(R, axis=1) - 1  # how many answers agent i rates as worse
    # Simple: use row_sums as score proxy
    mv_best = int(np.argmax(row_sums))
    print(f"Selected: Agent {mv_best} (true score: {s_true[mv_best]:.4f})")
    print(f"Regret: {s_true[true_best] - s_true[mv_best]:.4f}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"CCRR regret:  {s_true[true_best] - s_true[best_idx]:.4f}")
    print(f"MV regret:    {s_true[true_best] - s_true[mv_best]:.4f}")
    print(f"CCRR selected correct best: {best_idx == true_best}")
    print(f"MV selected correct best:   {mv_best == true_best}")