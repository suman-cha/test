"""
SWRA (Spectral Weighted Rank Aggregation) Algorithm.

This module implements the SWRA v2 algorithm for answer selection
under the hammer-spammer model.

Based on: swra_v2_algorithm.py and ALGORITHM_DESIGN_DETAILED.md
"""

import numpy as np
from scipy.linalg import svd
from typing import Tuple


class SWRARanking:
    """
    SWRA v2: Spectral Weighted Rank Aggregation.

    Key features:
    - Uses BOTH row-based (self-assessment) and column-based (peer-assessment) signals
    - Spectral reliability estimation filters lucky spammers
    - Combined scoring preserves the joint quality-reliability structure
    """

    def __init__(self, k: int = 2, max_iter: int = 15, tol: float = 1e-6,
                 version: str = "iterative"):
        """
        Initialize SWRA algorithm.

        Args:
            k: Number of singular vectors to use
            max_iter: Maximum iterations for iterative version
            tol: Convergence tolerance
            version: "iterative" or "simple"
        """
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.version = version

    def select_best(self, R: np.ndarray, return_details: bool = False) -> int:
        """
        Select the best answer from comparison matrix.

        Args:
            R: Comparison matrix (N, N) with values in {-1, 1}
            return_details: If True, return (index, scores, weights), else just index

        Returns:
            Index of best answer (or tuple if return_details=True)
        """
        if self.version == "iterative":
            best_idx, scores, weights = self._swra_v2_iterative(R)
        else:
            best_idx, scores, weights = self._swra_v2_simple(R)

        if return_details:
            return best_idx, scores, weights
        else:
            return best_idx

    def _swra_v2_iterative(self, R: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        SWRA v2 Iterative: Full version with iterative refinement.

        Returns:
            Tuple of (best_idx, scores, weights)
        """
        N = R.shape[0]
        R_tilde = R.astype(np.float64).copy()
        np.fill_diagonal(R_tilde, 0.0)

        # === Phase 1: Spectral Reliability Estimation ===
        k_use = min(self.k, N - 1)
        U_full, S_full, Vt_full = svd(R_tilde, full_matrices=False)
        U = U_full[:, :k_use]
        S = S_full[:k_use]
        V = Vt_full[:k_use, :].T

        # Judge reliability from left singular vectors
        reliability = np.sum(U**2, axis=1)
        reliability = reliability / (reliability.sum() + 1e-12)

        # === Phase 2: Three scoring signals ===

        # Signal 1: Self-assessment (row sum)
        row_scores = R_tilde.sum(axis=1)

        # Signal 2: Peer-assessment (weighted column sum)
        peer_scores = -R_tilde.T @ reliability

        # Signal 3: Reliability-gated self-assessment
        gated_scores = reliability * row_scores

        # === Phase 3: Iterative Combined Scoring ===
        # Initialize scores as combination of all signals
        scores = np.zeros(N)
        for signal in [row_scores, peer_scores, gated_scores]:
            s_norm = signal - signal.mean()
            s_std = s_norm.std()
            if s_std > 1e-12:
                s_norm = s_norm / s_std
            scores += s_norm

        # Iterative refinement
        weights = reliability.copy()
        for t in range(self.max_iter):
            old_weights = weights.copy()

            # Update reliability: check consistency
            score_diff = scores[:, None] - scores[None, :]
            consistency = np.sum(R_tilde * np.sign(score_diff + 1e-12), axis=1) / (N - 1)
            weights = np.maximum(consistency, 0)
            w_sum = weights.sum()
            if w_sum > 1e-12:
                weights = weights / w_sum
            else:
                weights = np.ones(N) / N

            # Update scores
            gated = weights * R_tilde.sum(axis=1)
            peer = -R_tilde.T @ weights

            # Normalize and combine
            gated_norm = (gated - gated.mean())
            gated_std = gated_norm.std()
            if gated_std > 1e-12:
                gated_norm = gated_norm / gated_std

            peer_norm = (peer - peer.mean())
            peer_std = peer_norm.std()
            if peer_std > 1e-12:
                peer_norm = peer_norm / peer_std

            scores = gated_norm + peer_norm

            if np.linalg.norm(weights - old_weights) < self.tol:
                break

        best_idx = int(np.argmax(scores))
        return best_idx, scores, weights

    def _swra_v2_simple(self, R: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        SWRA v2 Simple: Non-iterative version.

        Returns:
            Tuple of (best_idx, scores, reliability)
        """
        N = R.shape[0]
        R_tilde = R.astype(np.float64).copy()
        np.fill_diagonal(R_tilde, 0.0)

        # Spectral reliability from left singular vectors
        k_use = min(self.k, N - 1)
        U_full, S_full, _ = svd(R_tilde, full_matrices=False)
        U = U_full[:, :k_use]
        reliability = np.sum(U**2, axis=1)
        reliability = reliability / (reliability.sum() + 1e-12)

        # Gated self-assessment
        row_scores = R_tilde.sum(axis=1)
        gated_scores = reliability * row_scores

        # Weighted peer assessment
        peer_scores = -R_tilde.T @ reliability

        # Combine (z-score normalization then sum)
        g_std = gated_scores.std()
        p_std = peer_scores.std()

        scores = np.zeros(N)
        if g_std > 1e-12:
            scores += (gated_scores - gated_scores.mean()) / g_std
        if p_std > 1e-12:
            scores += (peer_scores - peer_scores.mean()) / p_std

        best_idx = int(np.argmax(scores))
        return best_idx, scores, reliability

    def rank_all_answers(self, R: np.ndarray) -> np.ndarray:
        """
        Rank all answers from best to worst.

        Args:
            R: Comparison matrix (N, N)

        Returns:
            Array of ranks (0 = best)
        """
        _, scores, _ = self.select_best(R, return_details=True)

        # Higher score = better rank
        # argsort returns indices in ascending order, so reverse
        ranked_indices = np.argsort(-scores)

        # Create rank array
        ranks = np.empty(len(scores), dtype=int)
        ranks[ranked_indices] = np.arange(len(scores))

        return ranks

    def identify_hammers_spammers(self, R: np.ndarray,
                                  threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify hammer and spammer agents.

        Args:
            R: Comparison matrix (N, N)
            threshold: Weight threshold for hammer classification

        Returns:
            Tuple of (hammer_indices, spammer_indices)
        """
        _, _, weights = self.select_best(R, return_details=True)

        hammers = np.where(weights > threshold)[0]
        spammers = np.where(weights <= threshold)[0]

        return hammers, spammers
