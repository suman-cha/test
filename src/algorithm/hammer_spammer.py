"""
Hammer-Spammer Model Algorithm (Problem 1).

Spectral Ranking under Hammer-Spammer Model.

핵심 직관:
- Row i는 에이전트 i가 "심사위원"으로서 생산한 비교 결과.
  Hammer의 row는 실제 점수 차이를 반영하는 의미 있는 신호,
  Spammer의 row는 순수 노이즈.
- SVD의 top left singular vector u₁은 "가장 많은 행들이 공유하는 패턴"을 잡아냄.
  → Hammer에 대응하는 u₁ 원소는 크고, Spammer는 0 근처.
- u₁²을 가중치로 사용하여 column 집계를 하면 spammer를 자동으로 걸러냄.

Algorithm:
1. 대각선 제거: R̃ ← R with diagonal zeroed out
2. SVD로 신뢰도 추정: R̃ ≈ σ₁·u₁·v₁ᵀ, wᵢ = (u₁ᵢ)²
3. 가중 점수 산출: score_j = -Σ_{i≠j} wᵢ · R̃ᵢⱼ
4. 최고 답 선택: argmax_j score_j
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.linalg import svd


class HammerSpammerRanking:
    """
    SVD-based Spectral Ranking for Hammer-Spammer model.

    SVD를 통해 각 에이전트(행 생산자)의 신뢰도를 자동으로 추정하고,
    신뢰도를 가중치로 써서 column 집계를 함으로써 spammer를 걸러냄.
    """

    def __init__(self, beta: float = 5.0, epsilon: float = 0.1):
        """
        Initialize Hammer-Spammer ranking algorithm.

        Args:
            beta: Bradley-Terry sigmoid sharpness parameter (hyperparameter)
            epsilon: Prior probability of an agent being a spammer
        """
        self.beta = beta
        self.epsilon = epsilon

        # Diagnostics from last computation
        self.last_singular_values = None
        self.last_quality_scores = None
        self.last_weights = None
        self.last_u1 = None
        self.last_v1 = None

    def select_best(self, R: np.ndarray, return_scores: bool = False):
        """
        Select the best answer from comparison matrix.

        Args:
            R: Comparison matrix (N, N) with values in {-1, 1}
            return_scores: If True, return (index, scores), else just index

        Returns:
            Index of best answer (or tuple of (index, scores) if return_scores=True)
        """
        scores = self.compute_scores(R)
        best_idx = int(np.argmax(scores))

        if return_scores:
            return best_idx, scores
        return best_idx

    def compute_scores(self, R: np.ndarray) -> np.ndarray:
        """
        Compute quality scores for all answers using the spectral method.

        Step 1: 대각선 제거 — Rᵢᵢ = 1은 모델 정의상 무조건 1이므로,
                포함하면 SVD가 trivial한 신호에 끌려감.
        Step 2: SVD로 u₁ 추출 → wᵢ = (u₁ᵢ)² 로 신뢰도 가중치 계산.
        Step 3: score_j = -Σ_{i≠j} wᵢ · R̃ᵢⱼ
                (R̃ᵢⱼ = -1이 많은 column j가 좋은 답 → 부호 반전)

        Args:
            R: Comparison matrix (N, N) with values in {-1, 1}

        Returns:
            Score array (N,) — higher is better
        """
        N = R.shape[0]
        assert R.shape == (N, N), f"R must be square, got {R.shape}"

        # ─── Step 1: Preprocessing — 대각선 제거 ───
        R_tilde = R.astype(float).copy()
        np.fill_diagonal(R_tilde, 0.0)

        # ─── Step 2: SVD로 신뢰도 추정 ───
        U, S, Vh = svd(R_tilde, full_matrices=False)

        self.last_singular_values = S

        # Top-1 singular vectors
        u1 = U[:, 0]      # left singular vector (row 구조 = 각 에이전트의 심사 패턴)
        v1 = Vh[0, :]      # right singular vector (column 구조 = 각 답의 평가받은 패턴)

        self.last_u1 = u1
        self.last_v1 = v1

        # 신뢰도 가중치: wᵢ = (u₁ᵢ)²
        # 제곱해서 부호 무관하게 크기만 봄.
        # Hammer → |u₁ᵢ| 큼 → wᵢ 큼
        # Spammer → u₁ᵢ ≈ 0 → wᵢ ≈ 0 → 자동 무시
        weights = u1 ** 2

        self.last_weights = weights

        # ─── Step 3: 가중 점수 산출 ───
        # score_j = -Σ_{i≠j} wᵢ · R̃ᵢⱼ
        #
        # R̃ᵢⱼ = -1 → "에이전트 i가 답 j를 자기보다 낫다고 판단"
        # 좋은 답 j는 column에 -1이 많음 → column 가중합이 음수
        # 부호를 뒤집어서 "높을수록 좋다"로 만듦
        #
        # Majority voting은 wᵢ = 1 (모두 동일 가중치),
        # 이 알고리즘은 wᵢ가 SVD에서 나온 신뢰도.
        scores = np.zeros(N)
        for j in range(N):
            weighted_sum = 0.0
            for i in range(N):
                if i != j:
                    weighted_sum += weights[i] * R_tilde[i, j]
            scores[j] = -weighted_sum

        self.last_quality_scores = scores
        return scores

    def compute_scores_vectorized(self, R: np.ndarray) -> np.ndarray:
        """
        Vectorized version of compute_scores for performance.
        Equivalent to compute_scores but uses matrix operations.

        Args:
            R: Comparison matrix (N, N) with values in {-1, 1}

        Returns:
            Score array (N,) — higher is better
        """
        N = R.shape[0]
        assert R.shape == (N, N), f"R must be square, got {R.shape}"

        # Step 1: 대각선 제거
        R_tilde = R.astype(float).copy()
        np.fill_diagonal(R_tilde, 0.0)

        # Step 2: SVD → 신뢰도 가중치
        U, S, Vh = svd(R_tilde, full_matrices=False)
        self.last_singular_values = S

        u1 = U[:, 0]
        v1 = Vh[0, :]
        self.last_u1 = u1
        self.last_v1 = v1

        weights = u1 ** 2
        self.last_weights = weights

        # Step 3: score_j = -Σ_{i≠j} wᵢ · R̃ᵢⱼ  (vectorized)
        # = -(w ᵀ R̃)_j  (since R̃ diagonal is 0, i≠j condition is auto-satisfied)
        scores = -(weights @ R_tilde)

        self.last_quality_scores = scores
        return scores

    def rank_all_answers(self, R: np.ndarray) -> np.ndarray:
        """
        Rank all answers from best to worst.

        Args:
            R: Comparison matrix (N, N)

        Returns:
            Indices sorted by score (best first)
        """
        scores = self.compute_scores_vectorized(R)
        return np.argsort(scores)[::-1]

    def identify_hammers_spammers(
        self, R: np.ndarray, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Identify hammers and spammers based on SVD weight.

        Uses wᵢ = (u₁ᵢ)² as reliability indicator.
        Hammer → wᵢ large, Spammer → wᵢ ≈ 0.

        Args:
            R: Comparison matrix (N, N)
            threshold: Weight threshold (default: median of weights)

        Returns:
            Dictionary with hammers, spammers, weights, quality_scores, threshold
        """
        scores = self.compute_scores_vectorized(R)
        weights = self.last_weights

        if threshold is None:
            threshold = float(np.median(weights))

        hammers = np.where(weights >= threshold)[0]
        spammers = np.where(weights < threshold)[0]

        return {
            'hammers': hammers,
            'spammers': spammers,
            'weights': weights,
            'quality_scores': scores,
            'threshold': threshold,
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
            'rank_1_ratio': (float(self.last_singular_values[0] / self.last_singular_values.sum())
                            if self.last_singular_values is not None else None),
            'weights': self.last_weights.tolist() if self.last_weights is not None else None,
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
        print(f"Spectral Ranking (Hammer-Spammer) Diagnostics")
        print(f"{'='*60}\n")

        print(f"Number of agents: {diag['num_agents']}")
        print(f"\nSingular values (top 5): {diag['singular_values'][:5]}")
        print(f"Rank-1 approximation ratio: {diag['rank_1_ratio']:.4f}")

        print(f"\nSVD weights (reliability):")
        weights = diag['weights']
        if weights is not None:
            for i, w in enumerate(weights):
                bar = '#' * int(w / max(weights) * 30)
                print(f"  Agent {i:2d}: w={w:.4f} {bar}")

        print(f"\nQuality scores:")
        print(f"  Mean: {diag['quality_mean']:.4f}")
        print(f"  Std:  {diag['quality_std']:.4f}")
        print(f"  Range: [{diag['quality_min']:.4f}, {diag['quality_max']:.4f}]")

        print(f"\nTop 5 answers by score:")
        sorted_indices = np.argsort(self.last_quality_scores)[::-1][:5]
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"  {rank}. Agent {idx}: score={self.last_quality_scores[idx]:.4f}  "
                  f"weight={self.last_weights[idx]:.4f}")

    def __repr__(self) -> str:
        return f"HammerSpammerRanking(beta={self.beta}, epsilon={self.epsilon})"


# ─────────────────────────────────────────────────────────
# Majority Voting Baseline (comparison matrix version)
# ─────────────────────────────────────────────────────────

def majority_voting_matrix(R: np.ndarray) -> int:
    """
    Majority Voting Baseline on comparison matrix.

    vote_j = -Σ_{i≠j} Rᵢⱼ  (단순 column 합, 부호 반전)
    return argmax_j vote_j

    문제점: spammer의 투표도 동일하게 셈 → 노이즈에 취약.

    Args:
        R: Comparison matrix (N, N) with values in {-1, 1}

    Returns:
        Index of best answer
    """
    R_tilde = R.astype(float).copy()
    np.fill_diagonal(R_tilde, 0.0)
    votes = -R_tilde.sum(axis=0)  # -Σᵢ R̃ᵢⱼ for each j
    return int(np.argmax(votes))


# ─────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────

def simulate_comparison_matrix(
    N: int, beta: float = 5.0, epsilon: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a comparison matrix following the Hammer-Spammer model.

    Model:
        - s_i ~ U(0, 1) for each agent i
        - z_i ∈ {H, S} with P(z_i = S) = epsilon
        - Hammer (z_i = H): P(R_ij = 1 | s, z_i = H) = σ(β(s_i - s_j))
        - Spammer (z_i = S): R_ij ~ uniform({-1, 1})
        - R_ii = 1 for all i

    Args:
        N: Number of agents
        beta: Bradley-Terry discrimination parameter
        epsilon: Probability of being a spammer

    Returns:
        Tuple of (comparison_matrix, true_scores, agent_types)
        where agent_types[i] = 1 for hammer, 0 for spammer
    """
    scores = np.random.uniform(0, 1, N)
    agent_types = (np.random.random(N) >= epsilon).astype(int)  # 1=hammer, 0=spammer

    R = np.ones((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            if i == j:
                R[i, j] = 1
            elif agent_types[i] == 1:
                # Hammer: P(R_ij = 1) = σ(β(s_i - s_j))
                prob = 1.0 / (1.0 + np.exp(-beta * (scores[i] - scores[j])))
                R[i, j] = 1 if np.random.random() < prob else -1
            else:
                # Spammer: R_ij ~ uniform({-1, 1})
                R[i, j] = 1 if np.random.random() < 0.5 else -1

    return R, scores, agent_types


if __name__ == "__main__":
    print("Testing Spectral Ranking (Hammer-Spammer) Algorithm")
    print("=" * 60)

    np.random.seed(42)

    # ─── Simulate ───
    N = 15
    beta = 5.0
    epsilon = 0.3  # 30% spammer rate

    R, true_scores, agent_types = simulate_comparison_matrix(N, beta=beta, epsilon=epsilon)

    num_hammers = np.sum(agent_types == 1)
    num_spammers = np.sum(agent_types == 0)
    print(f"\nGenerated {N} agents: {num_hammers} hammers, {num_spammers} spammers")
    print(f"True best agent: {np.argmax(true_scores)} (score: {true_scores[np.argmax(true_scores)]:.4f})")

    # ─── Spectral Ranking ───
    print(f"\n{'─'*60}")
    print("Spectral Ranking (SVD-weighted)")
    print(f"{'─'*60}")
    algo = HammerSpammerRanking(beta=beta, epsilon=epsilon)
    best_idx, scores = algo.select_best(R, return_scores=True)

    true_best = np.argmax(true_scores)
    print(f"Selected: Agent {best_idx} (score={scores[best_idx]:.4f}, true_score={true_scores[best_idx]:.4f})")
    print(f"True best: Agent {true_best} (true_score={true_scores[true_best]:.4f})")
    print(f"Correct: {best_idx == true_best}")

    # Verify loop vs vectorized give same result
    scores_loop = algo.compute_scores(R)
    scores_vec = algo.compute_scores_vectorized(R)
    assert np.allclose(scores_loop, scores_vec), "Loop and vectorized scores differ!"
    print(f"Loop/vectorized consistency check: PASSED")

    # ─── Majority Voting Baseline ───
    print(f"\n{'─'*60}")
    print("Majority Voting Baseline (uniform weights)")
    print(f"{'─'*60}")
    mv_best = majority_voting_matrix(R)
    print(f"Selected: Agent {mv_best} (true_score={true_scores[mv_best]:.4f})")
    print(f"Correct: {mv_best == true_best}")

    # ─── Hammer/Spammer Detection ───
    print(f"\n{'─'*60}")
    print("Hammer/Spammer Detection")
    print(f"{'─'*60}")
    classification = algo.identify_hammers_spammers(R)

    print(f"\nDetected hammers: {classification['hammers']}")
    print(f"Detected spammers: {classification['spammers']}")
    print(f"\nTrue hammers:  {np.where(agent_types == 1)[0]}")
    print(f"True spammers: {np.where(agent_types == 0)[0]}")

    # Detection accuracy
    detected_types = np.zeros(N)
    detected_types[classification['hammers']] = 1
    accuracy = np.mean(detected_types == agent_types)
    print(f"\nDetection accuracy: {accuracy:.1%}")

    # ─── Full Diagnostics ───
    algo.print_diagnostics()

    # ─── Correlation with true scores ───
    print(f"\n{'─'*60}")
    print("Correlation Analysis")
    print(f"{'─'*60}")
    corr = np.corrcoef(scores, true_scores)[0, 1]
    print(f"Score vs true_score correlation: {corr:.4f}")

    # ─── Monte Carlo: Spectral vs Majority Voting ───
    print(f"\n{'─'*60}")
    print("Monte Carlo Comparison (100 trials)")
    print(f"{'─'*60}")

    n_trials = 100
    spectral_correct = 0
    mv_correct = 0

    for _ in range(n_trials):
        R_mc, s_mc, _ = simulate_comparison_matrix(N, beta=beta, epsilon=epsilon)
        true_best_mc = np.argmax(s_mc)

        algo_mc = HammerSpammerRanking(beta=beta, epsilon=epsilon)
        if algo_mc.select_best(R_mc) == true_best_mc:
            spectral_correct += 1
        if majority_voting_matrix(R_mc) == true_best_mc:
            mv_correct += 1

    print(f"Spectral Ranking accuracy: {spectral_correct}/{n_trials} ({spectral_correct}%)")
    print(f"Majority Voting accuracy:  {mv_correct}/{n_trials} ({mv_correct}%)")
    print(f"Improvement: {spectral_correct - mv_correct} percentage points")
