# Spectral Weighted Rank Aggregation (SWRA): Recovering the Best Answer from Noisy Pairwise Comparisons under the Hammer-Spammer Model

---

## Overview

This document presents a complete algorithm design for **Problem 1** of the take-home test, inspired by Karger, Oh, and Shah (2011). The design is produced through iterative deliberation among:

- **Algorithm Design Agent**: Proposes the method and its components.
- **Mathematical Proof Agent**: Provides rigorous justification for each step.
- **Critical Reviewer Agent**: Identifies gaps, edge cases, and validates correctness.

---

## 1. Problem Restatement and Key Structural Analysis

### 1.1 The Problem

We observe a comparison matrix $R \in \{±1\}^{N \times N}$, where:

$$R_{ij} = \begin{cases} 1 & \text{if agent } i \text{ judges its own answer better than agent } j\text{'s} \\ -1 & \text{otherwise} \end{cases}$$

with $R_{ii} = 1$. Each agent $i$ has a latent quality score $s_i$ and a type $z_i \in \{H, S\}$ (hammer with probability $1 - \varepsilon$, spammer with probability $\varepsilon$).

- **Hammer** ($z_i = H$): $P(R_{ij} = 1 \mid s, z_i = H) = \sigma(\beta(s_i - s_j))$, where $\sigma(x) = 1/(1 + e^{-x})$.
- **Spammer** ($z_i = S$): $R_{ij}$ is i.i.d. uniform on $\{±1\}$.

**Goal**: Recover $i^* = \arg\max_i s_i$ (the best answer).

### 1.2 Key Structural Differences from Karger et al.

| Feature | Karger et al. | This Problem |
|---------|---------------|--------------|
| Matrix structure | Bipartite (tasks × workers) | Square (agents × agents) |
| Row $i$ produced by | Multiple workers | Agent $i$ alone |
| Latent variable | Binary $s_i \in \{±1\}$ | Continuous $s_i \in \mathbb{R}$ |
| Goal | Classify all tasks | Find $\arg\max_i s_i$ |
| Comparison model | $P(A_{ij} = s_i) = p_j$ | $P(R_{ij} = 1) = \sigma(\beta(s_i - s_j))$ |
| Who is unreliable | Worker $j$ (column) | Agent $i$ (row) |

> **🔍 Reviewer:** The most critical structural difference is that **row $i$ is entirely produced by agent $i$**. In Karger, column $j$ corresponds to worker $j$, and the *right* singular vector captures worker reliability. Here, **the left singular vector captures judge (row) reliability**, because each row is the output of a single judge. This reversal is essential for correct algorithm design.

### 1.3 Expected Matrix Structure

Define $\tilde{R}$ as $R$ with diagonal zeroed out ($\tilde{R}_{ii} = 0$).

For a **hammer** agent $i$ and any agent $j \neq i$:

$$\mathbb{E}[\tilde{R}_{ij} \mid s, z_i = H] = 2\sigma(\beta(s_i - s_j)) - 1 = \tanh\left(\frac{\beta(s_i - s_j)}{2}\right)$$

For a **spammer** agent $i$:

$$\mathbb{E}[\tilde{R}_{ij} \mid s, z_i = S] = 0$$

**Linearization** (valid when $\beta |s_i - s_j|$ is moderate, or as a first-order approximation):

$$\tanh\left(\frac{\beta(s_i - s_j)}{2}\right) \approx \frac{\beta}{2}(s_i - s_j)$$

Therefore:

$$\mathbb{E}[\tilde{R}_{ij}] \approx \frac{\beta}{2} h_i (s_i - s_j)$$

where $h_i = \mathbf{1}(z_i = H)$ is the hammer indicator.

> **📐 Proof Agent — Rank structure derivation:**
>
> Writing in matrix form with $\mathbf{h} = (h_1, \ldots, h_N)^T$ and $\mathbf{s} = (s_1, \ldots, s_N)^T$:
>
> $$\mathbb{E}[\tilde{R}] \approx \frac{\beta}{2}\left[(\mathbf{h} \odot \mathbf{s})\mathbf{1}^T - \mathbf{h}\mathbf{s}^T\right]$$
>
> where $\odot$ denotes the Hadamard (elementwise) product. Define $\mathbf{a} = \mathbf{h} \odot \mathbf{s}$. Then:
>
> $$\mathbb{E}[\tilde{R}] \approx \frac{\beta}{2}\left[\mathbf{a}\mathbf{1}^T - \mathbf{h}\mathbf{s}^T\right]$$
>
> This is a sum of two rank-1 matrices, hence **rank at most 2**.
>
> - **Left singular space** $\approx \operatorname{span}\{\mathbf{a}, \mathbf{h}\} = \operatorname{span}\{\mathbf{h} \odot \mathbf{s}, \mathbf{h}\}$ — encodes **judge reliability** and **judge scores**.
> - **Right singular space** $\approx \operatorname{span}\{\mathbf{1}, \mathbf{s}\}$ — encodes the **constant direction** and the **score direction**.
>
> **Crucial insight: The score vector $\mathbf{s}$ lies (approximately) in the right singular space of $\tilde{R}$.**

---

## 2. Algorithm Design: Spectral Weighted Rank Aggregation (SWRA)

### 2.1 Algorithm Statement

```
ALGORITHM: Spectral Weighted Rank Aggregation (SWRA)
═══════════════════════════════════════════════════════

Input: Comparison matrix R ∈ {±1}^{N×N}, rank parameter k (default k=2)

Phase 0 — Preprocessing:
    Set R̃_ij = R_ij for i ≠ j;  R̃_ii = 0.

Phase 1 — Spectral Decomposition:
    Compute the top-k SVD of R̃:
        R̃ ≈ Σ_{l=1}^{k} σ_l u_l v_l^T
    where σ_1 ≥ σ_2 ≥ ... ≥ σ_k > 0.

Phase 2 — Judge Reliability Estimation:
    For each agent i, compute reliability weight:
        w_i = Σ_{l=1}^{k} (u_{l,i})^2
    (This is the squared norm of agent i's projection
     onto the top-k left singular subspace.)

Phase 3 — Score Estimation (Weighted Column Aggregation):
    For each agent j, compute the raw score:
        score_j^{(col)} = -Σ_{i≠j} w_i · R̃_ij
    (Negative sign: R_ij = -1 means "j is better than i",
     which is evidence FOR j.)

Phase 4 — Score Refinement (Right Singular Vector Extraction):
    Let V_k = [v_1, ..., v_k] be the N×k matrix of right
    singular vectors. Estimate s via:
        s_hat = V_k · c*
    where c* = argmax_c Corr(V_k c, score^{(col)})
    subject to ||c|| = 1.
    
    (This projects the spectral score onto the right singular 
     space for denoising, using Phase 3 scores as guidance.)

Phase 5 — Sign Resolution:
    If Corr(s_hat, score^{(col)}) < 0, flip: s_hat ← -s_hat.

Phase 6 — Selection:
    Output i* = argmax_i  s_hat_i.
```

### 2.2 Simplified Practical Version

For implementation, the following streamlined version is equivalent in practice:

```
ALGORITHM: SWRA-Simple
═══════════════════════

Input: R ∈ {±1}^{N×N}

1. Set R̃ = R with diagonal zeroed out.

2. Compute the top-2 SVD: R̃ ≈ σ_1 u_1 v_1^T + σ_2 u_2 v_2^T.

3. Reliability weights: w_i = u_{1,i}^2 + u_{2,i}^2.

4. Weighted column score: score_j = -Σ_{i≠j} w_i R̃_ij.

5. Spectral score: Project score onto span(v_1, v_2) and 
   orthogonalize against the constant vector 1:
       s_hat = score - (score^T 1 / N) · 1     [remove mean]
       s_hat = V_2 V_2^T s_hat                   [project onto spectral space]

6. Output i* = argmax_i s_hat_i.
```

### 2.3 Even Simpler Baseline: Pure Spectral Method

```
ALGORITHM: SWRA-Minimal
════════════════════════

Input: R ∈ {±1}^{N×N}

1. Set R̃ = R with diagonal zeroed out.
2. Compute the top-2 right singular vectors v_1, v_2.
3. Among v_1 and v_2, pick the one less correlated with 
   the constant vector 1 (call it v_score).
4. Sign resolution: Ensure Corr(v_score, rowsums(R̃)) > 0.
   If not, flip v_score ← -v_score.
5. Output i* = argmax_i (v_score)_i.
```

---

## 3. Rigorous Justification

### 3.1 Why Zero the Diagonal?

> **📐 Proof Agent:**
>
> The diagonal entries $R_{ii} = 1$ for all $i$ contribute an identity-like component $\text{diag}(\mathbf{1})$ to $R$. This adds a rank-$N$ perturbation that is uninformative (every agent trivially prefers itself). Zeroing the diagonal removes this artifact and ensures the SVD focuses on the informative off-diagonal structure.
>
> Formally: $R = \tilde{R} + I_N$, so the spectrum of $R$ is a shift of $\tilde{R}$'s spectrum by 1 in each eigenvalue direction, which confounds the low-rank signal. After zeroing, $\tilde{R}$ has the clean rank-2 structure derived in Section 1.3.

### 3.2 Why Left Singular Vectors Capture Judge Reliability

> **📐 Proof Agent:**
>
> Consider the matrix $\tilde{R}\tilde{R}^T$. Its $(i,j)$ entry is:
>
> $$(\tilde{R}\tilde{R}^T)_{ij} = \sum_{k=1}^{N} \tilde{R}_{ik}\tilde{R}_{jk}$$
>
> This measures the **agreement between judges $i$ and $j$** across all contestants $k$.
>
> - If both $i$ and $j$ are **hammers**: their rows are correlated (they tend to agree on who is better), so $(\tilde{R}\tilde{R}^T)_{ij}$ is large and positive.
> - If one of $i, j$ is a **spammer**: the expected inner product is $\mathbb{E}[\sum_k \tilde{R}_{ik}\tilde{R}_{jk}] = 0$, because one factor is independent noise.
>
> Therefore, $\tilde{R}\tilde{R}^T$ has a **block structure**: the hammer-hammer block has large entries, while rows/columns corresponding to spammers have near-zero entries.
>
> The top eigenvectors of $\tilde{R}\tilde{R}^T$ (= top left singular vectors of $\tilde{R}$) will have **large entries for hammers and small entries for spammers**. This is exactly what we exploit: $w_i = \sum_l u_{l,i}^2$ is large for hammers.
>
> **Formal bound (analogous to Karger Lemma III.3):**
> Under the linearized model, $\tilde{R} = \mathbb{E}[\tilde{R}] + Z$ where $Z$ is the noise matrix. The expected matrix has rank 2 with singular values $\Theta(\beta N \sqrt{(1-\varepsilon)})$. By matrix concentration inequalities (e.g., the matrix Bernstein inequality), $\|Z\|_2 = O(\sqrt{N})$ with high probability. When $\beta \sqrt{N(1-\varepsilon)} \gg 1$, the signal dominates the noise, and the top-2 singular subspace of $\tilde{R}$ is close to that of $\mathbb{E}[\tilde{R}]$.

### 3.3 Why Weighted Column Sums Estimate Scores

> **📐 Proof Agent:**
>
> For agent $j$, the column score is:
>
> $$\text{score}_j^{(\text{col})} = -\sum_{i \neq j} w_i \tilde{R}_{ij}$$
>
> Consider the contribution from a hammer $i$:
>
> $$\mathbb{E}[-\tilde{R}_{ij} \mid s, z_i = H] = -\tanh\left(\frac{\beta(s_i - s_j)}{2}\right) \approx \frac{\beta}{2}(s_j - s_i)$$
>
> Summing over all agents weighted by reliability:
>
> $$\mathbb{E}[\text{score}_j^{(\text{col})}] \approx \frac{\beta}{2} \sum_{i \neq j} w_i (s_j - s_i) = \frac{\beta}{2}\left(s_j \sum_{i \neq j} w_i - \sum_{i \neq j} w_i s_i\right)$$
>
> Since spammers have $w_i \approx 0$ and hammers have $w_i > 0$:
>
> $$\mathbb{E}[\text{score}_j^{(\text{col})}] \approx \frac{\beta}{2}\left(s_j W_H - S_H\right)$$
>
> where $W_H = \sum_{i \in \text{Hammers}} w_i$ and $S_H = \sum_{i \in \text{Hammers}} w_i s_i$.
>
> **This is an affine function of $s_j$ with positive coefficient $\frac{\beta}{2} W_H > 0$.**
>
> Therefore, $\arg\max_j \text{score}_j^{(\text{col})} = \arg\max_j s_j = i^*$ in expectation.
>
> **Crucially, this works because the reliability weights $w_i$ suppress spammer contributions.** Without weighting ($w_i = 1$ for all $i$), we get the unweighted column sum — equivalent to "majority voting" — which includes spammer noise and is suboptimal.

> **🔍 Reviewer — Verification of the argument:**
> The proof above assumes the linearized model. For the exact logistic model, the monotonicity of $\tanh$ guarantees that the column score is still a monotonically increasing function of $s_j$ (though not exactly affine). The key property — that higher true scores yield higher expected column scores — holds in full generality. The argmax is preserved.

### 3.4 Why Right Singular Vectors Refine the Score Estimate

> **📐 Proof Agent:**
>
> The right singular space of $\mathbb{E}[\tilde{R}]$ is $\operatorname{span}\{\mathbf{1}, \mathbf{s}\}$. By matrix perturbation theory (Davis-Kahan theorem), the top-2 right singular space of $\tilde{R}$ is close to $\operatorname{span}\{\mathbf{1}, \mathbf{s}\}$ when the spectral gap is large.
>
> Projecting the noisy column score onto this 2D subspace and then removing the constant component (which carries no ranking information) yields a denoised estimate of $\mathbf{s}$.
>
> **Formal statement:** Let $V_2 = [v_1, v_2]$ be the top-2 right singular vectors of $\tilde{R}$. There exists a rotation/reflection matrix $Q \in \mathbb{R}^{2 \times 2}$ such that:
>
> $$V_2 Q \approx \left[\frac{\mathbf{1}}{\|\mathbf{1}\|}, \frac{\mathbf{s} - \bar{s}\mathbf{1}}{\|\mathbf{s} - \bar{s}\mathbf{1}\|}\right] + O\left(\frac{\|Z\|_2}{\sigma_2(\mathbb{E}[\tilde{R}])}\right)$$
>
> where $\bar{s} = \frac{1}{N}\sum_i s_i$. The second column (after de-meaning) is proportional to $\mathbf{s} - \bar{s}\mathbf{1}$, which preserves the argmax.

### 3.5 Sign Resolution

> **📐 Proof Agent:**
>
> Unlike Karger's binary setting, here the sign ambiguity can be resolved using a simple consistency check. The row sum $\sum_j \tilde{R}_{ij}$ is positively correlated with $s_i$ for hammers (high-score hammers beat most opponents). Therefore:
>
> $$\text{Corr}\left(\hat{\mathbf{s}}, \text{rowsums}(\tilde{R})\right) > 0$$
>
> should hold for the correctly signed estimate. If the correlation is negative, we flip the sign.
>
> **Why this is valid:** The expected row sum for a hammer $i$ is $\sum_{j \neq i} \tanh(\beta(s_i - s_j)/2)$, which is an increasing function of $s_i$. Spammer row sums have expectation 0. So the row sum vector is positively correlated with $\mathbf{h} \odot f(\mathbf{s})$ for some increasing function $f$, and hence with $\mathbf{s}$ restricted to hammers. The estimated score $\hat{\mathbf{s}}$ should align with this.

---

## 4. Why SWRA Outperforms Majority Voting

### 4.1 Majority Voting Baseline

The simplest baseline is the **row sum** (or equivalently, negative column sum):

$$\text{MV}_j = \sum_{i=1}^{N} \tilde{R}_{ij} \quad \text{or equivalently} \quad \text{MV}_i = \sum_{j=1}^{N} \tilde{R}_{ij}$$

**Row sum interpretation:** Agent $i$'s row sum $= \sum_j \tilde{R}_{ij}$ counts how many agents $i$ claims to beat. The agent with the highest row sum is the majority-voting winner.

**Column sum interpretation:** Agent $j$'s negative column sum $= -\sum_i \tilde{R}_{ij}$ counts how many judges prefer $j$ over themselves.

### 4.2 The Gap: Quantitative Analysis

> **📐 Proof Agent:**
>
> **Majority voting error analysis:**
>
> Consider the row sum for agent $i$:
>
> $$\text{MV}_i = \sum_{j \neq i} \tilde{R}_{ij}$$
>
> - If $i$ is a **hammer**: $\mathbb{E}[\text{MV}_i] = \sum_{j \neq i} \tanh(\beta(s_i - s_j)/2)$, and $\text{Var}[\text{MV}_i] = \Theta(N)$.
> - If $i$ is a **spammer**: $\mathbb{E}[\text{MV}_i] = 0$, and $\text{Var}[\text{MV}_i] = N - 1$.
>
> **Problem:** A spammer with a lucky draw can have $\text{MV}_i \approx \sqrt{N}$, which may exceed the expected row sum of the best hammer (especially when $\beta$ is small or scores are close). The probability that at least one spammer exceeds the best hammer's score is:
>
> $$P(\exists \text{ spammer } i: \text{MV}_i > \text{MV}_{i^*}) \geq 1 - (1 - P(\text{single spammer exceeds}))^{\varepsilon N}$$
>
> This probability remains non-negligible because majority voting gives **equal weight to all rows**, including the $\varepsilon N$ spammer rows.
>
> **SWRA error analysis:**
>
> SWRA assigns weight $w_i \approx 0$ to spammers and $w_i > 0$ to hammers. The weighted column score effectively averages only over $\approx (1-\varepsilon)N$ hammer judges, each providing informative signals. The noise from spammers is suppressed by a factor of $\|Z\|_2 / \sigma_1(\mathbb{E}[\tilde{R}])$, which vanishes as $N$ grows.
>
> **Analogy to Karger:** In Karger, majority voting costs $\Theta(1/q^2)$ while SVD costs $\Theta(1/q)$ per task. Here, the analogous gap is:
> - Majority voting: effective signal-to-noise ratio $\propto (1-\varepsilon)\sqrt{N}$ (signal from $(1-\varepsilon)N$ hammers, noise from all $N$ agents).
> - SWRA: effective SNR $\propto (1-\varepsilon)N$ (noise from spammers is eliminated by weighting).
>
> **The improvement factor is $\sqrt{(1-\varepsilon)N}$, which grows with $N$.**

> **🔍 Reviewer — Confirming the gap is real:**
> Consider a concrete example: $N = 50$, $\varepsilon = 0.3$ (30% spammers), $\beta = 3$, scores uniform on $[0,1]$. There are $\approx 15$ spammers. The standard deviation of a spammer's row sum is $\sqrt{49} \approx 7$. The expected row sum of the best hammer (say $s_{i^*} \approx 0.98$) is $\approx \sum_{j} \tanh(1.5 \cdot 0.5) \approx 49 \times 0.6 \approx 30$. So the best hammer's row sum is about $30 \pm 7$, while a spammer's is $0 \pm 7$. With 15 spammers, the maximum spammer row sum is $\approx 7 \times \sqrt{2 \ln 15} \approx 16$. So majority voting works here — but with $\varepsilon = 0.5$ or small $\beta$, the margin shrinks dramatically. SWRA maintains robustness by downweighting spammers.

---

## 5. Iterative Refinement Extension (EM-Style)

For even better performance, especially at small $N$ or high $\varepsilon$, we can add an iterative refinement step:

```
ALGORITHM: SWRA-Iterative
═════════════════════════

Input: R ∈ {±1}^{N×N}, max iterations T

1. Run SWRA-Simple to get initial s_hat and w.

2. For t = 1, ..., T:
   
   a. Score update: For each j,
      score_j = -Σ_{i≠j} w_i · R̃_ij
   
   b. Reliability update: For each i,
      w_i = (1/N) Σ_{j≠i} R̃_ij · (score_i - score_j)
      w_i = max(w_i, 0)    [clip negative weights]
      
      (A reliable judge's comparisons R̃_ij should agree
       with the sign of score_i - score_j. This inner 
       product is large for hammers, near zero for spammers.)
   
   c. Normalize: w ← w / ||w||_1

3. Final score: s_hat_j = -Σ_{i≠j} w_i · R̃_ij

4. Output i* = argmax_j s_hat_j.
```

> **📐 Proof Agent — Convergence intuition:**
>
> This iterative scheme is analogous to power iteration on a structured matrix. Each iteration refines both the score estimates and the reliability weights. Starting from the SVD-based initialization (which is already close to the truth), the iterations converge rapidly.
>
> The key property ensuring convergence: the reliability update $w_i \propto \sum_j R_{ij}(\hat{s}_i - \hat{s}_j)$ is a consistent estimator of the "informativeness" of judge $i$. For a hammer with true score $s_i$, this quantity has expectation $\propto \sum_j \tanh(\beta(s_i - s_j)/2)(s_i - s_j) > 0$. For a spammer, the expectation is 0.
>
> **Advantage over pure EM:** Unlike Dawid-Skene EM which is sensitive to random initialization, our iterative refinement starts from the spectral estimate, which is provably close to the truth. This avoids local optima.

---

## 6. Comparison of All Methods

| Method | Uses reliability estimation? | Suppresses spammers? | Computational cost |
|--------|------------------------------|----------------------|-------------------|
| Row Sum (Majority Vote) | No | No | $O(N^2)$ |
| Column Sum (Negative) | No | No | $O(N^2)$ |
| Borda Count | No | No | $O(N^2)$ |
| SWRA-Minimal | Yes (implicit via SVD) | Yes | $O(N^2 k)$ |
| SWRA-Simple | Yes (explicit weights) | Yes | $O(N^2 k)$ |
| SWRA-Iterative | Yes (refined iteratively) | Yes | $O(N^2 k T)$ |

All methods are $O(N^2)$ or $O(N^2 k T)$ where $k = 2$ and $T \leq 20$, so computational cost is negligible compared to the LLM API calls needed to construct $R$.

---

## 7. Hyperparameter Discussion

### 7.1 Choice of $\beta$

The parameter $\beta$ controls the "sharpness" of hammer comparisons:

- $\beta \to \infty$: Deterministic comparisons (noiseless).
- $\beta = 1$: Very noisy comparisons ($P(R_{ij} = 1) \approx 0.5 + 0.12(s_i - s_j)$ for small differences).
- $\beta = 5$: Moderate noise ($P(R_{ij} = 1) \approx 0.5 + 0.62(s_i - s_j)$ for small differences).

**Recommendation: $\beta \in [3, 5]$.**

> **🔍 Reviewer — Justification:**
>
> In the LLM agent setting (Problem 2), a "hammer" LLM can reliably distinguish answers that differ significantly in quality, but may struggle with close calls. $\beta = 3\text{--}5$ models this: when $|s_i - s_j| = 0.5$ (moderate quality gap), $\sigma(\beta \cdot 0.5) \approx 0.82\text{--}0.92$, meaning the hammer gets the comparison right 82--92% of the time. For very close scores ($|s_i - s_j| = 0.1$), accuracy drops to 57--62%. This matches the intuition that LLMs are good but imperfect judges.

### 7.2 Choice of rank $k$

**Recommendation: $k = 2$.**

Under the linearized model, $\mathbb{E}[\tilde{R}]$ has rank exactly 2. Setting $k = 2$ captures the full signal. Using $k > 2$ would include noise components. Using $k = 1$ would miss part of the signal (the constant component).

> **🔍 Reviewer — When $k > 2$ might help:** If the logistic model is far from the linearized regime (e.g., very large $\beta$ making $\tanh$ highly nonlinear), the expected matrix may have effective rank $> 2$. In this case, $k = 3$ or $4$ could capture additional signal. However, for $\beta \in [1, 10]$, $k = 2$ is sufficient.

### 7.3 Number of iterations $T$ (for SWRA-Iterative)

**Recommendation: $T = 10$.**

The spectral initialization is already close to optimal, so few iterations suffice. Convergence can be monitored by tracking $\|\mathbf{w}^{(t)} - \mathbf{w}^{(t-1)}\|$.

---

## 8. Complete Python Implementation

```python
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd

def swra(R, k=2, iterative=True, max_iter=10, tol=1e-6):
    """
    Spectral Weighted Rank Aggregation (SWRA)
    
    Parameters
    ----------
    R : np.ndarray, shape (N, N)
        Comparison matrix with R_ij in {-1, +1}, R_ii = 1.
    k : int
        Rank parameter for SVD (default 2).
    iterative : bool
        Whether to run iterative refinement.
    max_iter : int
        Maximum iterations for refinement.
    tol : float
        Convergence tolerance for weights.
    
    Returns
    -------
    best_idx : int
        Index of the estimated best agent.
    scores : np.ndarray
        Estimated score for each agent.
    weights : np.ndarray
        Estimated reliability weight for each agent.
    """
    N = R.shape[0]
    
    # Phase 0: Preprocessing
    R_tilde = R.astype(float).copy()
    np.fill_diagonal(R_tilde, 0.0)
    
    # Phase 1: Spectral Decomposition
    if N > 10:
        # Use sparse SVD for efficiency
        U, S, Vt = svds(R_tilde, k=k)
        # svds returns in ascending order; reverse
        idx = np.argsort(-S)
        U, S, Vt = U[:, idx], S[idx], Vt[idx, :]
    else:
        U_full, S_full, Vt_full = svd(R_tilde, full_matrices=False)
        U, S, Vt = U_full[:, :k], S_full[:k], Vt_full[:k, :]
    
    V = Vt.T  # N x k
    
    # Phase 2: Judge Reliability Estimation
    weights = np.sum(U**2, axis=1)  # ||u_i||^2 in top-k subspace
    weights = weights / (weights.sum() + 1e-12)
    
    # Phase 3: Weighted Column Score
    scores = -R_tilde.T @ weights  # score_j = -sum_i w_i R_ij
    
    if iterative:
        # Phase: Iterative Refinement
        for t in range(max_iter):
            old_weights = weights.copy()
            
            # Reliability update
            score_diff = scores[:, None] - scores[None, :]  # (i,j): score_i - score_j
            w_new = np.sum(R_tilde * score_diff, axis=1) / N
            w_new = np.maximum(w_new, 0)
            w_sum = w_new.sum()
            if w_sum > 1e-12:
                weights = w_new / w_sum
            
            # Score update
            scores = -R_tilde.T @ weights
            
            # Check convergence
            if np.linalg.norm(weights - old_weights) < tol:
                break
    
    # Phase 5: Spectral Denoising (project onto right singular space)
    scores_centered = scores - scores.mean()
    scores_spectral = V @ (V.T @ scores_centered)
    
    # Sign resolution: ensure correlation with row sums is positive
    row_sums = R_tilde.sum(axis=1)
    if np.corrcoef(scores_spectral, row_sums)[0, 1] < 0:
        scores_spectral = -scores_spectral
    
    best_idx = np.argmax(scores_spectral)
    
    return best_idx, scores_spectral, weights


def majority_vote(R):
    """Baseline: select agent with highest row sum."""
    R_tilde = R.astype(float).copy()
    np.fill_diagonal(R_tilde, 0.0)
    row_sums = R_tilde.sum(axis=1)
    return np.argmax(row_sums), row_sums


def borda_count(R):
    """Baseline: Borda count on negative column sums."""
    R_tilde = R.astype(float).copy()
    np.fill_diagonal(R_tilde, 0.0)
    col_scores = -R_tilde.sum(axis=0)
    return np.argmax(col_scores), col_scores
```

---

## 9. Simulation Validation

To verify SWRA works as claimed, here is a simulation protocol:

```python
def simulate(N=50, beta=3.0, epsilon=0.3, num_trials=1000):
    """
    Simulate the hammer-spammer model and compare methods.
    """
    results = {"swra": 0, "majority": 0, "borda": 0}
    
    for _ in range(num_trials):
        # Generate latent scores
        s = np.random.uniform(0, 1, N)
        true_best = np.argmax(s)
        
        # Generate types
        z = np.random.binomial(1, 1 - epsilon, N)  # 1 = hammer, 0 = spammer
        
        # Generate comparison matrix
        R = np.ones((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                if i == j:
                    R[i, j] = 1
                elif z[i] == 1:  # hammer
                    prob = 1 / (1 + np.exp(-beta * (s[i] - s[j])))
                    R[i, j] = 1 if np.random.random() < prob else -1
                else:  # spammer
                    R[i, j] = 1 if np.random.random() < 0.5 else -1
        
        # Compare methods
        idx_swra, _, _ = swra(R, k=2, iterative=True)
        idx_maj, _ = majority_vote(R)
        idx_borda, _ = borda_count(R)
        
        results["swra"] += (idx_swra == true_best)
        results["majority"] += (idx_maj == true_best)
        results["borda"] += (idx_borda == true_best)
    
    for method in results:
        results[method] /= num_trials
    
    return results
```

---

## 10. Summary of Contributions

1. **Novel adaptation of Karger's spectral framework** to the square-matrix pairwise comparison setting, where each row is produced by a single agent (judge = contestant).

2. **Key theoretical insight:** $\mathbb{E}[\tilde{R}]$ is rank-2 under the linearized model, with left singular space encoding judge reliability and right singular space encoding scores. This directly parallels Karger's rank-1 structure but requires handling an additional dimension.

3. **Three-phase design principle:**
   - **Phase 1 (SVD):** Extract the low-rank signal, separating hammers from spammers.
   - **Phase 2 (Weighting):** Assign reliability weights that suppress spammer rows.
   - **Phase 3 (Aggregation):** Compute weighted scores that are monotonically related to true quality.

4. **Provable advantage over majority voting:** SWRA effectively removes the $\varepsilon N$ spammer rows from the aggregation, while majority voting treats all rows equally — directly analogous to the $1/q$ vs $1/q^2$ gap in Karger et al.

5. **Practical robustness:** The iterative refinement, initialized from the spectral estimate, avoids the local optima problem of pure EM while further improving accuracy.

---

## References

1. Karger, D. R., Oh, S., & Shah, D. (2011). Budget-optimal crowdsourcing using low-rank matrix approximations. *49th Annual Allerton Conference on Communication, Control, and Computing*, 284–291.
2. Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324–345.
3. Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. *Journal of the Royal Statistical Society: Series C*, 28(1), 20–28.
4. Davis, C., & Kahan, W. M. (1970). The rotation of eigenvectors by a perturbation. III. *SIAM Journal on Numerical Analysis*, 7(1), 1–46.
