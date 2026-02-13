# CCRR (Cross-Consistency Robust Ranking) Algorithm Specification

## 1. Problem Definition

### 1.1 Input

A matrix `R` of size `N x N` where:
- `R[i][j] ∈ {+1, -1}` for all `i, j`
- `R[i][i] = 1` for all `i` (diagonal is always +1)
- Row `i` is entirely produced by agent `i`

### 1.2 Output

A single integer `best_idx` in `{0, 1, ..., N-1}` — the index of the agent whose answer is estimated to be the best (highest latent quality score).

### 1.3 Hyperparameters

| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `beta` | float | 5.0 | [1, 10] | Discrimination parameter of the Bradley-Terry comparison model |
| `epsilon` | float | 0.1 | (0, 0.5) | Prior probability that any given agent is a spammer |
| `T` | int | 5 | [2, 10] | Number of iterative refinement rounds |

### 1.4 Generative Model (Background — do NOT implement this as part of the algorithm; this explains WHY the algorithm works)

Each agent `i` has:
- A latent quality score `s_i ~ Uniform(0, 1)` (unknown)
- A latent type `z_i ∈ {H, S}` where `P(z_i = S) = epsilon` (unknown)

If `z_i = H` (hammer): `P(R[i][j] = 1) = sigmoid(beta * (s_i - s_j))`
If `z_i = S` (spammer): `P(R[i][j] = 1) = 0.5` (random coin flip)

The algorithm must recover `argmax_i s_i` from `R` alone, without knowing `s` or `z`.

---

## 2. Mathematical Primitives

### 2.1 Sigmoid Function

```
sigmoid(x) = 1.0 / (1.0 + exp(-x))
```

Implementation note: for numerical stability, use:
```
if x >= 0:
    return 1.0 / (1.0 + exp(-x))
else:
    ex = exp(x)
    return ex / (1.0 + ex)
```

### 2.2 Log-Sigmoid Function

```
log_sigmoid(x) = log(sigmoid(x))
```

Implementation note: for numerical stability, use:
```
log_sigmoid(x) = -log(1 + exp(-x))
```

Or more precisely:
```
if x >= 0:
    return -log(1 + exp(-x))
else:
    return x - log(1 + exp(x))
```

This is critical — naive `log(sigmoid(x))` will produce `-inf` or `nan` for large `|x|`.

### 2.3 Median

Standard median of an array. For even-length arrays, use the average of the two middle values.

### 2.4 MAD (Median Absolute Deviation)

```
MAD(arr) = median(|arr[i] - median(arr)| for all i)
```

If `MAD = 0` (can happen when more than half the values are identical), set `MAD = 1.0` to avoid division by zero.

---

## 3. Algorithm — Phase 1: Cross-Consistency Spammer Detection

### 3.1 Compute Cross-Consistency Score

For each agent `i` from `0` to `N-1`:

```
C[i] = sum over all j != i of: R[i][j] * R[j][i]
```

This is the dot product of row `i` and column `i` of `R`, excluding the diagonal.

**What this measures:**
- If both agent `i` and agent `j` are hammers, they will tend to DISAGREE (if `i` thinks `i > j`, then `j` thinks `j < i`), so `R[i][j] * R[j][i]` tends to be `-1`.
- If either is a spammer, `R[i][j] * R[j][i]` is equally likely `+1` or `-1`.
- Therefore: hammers get `C[i] << 0` (strongly negative), spammers get `C[i] ≈ 0`.

### 3.2 Compute Initial Hammer Weights

```
C_med = median(C)
C_mad = MAD(C)
if C_mad == 0:
    C_mad = 1.0

gamma = 2.0 / C_mad

For each agent i:
    w[i] = sigmoid(-gamma * (C[i] - C_med))
```

**Explanation:**
- `C_med` centers the distribution.
- `gamma` scales the sigmoid so that agents 1 MAD below the median get `w ≈ 0.88` and agents 1 MAD above get `w ≈ 0.12`.
- Agents with very negative `C[i]` (hammers) get `w[i]` close to `1`.
- Agents with `C[i]` near `0` (spammers) get `w[i]` close to `0`.

---

## 4. Algorithm — Phase 2: Weighted Score Estimation

For each agent `i` from `0` to `N-1`:

```
row_sum = sum over all j != i of: R[i][j]
col_sum = sum over all j != i of: w[j] * R[j][i]

s_hat[i] = w[i] * row_sum - col_sum
```

**Component breakdown:**
- `w[i] * row_sum`: Agent `i`'s self-assessment, weighted by how likely `i` is a hammer. If `i` is a spammer (`w[i] ≈ 0`), this term vanishes.
- `-col_sum = -sum_j w[j] * R[j][i]`: Other agents' assessment of `i`. If hammer `j` sets `R[j][i] = -1` (meaning `j` thinks `i` is better than `j`), this contributes `+w[j]` to the score. If `R[j][i] = +1` (meaning `j` thinks `j` is better than `i`), this contributes `-w[j]`.

**Why the minus sign on `col_sum`:**
`R[j][i] = 1` means "agent `j` judges its own answer better than agent `i`'s" — this is NEGATIVE evidence for agent `i`. So we subtract it.

---

## 5. Algorithm — Phase 3: Iterative Refinement

Repeat `T` times:

### 5.1 Update Weights (E-step)

For each agent `i` from `0` to `N-1`:

```
L_H = 0.0
for each j != i:
    diff = beta * (s_hat[i] - s_hat[j])
    if R[i][j] == 1:
        L_H += log_sigmoid(diff)
    else:  # R[i][j] == -1
        L_H += log_sigmoid(-diff)

L_S = (N - 1) * log(0.5)

# Log-posterior for hammer:
log_pH = log(1 - epsilon) + L_H
log_pS = log(epsilon) + L_S

# Numerically stable softmax:
max_log = max(log_pH, log_pS)
w[i] = exp(log_pH - max_log) / (exp(log_pH - max_log) + exp(log_pS - max_log))
```

**What this does:**
- `L_H` is the log-likelihood of observing row `i` of `R` under the hypothesis that agent `i` is a hammer with estimated scores `s_hat`.
- `L_S = (N-1) * log(0.5)` is the log-likelihood under the hypothesis that agent `i` is a spammer (each entry is a fair coin flip).
- `w[i]` is the posterior probability that agent `i` is a hammer, computed via Bayes' rule.

### 5.2 Update Scores (M-step)

For each agent `i` from `0` to `N-1`:

```
row_sum = sum over all j != i of: R[i][j]
col_sum = sum over all j != i of: w[j] * R[j][i]

s_hat[i] = w[i] * row_sum - col_sum
```

This is identical to Phase 2. The scores are re-estimated using the updated weights.

### 5.3 Normalize Scores (Important!)

After computing all `s_hat[i]`:

```
s_mean = mean(s_hat)
s_std = std(s_hat)
if s_std == 0:
    s_std = 1.0

for each i:
    s_hat[i] = (s_hat[i] - s_mean) / s_std
```

**Why normalize:** The score formula produces values on an arbitrary scale. Without normalization, the `beta * (s_hat[i] - s_hat[j])` term in the E-step can explode or vanish, causing numerical issues. Standardizing to mean=0, std=1 keeps the sigmoid inputs in a reasonable range.

---

## 6. Algorithm — Phase 4: Final Selection

```
best_idx = argmax over i of s_hat[i]
```

Return `best_idx`.

---

## 7. Complete Pseudocode

```
function CCRR(R, beta=5.0, epsilon=0.1, T=5):
    N = number of rows in R

    # ===== Phase 1: Cross-Consistency Spammer Detection =====

    C = array of size N, initialized to 0
    for i = 0 to N-1:
        for j = 0 to N-1:
            if j != i:
                C[i] += R[i][j] * R[j][i]

    C_med = median(C)
    C_mad = MAD(C)
    if C_mad == 0:
        C_mad = 1.0
    gamma = 2.0 / C_mad

    w = array of size N
    for i = 0 to N-1:
        w[i] = sigmoid(-gamma * (C[i] - C_med))

    # ===== Phase 2: Initial Weighted Score Estimation =====

    s_hat = array of size N
    for i = 0 to N-1:
        row_sum = 0
        col_sum = 0
        for j = 0 to N-1:
            if j != i:
                row_sum += R[i][j]
                col_sum += w[j] * R[j][i]
        s_hat[i] = w[i] * row_sum - col_sum

    # Normalize
    s_mean = mean(s_hat)
    s_std = std(s_hat)
    if s_std == 0:
        s_std = 1.0
    for i = 0 to N-1:
        s_hat[i] = (s_hat[i] - s_mean) / s_std

    # ===== Phase 3: Iterative Refinement =====

    for t = 1 to T:

        # --- 3a. Update weights (E-step) ---
        for i = 0 to N-1:
            L_H = 0.0
            for j = 0 to N-1:
                if j != i:
                    diff = beta * (s_hat[i] - s_hat[j])
                    if R[i][j] == 1:
                        L_H += log_sigmoid(diff)
                    else:
                        L_H += log_sigmoid(-diff)

            L_S = (N - 1) * log(0.5)

            log_pH = log(1 - epsilon) + L_H
            log_pS = log(epsilon) + L_S

            max_log = max(log_pH, log_pS)
            w[i] = exp(log_pH - max_log) / (exp(log_pH - max_log) + exp(log_pS - max_log))

        # --- 3b. Update scores (M-step) ---
        for i = 0 to N-1:
            row_sum = 0
            col_sum = 0
            for j = 0 to N-1:
                if j != i:
                    row_sum += R[i][j]
                    col_sum += w[j] * R[j][i]
            s_hat[i] = w[i] * row_sum - col_sum

        # --- 3c. Normalize scores ---
        s_mean = mean(s_hat)
        s_std = std(s_hat)
        if s_std == 0:
            s_std = 1.0
        for i = 0 to N-1:
            s_hat[i] = (s_hat[i] - s_mean) / s_std

    # ===== Phase 4: Selection =====

    best_idx = argmax(s_hat)
    return best_idx
```

---

## 8. Implementation in Python (numpy)

```python
import numpy as np

def sigmoid(x):
    """Numerically stable sigmoid."""
    pos = x >= 0
    result = np.empty_like(x, dtype=float)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result

def log_sigmoid(x):
    """Numerically stable log-sigmoid."""
    result = np.empty_like(x, dtype=float)
    pos = x >= 0
    result[pos] = -np.log1p(np.exp(-x[pos]))
    result[~pos] = x[~pos] - np.log1p(np.exp(x[~pos]))
    return result

def mad(arr):
    """Median Absolute Deviation."""
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def ccrr(R, beta=5.0, epsilon=0.1, T=5):
    """
    Cross-Consistency Robust Ranking algorithm.

    Parameters
    ----------
    R : np.ndarray of shape (N, N)
        Comparison matrix. R[i][j] ∈ {+1, -1}.
        R[i][j] = 1 means agent i judges its own answer better than agent j's.
        R[i][i] = 1 for all i.
    beta : float
        Bradley-Terry discrimination parameter. Default 5.0.
    epsilon : float
        Prior spammer probability. Default 0.1.
    T : int
        Number of refinement iterations. Default 5.

    Returns
    -------
    best_idx : int
        Index of the agent with the estimated highest quality score.
    s_hat : np.ndarray of shape (N,)
        Estimated quality scores for all agents (standardized).
    w : np.ndarray of shape (N,)
        Estimated hammer probabilities for all agents.
    """
    N = R.shape[0]

    # ===== Phase 1: Cross-Consistency Spammer Detection =====
    # C[i] = sum_{j != i} R[i][j] * R[j][i]
    # R[j][i] is column i of R, which equals R.T[i][j]
    # So element-wise: R * R.T, then sum rows, minus diagonal contribution
    cross = R * R.T  # element-wise product
    C = np.sum(cross, axis=1) - np.diag(cross)  # subtract diagonal (which is R[i][i]*R[i][i] = 1)

    C_med = np.median(C)
    C_mad = mad(C)
    if C_mad == 0:
        C_mad = 1.0
    gamma = 2.0 / C_mad

    w = sigmoid(-gamma * (C - C_med))

    # ===== Phase 2: Initial Weighted Score Estimation =====
    # row_sum[i] = sum_{j != i} R[i][j] = sum of row i minus R[i][i]
    row_sums = np.sum(R, axis=1) - 1.0  # subtract diagonal R[i][i] = 1

    # col_sum[i] = sum_{j != i} w[j] * R[j][i] = (w * R)^T column sums, minus diagonal
    # R[j][i] = R.T[i][j], so col_sum[i] = sum_j w[j] * R.T[i][j] - w[i] * R[i][i]
    col_sums = R.T @ w - w  # R.T @ w gives sum_j w[j]*R[j][i] for each i; subtract w[i]*R[i][i]=w[i]

    s_hat = w * row_sums - col_sums

    # Normalize
    s_std = np.std(s_hat)
    if s_std == 0:
        s_std = 1.0
    s_hat = (s_hat - np.mean(s_hat)) / s_std

    # ===== Phase 3: Iterative Refinement =====
    for t in range(T):

        # --- 3a. Update weights (E-step) ---
        # For each agent i, compute log-likelihood under H and S
        # L_H[i] = sum_{j != i} [ (R[i][j]==1)*log_sigmoid(beta*(s_hat[i]-s_hat[j]))
        #                        + (R[i][j]==-1)*log_sigmoid(-beta*(s_hat[i]-s_hat[j])) ]

        # Compute beta * (s_hat[i] - s_hat[j]) for all i, j
        diff_matrix = beta * (s_hat[:, None] - s_hat[None, :])  # shape (N, N)

        # For R[i][j] = +1: contribute log_sigmoid(diff)
        # For R[i][j] = -1: contribute log_sigmoid(-diff)
        # Unified: log_sigmoid(R[i][j] * diff_matrix[i][j])
        log_sig_values = log_sigmoid(R * diff_matrix)

        # Sum over j != i (exclude diagonal)
        np.fill_diagonal(log_sig_values, 0.0)
        L_H = np.sum(log_sig_values, axis=1)

        L_S = (N - 1) * np.log(0.5)

        log_pH = np.log(1 - epsilon) + L_H
        log_pS = np.log(epsilon) + L_S

        # Numerically stable softmax
        max_log = np.maximum(log_pH, log_pS)
        w = np.exp(log_pH - max_log) / (np.exp(log_pH - max_log) + np.exp(log_pS - max_log))

        # --- 3b. Update scores (M-step) ---
        row_sums = np.sum(R, axis=1) - 1.0
        col_sums = R.T @ w - w
        s_hat = w * row_sums - col_sums

        # --- 3c. Normalize ---
        s_std = np.std(s_hat)
        if s_std == 0:
            s_std = 1.0
        s_hat = (s_hat - np.mean(s_hat)) / s_std

    # ===== Phase 4: Selection =====
    best_idx = int(np.argmax(s_hat))

    return best_idx, s_hat, w
```

---

## 9. Simulation and Evaluation Code

The following code generates synthetic data from the Hammer-Spammer model and evaluates the algorithm.

```python
import numpy as np

def generate_hammer_spammer_data(N, beta, epsilon, seed=None):
    """
    Generate a comparison matrix R from the Hammer-Spammer model.

    Parameters
    ----------
    N : int
        Number of agents.
    beta : float
        Bradley-Terry discrimination parameter.
    epsilon : float
        Probability each agent is a spammer.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    R : np.ndarray of shape (N, N), entries in {+1, -1}
    s : np.ndarray of shape (N,), true latent scores
    z : np.ndarray of shape (N,), True types ('H' or 'S')
    """
    rng = np.random.default_rng(seed)

    # Generate latent scores
    s = rng.uniform(0, 1, size=N)

    # Generate types
    z = np.where(rng.random(N) < epsilon, 'S', 'H')

    # Generate comparison matrix
    R = np.ones((N, N), dtype=int)  # R[i][i] = 1

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if z[i] == 'H':
                prob = 1.0 / (1.0 + np.exp(-beta * (s[i] - s[j])))
                R[i, j] = 1 if rng.random() < prob else -1
            else:
                R[i, j] = 1 if rng.random() < 0.5 else -1

    return R, s, z

def evaluate(N, beta, epsilon, n_trials=1000, seed=0):
    """
    Evaluate CCRR accuracy over multiple trials.

    Returns
    -------
    accuracy : float
        Fraction of trials where CCRR correctly identified the best agent.
    baseline_accuracy : float
        Fraction of trials where simple row-sum (Borda count) identified the best agent.
    majority_accuracy : float
        Fraction of trials where majority voting identified the best agent.
    """
    rng_seeds = np.random.default_rng(seed).integers(0, 2**31, size=n_trials)

    ccrr_correct = 0
    borda_correct = 0
    majority_correct = 0

    for trial in range(n_trials):
        R, s, z = generate_hammer_spammer_data(N, beta, epsilon, seed=int(rng_seeds[trial]))
        true_best = int(np.argmax(s))

        # CCRR
        pred_best, _, _ = ccrr(R, beta=beta, epsilon=epsilon, T=5)
        if pred_best == true_best:
            ccrr_correct += 1

        # Baseline 1: Borda count (simple row sum)
        row_sums = np.sum(R, axis=1) - 1  # exclude diagonal
        borda_best = int(np.argmax(row_sums))
        if borda_best == true_best:
            borda_correct += 1

        # Baseline 2: Majority voting
        # For each pair (i,j), take majority of R[i][j] and -R[j][i]
        # Then count wins for each agent
        wins = np.zeros(N)
        for i in range(N):
            for j in range(i+1, N):
                vote = R[i][j] + (-R[j][i])  # +2, 0, or -2
                if vote > 0:
                    wins[i] += 1
                elif vote < 0:
                    wins[j] += 1
                else:
                    wins[i] += 0.5
                    wins[j] += 0.5
        majority_best = int(np.argmax(wins))
        if majority_best == true_best:
            majority_correct += 1

    return {
        'ccrr_accuracy': ccrr_correct / n_trials,
        'borda_accuracy': borda_correct / n_trials,
        'majority_accuracy': majority_correct / n_trials,
    }
```

---

## 10. Experiment Configurations

Run the following experiments to produce the analysis. Print results as a table.

### 10.1 Vary N (number of agents)

```python
for N in [10, 20, 50, 100]:
    result = evaluate(N=N, beta=5.0, epsilon=0.1, n_trials=1000)
    print(f"N={N}: {result}")
```

### 10.2 Vary epsilon (spammer ratio)

```python
for eps in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    result = evaluate(N=20, beta=5.0, epsilon=eps, n_trials=1000)
    print(f"epsilon={eps}: {result}")
```

### 10.3 Vary beta (discrimination parameter)

```python
for b in [1, 2, 3, 5, 7, 10]:
    result = evaluate(N=20, beta=b, epsilon=0.1, n_trials=1000)
    print(f"beta={b}: {result}")
```

### 10.4 Ablation study (which phase matters?)

Implement three variants:
1. **Phase 1 only**: Use initial weights from cross-consistency, skip Phase 3 refinement.
2. **No cross-consistency**: Set all `w[i] = 1` in Phase 2, then run Phase 3 (EM without good init).
3. **Full CCRR**: All phases.

Compare accuracy across these variants.

---

## 11. Expected Output Format

For each experiment, produce:

1. **A table** with columns: configuration, CCRR accuracy, Borda accuracy, Majority accuracy.
2. **A plot** (matplotlib) showing accuracy vs. the varied parameter, with three lines (CCRR, Borda, Majority).
3. **A brief text summary** of the key findings.

Save plots as PNG files:
- `experiment_vary_N.png`
- `experiment_vary_epsilon.png`
- `experiment_vary_beta.png`
- `experiment_ablation.png`

---

## 12. Numerical Edge Cases to Handle

| Situation | Cause | Fix |
|-----------|-------|-----|
| `log_sigmoid` returns `-inf` | Very large negative input | Use the stable implementation in Section 2.2 |
| `L_H` is `-inf` | All comparisons contradict the estimated scores | Clamp: `L_H = max(L_H, -500)` |
| `w[i]` is `nan` | Both `log_pH` and `log_pS` are `-inf` | Set `w[i] = 0.5` (uncertain) |
| `s_std == 0` | All scores are identical | Set `s_std = 1.0` |
| `C_mad == 0` | All `C[i]` are identical | Set `C_mad = 1.0` |
| `N <= 2` | Not enough agents for meaningful comparisons | Algorithm still runs but results are unreliable; warn user |

---

## 13. Complexity Analysis

| Phase | Time Complexity | Space Complexity |
|-------|----------------|-----------------|
| Phase 1 (Cross-consistency) | O(N^2) | O(N) |
| Phase 2 (Score estimation) | O(N^2) | O(N) |
| Phase 3 (T iterations) | O(T * N^2) | O(N^2) for diff_matrix |
| Phase 4 (Selection) | O(N) | O(1) |
| **Total** | **O(T * N^2)** | **O(N^2)** |

---

## 14. File Structure

When implementing, create the following files:

```
problem1/
├── ccrr.py              # Core algorithm (Section 8)
├── simulation.py        # Data generation + evaluation (Section 9)
├── run_experiments.py   # All experiments (Section 10)
└── results/
    ├── experiment_vary_N.png
    ├── experiment_vary_epsilon.png
    ├── experiment_vary_beta.png
    └── experiment_ablation.png
```
