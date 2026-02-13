# Experiment Setup: Hammer-Spammer Model Validation

## Overview

Validate Karger's Hammer-Spammer theory using real LLM agents on mathematical reasoning tasks.

**Research Question**: Do SVD-based methods (CCRR, Spectral Ranking) outperform Majority Voting when identifying reliable agents in the presence of noise?

---

## 🎯 Algorithms Under Test

### 1. CCRR (Cross-Consistency Robust Ranking)
**Type**: SVD-based with EM refinement

**Algorithm**:
```
Phase 1: Row-norm Spammer Detection
  - Agreement matrix G = M @ M^T / (N-2)
  - Row-norm ρᵢ = Σₖ Gᵢₖ²
  - Threshold: median(ρ) / 2
  - Hammers: ρᵢ ~ O(N), Spammers: ρᵢ ~ O(1)

Phase 2: EM Refinement (iterative)
  - M-step: Bradley-Terry MLE for scores
  - E-step: Update weights via likelihood
  - Iterate until convergence

Phase 3: Selection
  - Return argmax(score)
```

**Theory**: Karger et al. (2011) + Bradley-Terry (1952)

### 2. Spectral Ranking
**Type**: Pure SVD-based

**Algorithm**:
```
1. SVD(R̃) → U, Σ, V^T  (R̃ = R with diagonal removed)
2. u₁ = U[:, 0]        (left singular vector)
3. weights = u₁²       (judge reliability)
4. scores = -weights @ R̃  (weighted column aggregation)
5. Return argmax(scores)
```

**Theory**: Karger Theorem II.3 (SVD identifies reliable judges)

### 3. Majority Voting (Baseline)
**Type**: Simple voting

**Algorithm**:
```
1. Normalize all answers
2. Count occurrences of each answer
3. Return most common answer
```

**Theory**: None (treat all agents equally)

---

## 🤖 Agent Configuration

**Total Agents**: 10

### Tier Distribution
- **High-quality (2)**: Best available models
  - `openai/gpt-4-turbo`
  - `anthropic/claude-sonnet-4.5`

- **Mid-tier (3)**: Capable but not SOTA
  - `openai/gpt-3.5-turbo`
  - `anthropic/claude-haiku-4.5`
  - `google/gemini-2.5-flash`

- **Low-tier (5)**: Weak models (expected to struggle)
  - `meta-llama/llama-3.1-8b-instruct`
  - `mistralai/mistral-7b-instruct`
  - `google/gemma-2-9b-it`
  - `meta-llama/llama-3.2-3b-instruct`
  - `meta-llama/llama-3.2-1b-instruct`

### Design Rationale
- **Diversity**: Mix of model families (GPT, Claude, Llama, Mistral, Gemini)
- **Quality spread**: 2 strong + 3 mid + 5 weak ensures performance variance
- **Realistic**: Simulates real-world ensemble with varying capabilities

---

## 📚 Datasets

### GSM8K (Primary)
- **Source**: Grade School Math 8K
- **Problems**: 100 questions
- **Start Index**: 900 (harder multi-step reasoning problems)
- **Difficulty**: Progressive (index 900+ requires multi-hop reasoning)
- **Format**: Numeric answers

**Why GSM8K?**
- Clear ground truth (numeric answers)
- Difficulty progression allows testing on harder problems
- Weak models struggle → creates performance variance needed for SVD

### MATH (Secondary)
- **Source**: MATH dataset
- **Problems**: 50 questions (optional)
- **Difficulty Filter**: Level 4-5 only (hard problems)
- **Format**: LaTeX mathematical expressions

---

## 🔬 Experimental Tracks

### Track B: Natural Experiment
**Purpose**: Test real-world utility with actual LLM agents

**Process**:
1. All N=10 agents answer each question
2. Each agent compares their answer with all others (N×N comparison matrix R)
3. Run CCRR, Spectral, MV on R
4. Check correctness against ground truth

**API Calls**:
- Answer generation: 10 × Q questions
- Comparisons: ~90 × Q (only hammers compare, spammers are random)

### Track A: Artificial Spammer Injection
**Purpose**: Test robustness to varying noise levels (zero additional API cost)

**Process**:
1. Reuse Track B's comparison matrices and answers
2. For each spammer ratio k ∈ {0.1, 0.2, 0.4, 0.6, 0.8}:
   - Randomly select k×N agents as "spammers"
   - Replace spammer answers with WRONG answers from pool
   - Replace spammer rows in R with random ±1
   - Re-run CCRR, Spectral, MV
   - Measure accuracy
3. Repeat 5 trials per k for variance estimation

**Spammer Answer Injection Strategy**:
1. **Primary**: Use actual wrong answers from pool (realistic)
2. **Fallback**: Generate wrong answer (GT ± random offset for numeric)

**No Additional API Cost**: Purely post-hoc analysis

---

## ⚙️ Configuration Parameters

### Core Settings
```yaml
num_questions: 100
num_agents: 10
dataset: gsm8k
start_index: 900
```

### Track A Settings
```yaml
spammer_ratios: [0.1, 0.2, 0.4, 0.6, 0.8]
  # 0.1 = 10% spammers (1 agent)
  # 0.2 = 20% spammers (2 agents)
  # 0.4 = 40% spammers (4 agents)
  # 0.6 = 60% spammers (6 agents)
  # 0.8 = 80% spammers (8 agents)

track_a_trials: 5  # Random trials per ratio for variance
```

### Algorithm Parameters
```yaml
# CCRR
beta: 5.0        # Bradley-Terry discrimination
epsilon: 0.1     # Prior spammer probability
T: 1000         # EM iterations

# Spectral
beta: 5.0       # (not used, kept for interface)
```

### Performance Settings
```yaml
parallel_generation: true   # Parallel answer generation (max_workers=2)
parallel_comparison: true   # Parallel comparisons (max_workers=8)
save_frequency: 10         # Save intermediate results every N questions
```

---

## 📊 Expected Results

### Hypothesis (based on Karger Theorem II.3)

**Low spammer rate (k ≤ 20%)**:
- All methods perform similarly (~90-95%)
- MV may slightly outperform (less overhead)

**Medium spammer rate (k = 40-60%)**:
- CCRR and Spectral start showing advantage
- Gap widens as k increases
- Expected: CCRR ≈ Spectral > MV

**High spammer rate (k ≥ 60%)**:
- Dramatic SVD advantage
- MV collapses (overwhelmed by noise)
- Expected at k=80%:
  - CCRR: 40-50%
  - Spectral: 20-40%
  - MV: <10%

### Key Metrics

1. **Accuracy**: % of questions answered correctly
2. **Robustness Curve**: Accuracy vs spammer ratio
3. **Crossover Point**: Spammer ratio where SVD > MV
4. **Degradation Rate**: Slope of accuracy decline

---

## 🚀 How to Run

### Full Experiment (100 questions)

**Docker Container**:
```bash
cd /workspace/coding-test/test
git pull

python -m src.agents.run_experiment \
    --dataset gsm8k \
    --num-questions 100 \
    --start-index 900 \
    --track-a \
    --spammer-ratios 0.1,0.2,0.4,0.6,0.8 \
    --output-dir results/gsm8k_100q \
    --save-frequency 10 \
    --verbose
```

**WSL Local**:
```bash
cd /mnt/c/workspace/24h_test/24h_hammer_spammer
git pull

.venv/bin/python -m src.agents.run_experiment \
    --dataset gsm8k \
    --num-questions 100 \
    --start-index 900 \
    --track-a \
    --spammer-ratios 0.1,0.2,0.4,0.6,0.8 \
    --output-dir results/gsm8k_100q \
    --save-frequency 10 \
    --verbose
```

### Quick Test (10 questions)
```bash
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --num-questions 10 \
    --start-index 900 \
    --track-a \
    --spammer-ratios 0.2,0.4,0.6,0.8 \
    --verbose
```

### Using Shell Script
```bash
# Default: 50 questions per dataset
./run_main_test.sh

# Custom: 100 questions
./run_main_test.sh 100
```

---

## 📈 Output Structure

### Files Generated
```
results/gsm8k_100q/
├── intermediate_results_TIMESTAMP.json  (every 10 questions)
└── final_results_TIMESTAMP.json         (complete results)
```

### JSON Structure
```json
{
  "config": { ... },
  "results": [
    {
      "question_idx": 0,
      "question": "...",
      "ground_truth": "...",
      "answers": [ ... ],
      "comparison_matrix": [ ... ],
      "ccrr_correct": true,
      "spectral_correct": true,
      "mv_correct": true,
      ...
    }
  ],
  "track_a": {
    "0": { "ccrr_mean": 90.0, "spectral_mean": 90.0, "mv_mean": 100.0, ... },
    "1": { "ccrr_mean": 88.0, "spectral_mean": 84.0, "mv_mean": 88.0, ... },
    ...
  },
  "track_a_detailed": {
    "0": {
      "0": { "question": "...", "gt": "...", "ccrr": [true], "spectral": [true], "mv": [true] },
      ...
    }
  }
}
```

---

## ⏱️ Resource Estimates

### Time
- **10 questions**: ~10-15 minutes
- **100 questions**: ~1.5-2 hours

### API Cost (OpenRouter)
- **10 questions**: ~$1-2
- **100 questions**: ~$10-20
- **Track A**: $0 (no additional API calls)

### API Calls
- **Answer generation**: 100 questions × 10 agents = 1,000 calls
- **Comparisons**: 100 questions × ~90 comparisons = 9,000 calls
- **Total**: ~10,000 API calls (Track B only)

---

## 🔧 Analysis Tools

### Analyze Results
```bash
# Analyze final results
python analyze_results.py results/gsm8k_100q/final_results_*.json

# Analyze all intermediate results
python analyze_results.py 'results/gsm8k_100q/intermediate_*.json'
```

### Output Includes
- Track B accuracy summary (CCRR, Spectral, MV)
- Track A robustness curves
- Per-question detailed breakdown (✓/✗ for each k)
- Agent weight analysis (reliability scores)

---

## 📝 Notes

### Implementation Details

1. **Answer Normalization**:
   - Lowercase, remove punctuation
   - Extract numeric values: "8.00" → "8"
   - Handles various formats consistently

2. **Comparison Matrix R**:
   - R_ij = +1: agent i prefers their answer over j's
   - R_ij = -1: agent i prefers j's answer over theirs
   - R_ii = +1: always (self-comparison)

3. **Spammer Injection** (Track A):
   - Corrupts both answers AND comparison matrix
   - Ensures fair comparison (all methods affected equally)
   - Uses realistic wrong answers when available

### Known Limitations

1. **Problem Difficulty**: GSM8K 900+ still relatively easy for strong models
   - High-quality models: 90-100% accuracy
   - May need MATH Level 5 for more challenging test

2. **Agent Diversity**: Only 10 agents
   - Karger's theory assumes large N
   - Results more variable with small N

3. **Comparison Quality**: LLM-based pairwise comparison
   - May be noisy (models can misjudge)
   - Could affect both CCRR and Spectral

---

## 🎓 References

1. Karger, D. R., Oh, S., & Shah, D. (2011). Budget-optimal task allocation for reliable crowdsourcing systems. *Operations Research*.

2. Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*.

3. Hammer-Spammer Model: Assumes workers are either reliable ("hammers") or random ("spammers").

---

**Last Updated**: 2026-02-14
**Experiment ID**: gsm8k_100q_track_a
**Status**: Ready to run
