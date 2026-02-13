# 24h Hammer Spammer - LLM Agent Ranking

Implementation and comparison of ranking algorithms for LLM agent systems.

## Algorithms

- **CCRR** (Cross-Consistency Robust Ranking) - EM-based algorithm with spammer detection
- **SVD** (Singular Value Decomposition) - Matrix factorization approach

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running Experiments

**Quick Test (10 questions per dataset):**
```bash
./run_quick_test.sh
```
Tests both GSM8K and MATH datasets with 10 questions each (~5-10 min).

**Hard Problems Test (challenging problems only):**
```bash
./run_hard_test.sh
```
Tests on MATH Level 4-5 problems only (hardest difficulty).

**Main Test (adjustable budget):**
```bash
./run_main_test.sh          # 50 questions per dataset (default)
./run_main_test.sh 100      # 100 questions per dataset
```
Process as many questions as your API limits and budget allow.

**Single Dataset with Difficulty Filter:**
```bash
# GSM8K (no difficulty filter available)
python -m src.agents.run_experiment --dataset gsm8k --num-questions 10 --verbose

# MATH - all levels
python -m src.agents.run_experiment --dataset math --num-questions 10 --verbose

# MATH - hard only (Level 4-5)
python -m src.agents.run_experiment --dataset math --num-questions 10 --difficulty hard --verbose

# MATH - medium (Level 3)
python -m src.agents.run_experiment --dataset math --num-questions 10 --difficulty medium --verbose

# MATH - easy (Level 1-2)
python -m src.agents.run_experiment --dataset math --num-questions 10 --difficulty easy --verbose
```

## Methods Compared

| Method | Type | Description |
|--------|------|-------------|
| **CCRR** | Algorithm | Cross-Consistency + EM (Primary) |
| **SVD** | Algorithm | Rank-1 SVD (Comparison) |
| **Majority Voting** | Baseline | Simple majority |
| **Best Single Model** | Baseline | GPT-4o only |
| **Random** | Baseline | Random selection |

## Project Structure

```
24h_hammer_spammer/
├── src/
│   ├── algorithm/       # CCRR and SVD algorithms
│   ├── agents/          # LLM agent system (N=15)
│   ├── evaluation/      # Validation and baselines
│   └── utils/           # Dataset loaders
├── results/
│   ├── quick_test/      # Quick validation results
│   └── main_test/       # Full experiment results
└── CCRR_Algorithm_Spec.md  # Algorithm specification
```

## Configuration

- **N = 15 agents**: GPT-4o, Claude, Gemini, Llama, etc.
- **Datasets**: GSM8K (math word problems), MATH (competition math)
- **Validation**: Ground truth + Oracle model (GPT-4o)

## Results

Results include:
- Accuracy comparison across all methods
- Per-agent statistics (API usage, tokens)
- Oracle validation scores
- Detailed per-question results (JSON)

Results are saved with timestamps to `results/` directory.
