# Hammer-Spammer Model for LLM Agent Ranking

## 24-Hour Coding Test

### Problem Statement
Design an algorithm to recover the best answer from N LLM agents using pairwise comparisons, inspired by the Hammer-Spammer model and crowdsourcing algorithms.

### Project Structure
```
24h_hammer_spammer/
├── src/
│   ├── algorithm/       # Problem 1: SVD-based ranking algorithm
│   ├── agents/          # Problem 2: LLM agent system
│   ├── evaluation/      # Evaluation and comparison
│   └── utils/          # Utility functions
├── data/               # Datasets (GSM8K, MATH500)
├── results/            # Experiment results
├── logs/               # Execution logs
└── requirements.txt    # Dependencies
```

### Quick Start

1. **Setup Environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Run Algorithm Test (Problem 1)**
```bash
python src/algorithm/test_svd.py
```

3. **Run LLM Experiment (Problem 2)**
```bash
python src/agents/run_experiment.py
```

### References
- Karger et al. (2011) "Budget-optimal crowdsourcing using low-rank matrix approximations"

