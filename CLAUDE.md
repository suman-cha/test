# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Agent Ranking system implementing and comparing algorithms (CCRR, SVD, SWRA v2) that identify high-quality "Hammer" agents vs. low-quality "Spammer" agents and select the best answer from N=15 diverse LLM agents. Built in Python 3.11+.

## Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Experiments
```bash
# Quick test (10 questions)
python -m src.agents.run_experiment --dataset gsm8k --num-questions 10 --verbose

# MATH with difficulty filter
python -m src.agents.run_experiment --dataset math --num-questions 10 --difficulty hard --verbose

# Main test (50 questions per dataset, configurable)
./run_main_test.sh          # default 50
./run_main_test.sh 100      # custom count
```

### Run Tests
```bash
pytest
```

### Key CLI Arguments for `run_experiment`
`--dataset` (gsm8k, math, math500), `--num-questions`, `--difficulty` (easy/medium/hard, MATH only), `--start-index`, `--num-agents` (default 15), `--beta` (Bradley-Terry param, default 5.0), `--epsilon` (spammer probability, default 0.1), `--oracle-model` (default openai/gpt-4o), `--no-oracle`, `--output-dir`, `--save-frequency`, `--verbose`, `--debug` (3 questions only).

## Architecture

### Pipeline Flow
Question → **Generate Answers** (N=15 agents in parallel) → **Construct Comparison Matrix** R ∈ {±1}^(N×N) (hammers use real LLM comparisons, spammers use random ±1) → **Apply Ranking Algorithms** (SVD, CCRR, SWRA v2) → **Validate** (ground truth + GPT-4o oracle) → **Store Results** (timestamped JSON).

### Module Layout

- **`src/agents/`** — Agent orchestration
  - `agent_config.py`: Defines 15 agents (5 mid-tier hammers, 10 low-tier spammers) with model names and OpenRouter endpoints
  - `llm_agent.py`: LLM API wrapper using OpenRouter; handles answer generation and pairwise comparison with retry logic
  - `agent_system.py`: Orchestrates all agents — parallel answer generation (2 workers), comparison matrix construction (8 workers for hammer rows, random for spammer rows)
  - `run_experiment.py`: Main entry point (`ExperimentRunner` class); orchestrates the full pipeline, computes summary statistics, correlation analysis, and saves results

- **`src/algorithm/`** — Ranking algorithms
  - `hammer_spammer.py`: SVD-based ranking — rank-1 SVD of R, quality scores q_i = (u₁[i] + v₁[i]) / 2, selects argmax
  - `ccrr.py`: Cross-Consistency Robust Ranking — Phase 1: row-norm spammer detection via agreement matrix; Phase 2: EM refinement (Bradley-Terry MLE + weight updates); Phase 3: best answer selection
  - `swra.py`: Spectral Weighted Rank Aggregation v2 — spectral reliability estimation using row/column signals from top-k SVD

- **`src/evaluation/`** — Validation and baselines
  - `validator.py`: Ground truth validation (exact/numerical matching) and oracle validation (GPT-4o scoring 1-10)
  - `baselines.py`: Majority voting, weighted majority voting, best single model, random selection
  - `analysis.py`, `comparison_analysis.py`, `compare_experiments.py`: Results analysis, metrics, and multi-experiment comparison
  - `visualize_results.py`: Plotting utilities

- **`src/utils/`** — Dataset loading
  - `dataset.py`: Loads GSM8K, MATH (full), MATH-500 from HuggingFace `datasets`; supports difficulty filtering (easy=L1-2, medium=L3, hard=L4-5) and start-index offset

### Key Design Decisions
- All LLM calls go through OpenRouter API (requires `OPENROUTER_API_KEY` in `.env`)
- Temperature fixed at 0.0 for reproducibility; max 500 tokens per response
- Comparison matrix diagonal is always 1 (R_ii = 1)
- Results saved as timestamped JSON to `results/` directory

## Environment Setup

Requires `OPENROUTER_API_KEY` in a `.env` file at project root:
```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
```
