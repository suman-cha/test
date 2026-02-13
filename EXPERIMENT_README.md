# Hammer-Spammer Model Validation Experiment (Problem 2)

## Overview

This experiment validates the Hammer-Spammer ranking algorithm (Problem 1) using a system of N=15 diverse LLM agents to answer mathematical reasoning questions from the GSM8K dataset.

## System Architecture

### Components

1. **Dataset Module** (`src/utils/dataset.py`)
   - Loads GSM8K/MATH500 datasets from Hugging Face
   - Extracts questions and ground truth answers
   - Normalizes answers for comparison

2. **Agent Configuration** (`src/agents/agent_config.py`)
   - Defines 15 diverse LLM agents using different models
   - 3 high-quality models (Strong Hammers)
   - 6 mid-tier models (Moderate Hammers)
   - 6 lower-tier models (Potential Spammers)

3. **LLM Agent Wrapper** (`src/agents/llm_agent.py`)
   - Unified interface for OpenRouter API
   - Answer generation
   - Pairwise comparisons

4. **Agent System** (`src/agents/agent_system.py`)
   - Orchestrates all N agents
   - Generates answers from all agents
   - Constructs N×N comparison matrix R

5. **Hammer-Spammer Algorithm** (`src/algorithm/hammer_spammer.py`)
   - SVD-based ranking algorithm from Problem 1
   - Computes quality scores q = (u₁ + v₁) / 2
   - Selects best answer

6. **Validation System** (`src/evaluation/validator.py`)
   - Ground truth validation
   - Oracle model (GPT-4o) validation

7. **Baseline Methods** (`src/evaluation/baselines.py`)
   - Majority voting
   - Random selection
   - Row/column sum baselines

8. **Results Analysis** (`src/evaluation/analysis.py`)
   - Statistical comparisons
   - Visualizations
   - Comprehensive reports

## Quick Start

### 1. Environment Setup

The environment should already be set up. Verify:

```bash
cd ~/24h_hammer_spammer
source venv/bin/activate
```

### 2. Test Individual Components

Test the dataset loader:
```bash
python -m src.utils.dataset
```

Test agent configuration:
```bash
python -m src.agents.agent_config
```

Test a single agent:
```bash
python -m src.agents.llm_agent
```

Test the algorithm:
```bash
python -m src.algorithm.hammer_spammer
```

### 3. Run Small-Scale Test (3 questions)

```bash
python -m src.agents.run_experiment --debug --verbose
```

This will:
- Use only 3 questions (for quick testing)
- Use all 15 agents
- Run both answer generation and comparisons
- Validate with ground truth
- Save results to `results/` directory

### 4. Run Full Experiment (50 questions)

```bash
python -m src.agents.run_experiment \
    --num-questions 50 \
    --dataset gsm8k \
    --num-agents 15 \
    --verbose
```

### 5. Run Full Experiment with Oracle Validation

```bash
python -m src.agents.run_experiment \
    --num-questions 50 \
    --dataset gsm8k \
    --num-agents 15 \
    --oracle-model openai/gpt-4o \
    --verbose
```

### 6. Analyze Results

After running the experiment:

```bash
python -m src.evaluation.analysis \
    --results results/final_results_TIMESTAMP.json \
    --output-dir results/analysis
```

This generates:
- Accuracy comparison plot
- Agent quality analysis plot
- Quality vs selection correlation plot
- Comprehensive text report

## Command-Line Options

### Run Experiment

```
python -m src.agents.run_experiment [OPTIONS]

Dataset Options:
  --dataset {gsm8k,math,math500}  Dataset to use (default: gsm8k)
  --split {train,test}            Dataset split (default: test)
  --num-questions N               Number of questions (default: 50)

Agent Options:
  --num-agents N                  Number of agents (default: 15)
  --no-parallel-generation        Disable parallel answer generation
  --no-parallel-comparison        Disable parallel comparisons

Algorithm Options:
  --beta FLOAT                    Quality gap parameter (default: 5.0)
  --epsilon FLOAT                 Minimum quality (default: 0.1)

Validation Options:
  --oracle-model MODEL            Oracle model (default: openai/gpt-4o)
  --no-oracle                     Disable oracle validation

Output Options:
  --output-dir DIR                Output directory (default: results)
  --save-frequency N              Save every N questions (default: 10)

Other Options:
  --verbose                       Enable verbose output
  --debug                         Debug mode (3 questions only)
```

### Analyze Results

```
python -m src.evaluation.analysis [OPTIONS]

Required:
  --results PATH                  Path to results JSON file

Optional:
  --output-dir DIR                Output directory for analysis
```

## Expected Results

### Success Criteria

The experiment is successful if:

1. **Algorithm Performance**: Hammer-Spammer accuracy ≥ Majority Voting accuracy
2. **Statistical Significance**: p-value < 0.05 in McNemar's test
3. **Oracle Agreement**: Oracle agreement rate > 50%
4. **Quality Differentiation**: Clear separation between hammers and spammers

### Sample Output

```
=============================================================
EXPERIMENT SUMMARY
=============================================================

Dataset: gsm8k
Number of questions: 50
Number of agents: 15
Total time: 1234.5s

=============================================================
ACCURACY RESULTS
=============================================================

Hammer-Spammer Algorithm: 76.00% (38/50)

Baseline Methods:
  majority_voting          : 72.00%
  row_sum                  : 74.00%
  column_sum               : 68.00%
  random                   : 50.00%
  first_answer             : 64.00%

=============================================================
ORACLE VALIDATION
=============================================================

Average oracle score: 7.8/10
Oracle agreement rate: 82.00%
```

## Budget Estimation

With N=15 agents and 50 questions:

- **Answer generation**: 15 × 50 × ~200 tokens = 150K tokens
- **Pairwise comparisons**: 15 × 14 × 50 × ~300 tokens = 3.15M tokens
- **Oracle validation**: 50 × ~400 tokens = 20K tokens
- **Total**: ~3.32M tokens

At ~$0.000015/token average: **~$50 total cost**

This is within the allocated budget.

## Troubleshooting

### API Rate Limits

If you hit rate limits:
1. The system has exponential backoff built in
2. Reduce parallelization: `--no-parallel-generation --no-parallel-comparison`
3. Reduce number of questions: `--num-questions 30`

### Out of Memory

If you run out of memory:
1. Reduce batch size by disabling parallelization
2. Use fewer agents: `--num-agents 10`

### API Key Issues

If API calls fail:
1. Verify API key: `cat .env | grep OPENROUTER_API_KEY`
2. Check balance at OpenRouter dashboard
3. Verify internet connection

## Output Files

The experiment generates:

1. **Results JSON** (`results/final_results_TIMESTAMP.json`)
   - Complete experiment data
   - All answers, comparisons, validations
   - Summary statistics

2. **Analysis Report** (`results/analysis/analysis_report.txt`)
   - Comprehensive text report
   - Statistical comparisons
   - Agent quality analysis

3. **Visualizations** (`results/analysis/*.png`)
   - `accuracy_comparison.png`: Bar chart of accuracies
   - `agent_quality.png`: Agent quality scores
   - `quality_vs_selection.png`: Correlation plot

## Next Steps

After completing the experiment:

1. Review the analysis report
2. Examine visualizations
3. Identify top-performing agents (hammers)
4. Verify statistical significance
5. Document findings for submission

## Implementation Checklist

- [x] Dataset loader (GSM8K)
- [x] Agent configuration (N=15)
- [x] LLM agent wrapper (OpenRouter API)
- [x] Agent system orchestration
- [x] Hammer-Spammer algorithm (SVD-based)
- [x] Validation system (Ground Truth + Oracle)
- [x] Baseline methods (Majority Voting, etc.)
- [x] Main experiment script
- [x] Results analysis and visualization
- [ ] Run small-scale test (3 questions)
- [ ] Run full experiment (50 questions)
- [ ] Generate analysis report
- [ ] Document findings

## Contact

For issues or questions, refer to the 24-hour coding test instructions.
