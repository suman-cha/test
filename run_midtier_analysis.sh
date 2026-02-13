#!/bin/bash
# Run experiment with existing 5 mid-tier models and analyze

set -e

echo "========================================================================"
echo "RUNNING MID-TIER EXPERIMENT (5 agents) + ROW SUM ANALYSIS"
echo "========================================================================"
echo ""

# Configuration
NUM_QUESTIONS=${1:-5}
DATASET=${2:-math}
DIFFICULTY=${3:-hard}

echo "Configuration:"
echo "  Agents: 5 (mid-tier from agent_config.py)"
echo "  Questions: $NUM_QUESTIONS"
echo "  Dataset: $DATASET"
echo "  Difficulty: $DIFFICULTY"
echo ""

# Run experiment
echo "========================================================================"
echo "STEP 1: Running experiment..."
echo "========================================================================"
echo ""

python -m src.agents.run_experiment \
    --dataset $DATASET \
    --difficulty $DIFFICULTY \
    --num-questions $NUM_QUESTIONS \
    --num-agents 5 \
    --verbose \
    --output-dir results/midtier_pilot

echo ""
echo "========================================================================"
echo "STEP 2: Finding latest results..."
echo "========================================================================"
echo ""

# Find the latest results file
LATEST_RESULT=$(ls -t results/midtier_pilot/final_results_*.json 2>/dev/null | head -1)

if [ -z "$LATEST_RESULT" ]; then
    echo "❌ No results file found!"
    exit 1
fi

echo "Found: $LATEST_RESULT"
echo ""

# Run analysis
echo "========================================================================"
echo "STEP 3: Analyzing row sum vs correctness correlation..."
echo "========================================================================"
echo ""

python analyze_rowsum_correctness.py "$LATEST_RESULT"

echo ""
echo "========================================================================"
echo "COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: results/midtier_pilot/"
echo "Analysis plot: results/midtier_pilot/rowsum_correctness_analysis.png"
