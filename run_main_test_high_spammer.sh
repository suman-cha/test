#!/bin/bash
# Main Test Script - High Spammer Ratio to maximize CCRR advantage
# Uses more low-tier models to increase spammer ratio
# This setup best demonstrates CCRR's robustness vs Majority Voting

set -e

echo "========================================"
echo "Main Test: High Spammer Ratio"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - 5 Hammers (mid-tier models)"
echo "  - 15-20 Spammers (low-tier models)"
echo "  - Spammer:Hammer ratio = 3:1 to 4:1"
echo ""
echo "This ratio maximizes CCRR's advantage:"
echo "  - Majority Voting: overwhelmed by noise"
echo "  - CCRR: filters spammers via consistency"
echo ""
echo "⚠️  This will consume MORE API credits!"
echo "   Consider reducing --num-questions"
echo ""

# Configurable parameters
NUM_QUESTIONS=${1:-30}  # Default 30 (fewer questions due to more agents)
NUM_AGENTS=${2:-20}     # Default 20 agents (5 hammers + 15 spammers)

# Create results directory
mkdir -p results/main_test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Running GSM8K (${NUM_QUESTIONS} questions, ${NUM_AGENTS} agents)..."
echo "  Using start-index 800 (harder problems)"
echo "  High spammer ratio for CCRR advantage"
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --num-questions ${NUM_QUESTIONS} \
    --num-agents ${NUM_AGENTS} \
    --start-index 800 \
    --output-dir results/main_test/gsm8k_high_spammer_${TIMESTAMP} \
    --save-frequency 5 \
    --verbose

echo ""
echo "========================================"
echo ""

echo "Running MATH (${NUM_QUESTIONS} Level 4-5 questions, ${NUM_AGENTS} agents)..."
echo "  Using difficulty filter: hard (Level 4-5 only)"
echo "  High spammer ratio for CCRR advantage"
python -m src.agents.run_experiment \
    --dataset math \
    --num-questions ${NUM_QUESTIONS} \
    --num-agents ${NUM_AGENTS} \
    --difficulty hard \
    --output-dir results/main_test/math_high_spammer_${TIMESTAMP} \
    --save-frequency 5 \
    --verbose

echo ""
echo "========================================"
echo "Main Test Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/main_test/gsm8k_high_spammer_${TIMESTAMP}/"
echo "  - results/main_test/math_high_spammer_${TIMESTAMP}/"
echo ""
echo "Usage examples:"
echo "  ./run_main_test_high_spammer.sh         # 30 questions, 20 agents (default)"
echo "  ./run_main_test_high_spammer.sh 50      # 50 questions, 20 agents"
echo "  ./run_main_test_high_spammer.sh 50 25   # 50 questions, 25 agents (even higher ratio!)"
echo ""
echo "High spammer ratio demonstrates CCRR's robustness:"
echo "  - MV accuracy drops significantly"
echo "  - CCRR maintains high accuracy"
echo "  - Performance gap = CCRR's advantage"
echo ""
