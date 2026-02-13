#!/bin/bash
# Main Test Script - Challenging problems where CCRR outperforms Majority Voting
# Tests on harder problems to demonstrate CCRR's advantage
# Default: 50 questions per dataset (adjust based on your budget)

set -e

echo "========================================"
echo "Main Test: CCRR vs MV on Hard Problems"
echo "========================================"
echo ""
echo "Testing on challenging problems where:"
echo "  - Majority Voting likely fails"
echo "  - CCRR algorithm excels"
echo ""
echo "⚠️  This will consume API credits!"
echo "   Adjust --num-questions based on your budget"
echo ""

# Configurable parameters
NUM_QUESTIONS=${1:-50}  # Default 50, override with first argument

# Create results directory
mkdir -p results/main_test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Running GSM8K (${NUM_QUESTIONS} questions from hardest problems)..."
echo "  Using start-index 900 (multi-step reasoning problems)"
echo "  Track A: Testing spammer robustness (ratios: 0%, 10%, 20%, 40%, 60%, 80%, 100%)"
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --num-questions ${NUM_QUESTIONS} \
    --start-index 900 \
    --output-dir results/main_test/gsm8k_${TIMESTAMP} \
    --save-frequency 10 \
    --track-a \
    --spammer-ratios 0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
    --verbose

echo ""
echo "========================================"
echo ""

echo "Running MATH (${NUM_QUESTIONS} Level 4-5 hard questions)..."
echo "  Using difficulty filter: hard (Level 4-5 only)"
echo "  Track A: Testing spammer robustness (ratios: 0%, 10%, 20%, 40%, 60%, 80%, 100%)"
python -m src.agents.run_experiment \
    --dataset math \
    --num-questions ${NUM_QUESTIONS} \
    --difficulty hard \
    --output-dir results/main_test/math_${TIMESTAMP} \
    --save-frequency 10 \
    --track-a \
    --spammer-ratios 0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
    --verbose

echo ""
echo "========================================"
echo "Main Test Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/main_test/gsm8k_${TIMESTAMP}/ (GSM8K hardest problems, index 900+)"
echo "  - results/main_test/math_${TIMESTAMP}/ (MATH Level 4-5)"
echo ""
echo "To run with custom number of questions:"
echo "  ./run_main_test.sh 100  # runs 100 questions per dataset"
echo ""
echo "Why these problems?"
echo "  - MATH Level 4-5: Complex problems where weak models struggle"
echo "  - GSM8K 900+: Multi-step reasoning where weak models often fail"
echo "  → Weak models' low accuracy makes SVD-based weighting more effective"
echo ""
