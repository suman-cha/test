#!/bin/bash
# Hard Test Script - Run experiments on harder problems
# Tests with 10 challenging questions to stress-test the system

set -e

echo "========================================"
echo "Hard Test: Challenging Problems"
echo "========================================"
echo ""

# Create results directory
mkdir -p results/hard_test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Testing GSM8K Hard Problems (questions 800-810)..."
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --start-index 800 \
    --num-questions 10 \
    --output-dir results/hard_test/gsm8k_hard_${TIMESTAMP} \
    --verbose

echo ""
echo "========================================"
echo ""

echo "Testing MATH Hard Problems (Level 4-5)..."
python -m src.agents.run_experiment \
    --dataset math \
    --difficulty hard \
    --num-questions 10 \
    --output-dir results/hard_test/math_hard_${TIMESTAMP} \
    --verbose

echo ""
echo "========================================"
echo "Hard Test Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/hard_test/gsm8k_hard_${TIMESTAMP}/"
echo "  - results/hard_test/math_hard_${TIMESTAMP}/"
echo ""
