#!/bin/bash
# Quick Test Script - Run experiments on both GSM8K and MATH datasets
# Tests with 10 questions each to validate the system

set -e

echo "========================================"
echo "Quick Test: Multi-Dataset Validation"
echo "========================================"
echo ""

# Create results directory
mkdir -p results/quick_test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Testing GSM8K (10 harder questions, starting from index 800)..."
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --start-index 800 \
    --num-questions 10 \
    --output-dir results/quick_test/gsm8k_${TIMESTAMP} \
    --verbose

echo ""
echo "========================================"
echo ""

echo "Testing MATH (10 questions)..."
python -m src.agents.run_experiment \
    --dataset math \
    --num-questions 10 \
    --output-dir results/quick_test/math_${TIMESTAMP} \
    --verbose

echo ""
echo "========================================"
echo "Quick Test Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/quick_test/gsm8k_${TIMESTAMP}/"
echo "  - results/quick_test/math_${TIMESTAMP}/"
echo ""
