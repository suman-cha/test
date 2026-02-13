#!/bin/bash
# Main Test Script - Process as many questions as API limits and budget allow
# Default: 50 questions per dataset (adjust based on your budget)

set -e

echo "========================================"
echo "Main Test: Full Validation"
echo "========================================"
echo ""
echo "⚠️  This will consume API credits!"
echo "   Adjust --num-questions based on your budget"
echo ""

# Configurable parameters
NUM_QUESTIONS=${1:-50}  # Default 50, override with first argument

# Create results directory
mkdir -p results/main_test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Running GSM8K (${NUM_QUESTIONS} questions)..."
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --num-questions ${NUM_QUESTIONS} \
    --output-dir results/main_test/gsm8k_${TIMESTAMP} \
    --save-frequency 10 \
    --verbose

echo ""
echo "========================================"
echo ""

echo "Running MATH (${NUM_QUESTIONS} questions)..."
python -m src.agents.run_experiment \
    --dataset math \
    --num-questions ${NUM_QUESTIONS} \
    --output-dir results/main_test/math_${TIMESTAMP} \
    --save-frequency 10 \
    --verbose

echo ""
echo "========================================"
echo "Main Test Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/main_test/gsm8k_${TIMESTAMP}/"
echo "  - results/main_test/math_${TIMESTAMP}/"
echo ""
echo "To run with custom number of questions:"
echo "  ./run_main_test.sh 100  # runs 100 questions per dataset"
echo ""
