#!/bin/bash
# GSM8K Test - Test algorithms on grade school math problems
# Simpler problems for quick testing
#
# Usage:
#   bash run_gsm8k_test.sh        # Default: 10 questions
#   bash run_gsm8k_test.sh 5      # Run 5 questions
#   bash run_gsm8k_test.sh 20     # Run 20 questions

set -e

# Get number of questions from argument, default to 10
NUM_QUESTIONS=${1:-10}

echo "========================================"
echo "GSM8K Test (Grade School Math)"
echo "========================================"
echo ""
echo "Testing algorithms on GSM8K dataset"
echo "Simple arithmetic and word problems"
echo "Number of questions: ${NUM_QUESTIONS}"
echo ""

# Create results directory
mkdir -p results/gsm8k_test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Running GSM8K (${NUM_QUESTIONS} questions)..."
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --num-questions ${NUM_QUESTIONS} \
    --output-dir results/gsm8k_test/gsm8k_${TIMESTAMP} \
    --no-parallel-generation \
    --verbose

echo ""
echo "========================================"
echo "GSM8K Test Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/gsm8k_test/gsm8k_${TIMESTAMP}/"
echo ""
