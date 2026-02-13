#!/bin/bash
# Hard Problems Test - Test algorithms on difficult problems only
# MATH Level 4-5 only (hardest problems)

set -e

echo "========================================"
echo "Hard Problems Test (Level 4-5 only)"
echo "========================================"
echo ""
echo "Testing algorithms on the most difficult problems"
echo "from MATH-500 dataset (Level 4-5)"
echo ""

# Create results directory
mkdir -p results/hard_test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Running MATH-500 Hard Problems (10 questions, Level 4-5)..."
python -m src.agents.run_experiment \
    --dataset math500 \
    --num-questions 10 \
    --difficulty hard \
    --output-dir results/hard_test/math500_hard_${TIMESTAMP} \
    --no-parallel-generation \
    --verbose

echo ""
echo "========================================"
echo "Hard Problems Test Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/hard_test/math_hard_${TIMESTAMP}/"
echo ""
echo "This tests the algorithms' ability to handle"
echo "the most challenging mathematical problems."
echo ""
