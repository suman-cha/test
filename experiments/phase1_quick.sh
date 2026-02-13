#!/bin/bash
# Phase 1: Quick Validation (10 questions)
# 빠르게 시스템 동작 확인

set -e

echo "========================================"
echo "Phase 1: Quick Validation"
echo "6 experiments × 10 questions"
echo "Estimated cost: ~$3"
echo "Estimated time: ~30 minutes"
echo "========================================"
echo ""

mkdir -p results/phase1_quick
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EPSILONS=(0.1 0.3 0.5)

# Test 1: GSM8K Easy
echo "Test 1/6: GSM8K Easy"
for EPSILON in "${EPSILONS[@]}"; do
    python -m src.agents.run_experiment \
        --dataset gsm8k \
        --start-index 0 \
        --num-questions 10 \
        --epsilon ${EPSILON} \
        --output-dir results/phase1_quick/gsm8k_easy_eps$(echo $EPSILON | tr '.' '_') \
        --verbose
done

# Test 2: MATH Hard
echo "Test 2/6: MATH Hard"
for EPSILON in "${EPSILONS[@]}"; do
    python -m src.agents.run_experiment \
        --dataset math \
        --difficulty hard \
        --num-questions 10 \
        --epsilon ${EPSILON} \
        --output-dir results/phase1_quick/math_hard_eps$(echo $EPSILON | tr '.' '_') \
        --verbose
done

echo ""
echo "Phase 1 Complete!"
echo "Results: results/phase1_quick/"
