#!/bin/bash
# Adaptive Testing - Run experiments while monitoring actual costs
# Start small, scale up based on actual performance

set -e

echo "========================================"
echo "Adaptive Testing Strategy"
echo "========================================"
echo ""

# Create results directory
mkdir -p results/adaptive_test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Phase 1: Quick validation (5 questions)"
echo "Testing system with minimal cost..."
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --num-questions 5 \
    --output-dir results/adaptive_test/phase1_${TIMESTAMP} \
    --no-parallel-generation \
    --verbose

echo ""
echo "✓ Phase 1 complete. Check results and actual cost."
echo "  If system works well, continue to Phase 2."
echo ""
read -p "Continue to Phase 2 (10 more questions)? [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Phase 2: Algorithm comparison (10 questions)"
    python -m src.agents.run_experiment \
        --dataset gsm8k \
        --num-questions 10 \
        --output-dir results/adaptive_test/phase2_${TIMESTAMP} \
        --no-parallel-generation \
        --verbose

    echo ""
    echo "✓ Phase 2 complete."
    echo ""
    read -p "Continue to Phase 3 (MATH dataset)? [y/N] " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Phase 3: MATH dataset test (10 questions)"
        python -m src.agents.run_experiment \
            --dataset math500 \
            --num-questions 10 \
            --difficulty hard \
            --output-dir results/adaptive_test/phase3_${TIMESTAMP} \
            --no-parallel-generation \
            --verbose

        echo ""
        echo "✓ Phase 3 complete."
    fi
fi

echo ""
echo "========================================"
echo "Adaptive Testing Complete!"
echo "========================================"
echo ""
echo "Total experiments run based on your budget decisions."
echo "Results saved to: results/adaptive_test/"
echo ""
