#!/bin/bash
# Spammer Ratio 실험 자동화 스크립트
# 3개 epsilon (0.1, 0.3, 0.5) × 다양한 데이터셋

set -e

echo "========================================"
echo "Spammer Ratio Experiments"
echo "Testing ε = 0.1, 0.3, 0.5"
echo "========================================"
echo ""

# Create results directory
mkdir -p results/spammer_experiments
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configuration
EPSILONS=(0.1 0.3 0.5)
QUESTIONS=30  # Standard: 30 questions per experiment

echo "Configuration:"
echo "  Questions per experiment: ${QUESTIONS}"
echo "  Spammer ratios: ${EPSILONS[@]}"
echo "  Estimated total cost: \$30-45"
echo "  Estimated time: 2-3 hours"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Experiment counter
EXPERIMENT=0
TOTAL=12  # 3 epsilons × 4 dataset combinations

# ============================================
# Phase 2: Main Experiments (30 questions)
# ============================================

echo ""
echo "========================================"
echo "Phase 2: Main Experiments"
echo "========================================"
echo ""

# Dataset configurations
declare -A DATASETS
DATASETS["gsm8k_easy"]="--dataset gsm8k --start-index 0"
DATASETS["gsm8k_hard"]="--dataset gsm8k --start-index 800"
DATASETS["math_medium"]="--dataset math --difficulty medium"
DATASETS["math_hard"]="--dataset math --difficulty hard"

# Run experiments
for DATASET_NAME in "${!DATASETS[@]}"; do
    DATASET_ARGS=${DATASETS[$DATASET_NAME]}
    
    for EPSILON in "${EPSILONS[@]}"; do
        EXPERIMENT=$((EXPERIMENT + 1))
        
        echo ""
        echo "----------------------------------------"
        echo "Experiment ${EXPERIMENT}/${TOTAL}"
        echo "Dataset: ${DATASET_NAME}"
        echo "Epsilon: ${EPSILON}"
        echo "----------------------------------------"
        
        OUTPUT_DIR="results/spammer_experiments/${DATASET_NAME}_eps$(echo $EPSILON | tr '.' '_')_${TIMESTAMP}"
        
        python -m src.agents.run_experiment \
            ${DATASET_ARGS} \
            --num-questions ${QUESTIONS} \
            --epsilon ${EPSILON} \
            --beta 5.0 \
            --num-agents 15 \
            --output-dir ${OUTPUT_DIR} \
            --save-frequency 10 \
            --verbose
        
        echo "Completed: ${OUTPUT_DIR}"
        
        # Small delay to avoid rate limiting
        sleep 5
    done
done

echo ""
echo "========================================"
echo "All Experiments Complete!"
echo "========================================"
echo ""
echo "Results saved to: results/spammer_experiments/"
echo ""
echo "Next steps:"
echo "1. Run analysis:"
echo "   python -m src.evaluation.compare_experiments --results-dir results/spammer_experiments/"
echo ""
echo "2. Generate plots:"
echo "   python -m src.evaluation.visualize_all --results-dir results/spammer_experiments/"
echo ""

# Save experiment log
cat > results/spammer_experiments/experiment_log_${TIMESTAMP}.txt << LOGEOF
Spammer Ratio Experiments
Timestamp: ${TIMESTAMP}

Configuration:
- Questions per experiment: ${QUESTIONS}
- Spammer ratios: ${EPSILONS[@]}
- Datasets: ${!DATASETS[@]}
- Total experiments: ${TOTAL}

Completed: ${EXPERIMENT}/${TOTAL}
LOGEOF

echo "Experiment log saved to: results/spammer_experiments/experiment_log_${TIMESTAMP}.txt"
