#!/bin/bash
# Main Test Script - Spectral Ranking vs Majority Voting on Hard Problems
#
# 실험 스토리:
#   3 strong hammers (Sonnet, Gemini Pro, GPT-4 Turbo)의 신호를
#   SVD가 자동으로 포착하여 12개 약한 모델의 노이즈를 걸러내는 반면,
#   Majority Voting은 신호가 노이즈에 희석된다.
#
# Algorithms tested: SVD Spectral Ranking, CCRR, SWRA vs Majority Voting
# Default: 50 questions per dataset (adjust based on your budget)

set -e

echo "========================================"
echo "Main Test: Spectral Ranking vs Baselines"
echo "========================================"
echo ""
echo "Agent configuration (N=10):"
echo "  HIGH-TIER (2): Claude Sonnet, Gemini Pro"
echo "  MID-TIER  (3): GPT-4o-mini, Haiku, Flash"
echo "  LOW-TIER  (5): Llama 8B/3B, Mistral 7B, Gemma 9B, Llama-3-8B"
echo ""
echo "Algorithms: SVD Spectral, CCRR, SWRA vs Majority Voting"
echo ""
echo "Testing on challenging problems where:"
echo "  - Strong hammers provide reliable signal for SVD"
echo "  - Majority Voting gets diluted by weak model noise"
echo ""
echo "Warning: This will consume API credits!"
echo "  Adjust --num-questions based on your budget"
echo ""

# Configurable parameters
NUM_QUESTIONS=${1:-50}  # Default 50, override with first argument
BETA=${2:-5.0}          # Bradley-Terry sharpness
EPSILON=${3:-0.1}       # Spammer probability

# Create results directory
mkdir -p results/main_test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Parameters: num_questions=${NUM_QUESTIONS}, beta=${BETA}, epsilon=${EPSILON}"
echo ""

echo "Running GSM8K (${NUM_QUESTIONS} questions from harder problems)..."
echo "  Using start-index 800 (later problems are harder)"
python -m src.agents.run_experiment \
    --dataset gsm8k \
    --num-questions ${NUM_QUESTIONS} \
    --start-index 800 \
    --beta ${BETA} \
    --epsilon ${EPSILON} \
    --output-dir results/main_test/gsm8k_${TIMESTAMP} \
    --save-frequency 10 \
    --verbose

echo ""
echo "========================================"
echo ""

echo "Running MATH (${NUM_QUESTIONS} Level 4-5 hard questions)..."
echo "  Using difficulty filter: hard (Level 4-5 only)"
python -m src.agents.run_experiment \
    --dataset math \
    --num-questions ${NUM_QUESTIONS} \
    --difficulty hard \
    --beta ${BETA} \
    --epsilon ${EPSILON} \
    --output-dir results/main_test/math_${TIMESTAMP} \
    --save-frequency 10 \
    --verbose

echo ""
echo "========================================"
echo "Main Test Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - results/main_test/gsm8k_${TIMESTAMP}/ (hard GSM8K problems, index 800+)"
echo "  - results/main_test/math_${TIMESTAMP}/ (MATH Level 4-5)"
echo ""
echo "To run with custom parameters:"
echo "  ./run_main_test.sh <num_questions> <beta> <epsilon>"
echo "  ./run_main_test.sh 100          # 100 questions, default beta/epsilon"
echo "  ./run_main_test.sh 50 3.0 0.2   # 50 questions, beta=3.0, epsilon=0.2"
echo ""
