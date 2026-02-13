#!/bin/bash
# Quick test script for Problem 2 implementation
# This runs a minimal test to verify all components work

set -e  # Exit on error

echo "========================================"
echo "Problem 2 - Quick Test Script"
echo "========================================"
echo ""

# Navigate to project directory
cd ~/24h_hammer_spammer

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "========================================"
echo "Step 1: Testing Dataset Loader"
echo "========================================"
python -m src.utils.dataset

echo ""
echo "========================================"
echo "Step 2: Testing Agent Configuration"
echo "========================================"
python -m src.agents.agent_config

echo ""
echo "========================================"
echo "Step 3: Testing Hammer-Spammer Algorithm"
echo "========================================"
python -m src.algorithm.hammer_spammer

echo ""
echo "========================================"
echo "Step 4: Testing Baseline Methods"
echo "========================================"
python -m src.evaluation.baselines

echo ""
echo "========================================"
echo "Step 5: Running Mini Experiment (3 questions, debug mode)"
echo "========================================"
python -m src.agents.run_experiment --debug --verbose

echo ""
echo "========================================"
echo "All Tests Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review the debug output above"
echo "2. Check results in results/ directory"
echo "3. If everything looks good, run full experiment:"
echo "   python -m src.agents.run_experiment --num-questions 50 --verbose"
echo ""
