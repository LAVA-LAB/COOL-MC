#!/bin/bash
# Simple Dead-End State Analysis Example
# Demonstrates dead-end analysis on a smaller PRISM model

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Dead-End Analysis - Simple Example"
echo "=========================================="
echo ""
echo "This example analyzes the MDP structure to identify:"
echo "  1. Goal states (states with the target label)"
echo "  2. Dead-end states (cannot reach goal)"
echo "  3. Reachable states (can still reach goal)"
echo ""
echo "Note: Uses 'last' run for framework, but analysis is pure MDP"
echo "      structure (no trained policy involved)."
echo ""

# Example: Frozen Lake environment
echo "Running dead-end analysis on frozen_lake..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="frozen_lake" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="frozen_lake.prism" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop="Pmax=? [ F \"goal\" ]" \
    --interpreter="dead_end;Pmax=? [ F \"goal\" ];dead_end_frozen_lake.csv"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Check the output above for statistics including:"
echo "  - Total number of states in the MDP"
echo "  - Number of goal states"
echo "  - Number of dead-end states (% of total)"
echo "  - Number of reachable states"
echo ""
echo "Detailed results saved to:"
echo "  rl_model_checking/dead_end_frozen_lake.csv"
echo ""
