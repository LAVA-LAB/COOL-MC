#!/bin/bash
# Dead-End State Analysis Example
# Analyzes the MDP structure to find states from which goal states
# cannot be reached, regardless of actions taken.
# Note: Loads last training run for framework compatibility, but the
# dead-end analysis itself only analyzes the MDP structure (no policy used).

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Dead-End State Analysis Example"
echo "=========================================="

# Example 1: ICU Sepsis - Find states that cannot reach survival
echo ""
echo "Example 1: ICU Sepsis - Dead-end analysis for survival..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_ppo" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop="Pmin=? [ F \"survival\" ]" \
    --interpreter="dead_end;Pmin=? [ F \"survival\" ];icu_sepsis_dead_end_survival.csv"

echo ""
echo "=========================================="
echo "Dead-End Analysis Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - CSV: rl_model_checking/icu_sepsis_dead_end_survival.csv"
echo ""
echo "The CSV contains all states classified as:"
echo "  - 'goal': States labeled with the goal (e.g., survival)"
echo "  - 'dead_end': States that cannot reach the goal"
echo "  - 'reachable': States that can still reach the goal"
echo ""
