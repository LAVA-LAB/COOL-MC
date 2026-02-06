#!/bin/bash
# ICU Sepsis Behavioral Cloning with Decision Tree
# Trains a BC decision tree to maximize survival probability,
# then performs model checking separately.

cd "$(dirname "$0")/.."

echo "=========================================="
echo "ICU Sepsis BC Decision Tree Training & Verification"
echo "=========================================="

# Step 1: Train BC Decision Tree agent
echo ""
echo "Phase 1: Training BC Decision Tree Agent..."
echo "------------------------------------------"

python cool_mc.py \
    --project_name="icu_sepsis_bc_dt" \
    --algorithm=bc_decision_tree_agent \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --constant_definitions="" \
    --behavioral_cloning="raw_dataset;../prism_files/icu_sepsis.prism;Pmin=? [ F \"survival\" ];" \
    --layers=0 \
    --neurons=1 \
    --num_episodes=1 \
    #--prop="Pmax=? [ F \"survival\" ]" \
    --reward_flag=1 \
    --seed=128


# Step 2: Feature Importance Ranking - Survival Reachability
echo ""
echo "Phase 2: Feature Importance Ranking - Survival..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_bc_dt" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"survival\" ]" \
    --interpreter="feature_importance_ranking"

echo ""
echo "=========================================="
echo "ICU Sepsis BC Decision Tree Complete!"
echo "=========================================="
