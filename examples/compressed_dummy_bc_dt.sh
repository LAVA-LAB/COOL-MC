#!/bin/bash
# Compressed Dummy Behavioral Cloning with Decision Tree
# This script trains a behavioral cloning decision tree agent using
# the compressed_dummy.prism model with decompressed state features

echo "=========================================="
echo "Compressed Dummy BC Decision Tree Training & Verification"
echo "=========================================="

# Step 1: Train BC Decision Tree agent using raw_dataset
echo ""
echo "Phase 1: Training BC Decision Tree Agent..."
echo "------------------------------------------"

python cool_mc.py \
    --project_name="compressed_dummy_bc_dt" \
    --algorithm=bc_decision_tree_agent \
    --prism_dir="../prism_files" \
    --prism_file_path="compressed_dummy.prism" \
    --constant_definitions="" \
    --behavioral_cloning="raw_dataset;../prism_files/compressed_dummy.prism;Pmax=? [ F \"goal\" ];" \
    --layers=0 \
    --neurons=1 \
    --num_episodes=1 \
    --prop="Pmax=? [ F \"goal\" ]" \
    --reward_flag=1 \
    --seed=128

echo ""
echo "Phase 2: Model Checking - Goal Reachability with Interpretation..."
echo "------------------------------------------"

# Verification Phase - Probability of reaching goal state with decision tree interpretation
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="compressed_dummy_bc_dt" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="compressed_dummy.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"goal\" ]" \
    --interpreter="decision_tree"

echo ""
echo "Phase 3: Model Checking - Bad State Reachability with Interpretation..."
echo "------------------------------------------"

# Verification Phase - Probability of reaching bad state with decision tree interpretation
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="compressed_dummy_bc_dt" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="compressed_dummy.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"bad\" ]" \
    --interpreter="decision_tree"

echo ""
echo "=========================================="
echo "Compressed Dummy BC Decision Tree Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Trained interpretable Decision Tree using raw_dataset"
echo "  - Decision tree learned from optimal state-action pairs"
echo "  - Used decompressed state features for richer representation"
echo "  - Verified probability of reaching goal state"
echo "  - Verified probability of reaching bad state"
echo ""
