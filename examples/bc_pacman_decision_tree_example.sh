#!/bin/bash
# Example: Behavioral Cloning with Decision Tree Agent on Pacman
#
# This example demonstrates:
# 1. Training a Decision Tree agent with behavioral cloning (MAXSTEPS=15)
# 2. Model checking the trained policy with a larger horizon (MAXSTEPS=20)
#
# The scaling parameter MAXSTEPS controls how many steps the pacman can take.

# ============================================================================
# Step 1: Train Decision Tree agent with behavioral cloning (MAXSTEPS=15)
# ============================================================================
echo "=== Step 1: Training Decision Tree agent with raw_dataset BC (MAXSTEPS=20) ==="
python cool_mc.py \
  --project_name="bc_pacman_decision_tree_example" \
  --algorithm="bc_decision_tree_agent" \
  --prism_dir="../prism_files" \
  --prism_file_path="pacman.prism" \
  --constant_definitions="MAXSTEPS=15" \
  --seed=42 \
  --layers=0 \
  --neurons=1 \
  --num_episodes=1 \
  --prop="" \
  --behavioral_cloning="raw_dataset_with_anon_labels;../prism_files/pacman.prism;Pmin=? [ F \"Crash\" ];MAXSTEPS=20" \
  --deploy=1

# ============================================================================
# Step 2: Model check the trained Decision Tree policy (MAXSTEPS=20)
# ============================================================================
echo "=== Step 2: Model checking trained Decision Tree agent (MAXSTEPS=20) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="bc_pacman_decision_tree_example" \
  --prism_dir="../prism_files" \
  --prism_file_path="pacman.prism" \
  --constant_definitions="MAXSTEPS=20" \
  --task="rl_model_checking" \
  --prop="P=? [ F \"Crash\" ]" \
  --seed=42

# ============================================================================
# Step 3: Model check the trained Decision Tree policy (MAXSTEPS=60)
# ============================================================================
echo "=== Step 3: Model checking trained Decision Tree agent (MAXSTEPS=60) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="bc_pacman_decision_tree_example" \
  --prism_dir="../prism_files" \
  --prism_file_path="pacman.prism" \
  --constant_definitions="MAXSTEPS=60" \
  --task="rl_model_checking" \
  --prop="P=? [ F \"Crash\" ]" \
  --seed=42

echo "=== Done! ==="
echo "Training was performed with MAXSTEPS=15"
echo "Model checking was performed with MAXSTEPS=20 and MAXSTEPS=60 (larger horizons)"
