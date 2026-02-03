#!/bin/bash
# Example: Compare raw_dataset vs raw_dataset_with_anon_labels for Zeroconf
#
# This example trains two Decision Tree agents:
# 1. Using raw_dataset (skips states with unlabeled actions)
# 2. Using raw_dataset_with_anon_labels (includes all states)
#
# Both are verified with model checking on the same configuration to compare performance.

CONFIG="N=1000,K=4,reset=true"

# ============================================================================
# Approach 1: Train Decision Tree with raw_dataset (original)
# ============================================================================
echo "=== Training Decision Tree with raw_dataset ==="
python cool_mc.py \
  --project_name="zeroconf_bc_dt_raw" \
  --algorithm="bc_decision_tree_agent" \
  --prism_dir="../prism_files" \
  --prism_file_path="zeroconf.prism" \
  --constant_definitions="${CONFIG}" \
  --seed=42 \
  --layers=0 \
  --neurons=1 \
  --num_episodes=1 \
  --prop="" \
  --behavioral_cloning="raw_dataset;../prism_files/zeroconf.prism;Pmax=? [ F (l=4 & ip=1) ];${CONFIG}" \
  --deploy=1

echo "=== Model checking Decision Tree (raw_dataset) ==="
# Note: Use Pmax=? for verification with deterministic policies to handle
# nondeterminism from multiple PRISM commands sharing the same action label
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="zeroconf_bc_dt_raw" \
  --prism_dir="../prism_files" \
  --prism_file_path="zeroconf.prism" \
  --constant_definitions="${CONFIG}" \
  --task="rl_model_checking" \
  --prop="Pmax=? [ F (l=4 & ip=1) ]" \
  --seed=42

# ============================================================================
# Approach 2: Train Decision Tree with raw_dataset_with_anon_labels (new)
# ============================================================================
echo ""
echo "=== Training Decision Tree with raw_dataset_with_anon_labels ==="
python cool_mc.py \
  --project_name="zeroconf_bc_dt_anon" \
  --algorithm="bc_decision_tree_agent" \
  --prism_dir="../prism_files" \
  --prism_file_path="zeroconf.prism" \
  --constant_definitions="${CONFIG}" \
  --seed=42 \
  --layers=0 \
  --neurons=1 \
  --num_episodes=1 \
  --prop="" \
  --behavioral_cloning="raw_dataset_with_anon_labels;../prism_files/zeroconf.prism;Pmax=? [ F (l=4 & ip=1) ];${CONFIG}" \
  --deploy=1

echo "=== Model checking Decision Tree (raw_dataset_with_anon_labels) ==="
# Note: Use Pmax=? for verification with deterministic policies
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="zeroconf_bc_dt_anon" \
  --prism_dir="../prism_files" \
  --prism_file_path="zeroconf.prism" \
  --constant_definitions="${CONFIG}" \
  --task="rl_model_checking" \
  --prop="Pmax=? [ F (l=4 & ip=1) ]" \
  --seed=42

echo ""
echo "=== Done! ==="
echo "Compare the model checking results above."
echo "Reference optimal: correct_max = 0.0000368"



# zeroconf.jani
#Converted from zeroconf.prism and zeroconf.props with Storm-conv version 1.2.4 (dev) using the following command:
#storm-conv --prism zeroconf.prism --tojani zeroconf.jani --prop zeroconf.props --globalvars
#Parameter settings:
#N = 20, K = 2, reset = true
#670 states 	(Storm)
#Reference results
#correct_max:	0.000020103281776956928	(Storm/exact)
#correct_min:	0.000002110327218406747	(Storm/exact)
