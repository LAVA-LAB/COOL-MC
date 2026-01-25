#!/bin/bash

# Test script for optimal_control preprocessor
# This script:
# 1. Trains a dummy DQN agent for 101 episodes
# 2. Runs rl_model_checking with optimal_control preprocessor using different datasets
# 3. Verifies that Pmin=Pmax when using all_optimal_dataset (all actions are optimal)

echo ""
echo "=========================================="
echo "Optimal Control Preprocessor Test"
echo "=========================================="
echo ""

cd /workspaces/coolmc

PROJECT_NAME="test_optimal_control"
PRISM_FILE="transporter.prism"
CONSTANTS="MAX_JOBS=2,MAX_FUEL=10"
PRISM_PATH="../prism_files/${PRISM_FILE}"

# Step 1: Train a dummy DQN agent for 101 episodes
echo "=== Step 1: Training DQN agent for 101 episodes ==="
echo ""

python cool_mc.py \
  --project_name="${PROJECT_NAME}" \
  --task="safe_training" \
  --algorithm=dqn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="${PRISM_FILE}" \
  --constant_definitions="${CONSTANTS}" \
  --prop="Pmax=? [ F jobs_done=2 ]" \
  --num_episodes=101 \
  --eval_interval=100 \
  --seed=42

echo ""
echo "=== Step 2: rl_model_checking with raw_dataset (Pmax) ==="
echo "Using raw_dataset extracts ONE optimal action per state"
echo ""

python cool_mc.py \
  --project_name="${PROJECT_NAME}" \
  --parent_run_id="last" \
  --task="rl_model_checking" \
  --prism_dir="../prism_files" \
  --prism_file_path="${PRISM_FILE}" \
  --constant_definitions="${CONSTANTS}" \
  --preprocessor="optimal_control;raw_dataset;${PRISM_PATH};Pmax=? [ F jobs_done=2 ];${CONSTANTS}" \
  --prop="Pmax=? [ F jobs_done=2 ]" \
  --seed=42

echo ""
echo "=== Step 3: rl_model_checking with all_optimal_dataset (Pmax) ==="
echo "Using all_optimal_dataset extracts ALL optimal actions per state"
echo ""

python cool_mc.py \
  --project_name="${PROJECT_NAME}" \
  --parent_run_id="last" \
  --task="rl_model_checking" \
  --prism_dir="../prism_files" \
  --prism_file_path="${PRISM_FILE}" \
  --constant_definitions="${CONSTANTS}" \
  --preprocessor="optimal_control;all_optimal_dataset;${PRISM_PATH};Pmax=? [ F jobs_done=2 ];${CONSTANTS}" \
  --prop="Pmax=? [ F jobs_done=2 ]" \
  --seed=42

echo ""
echo "=== Step 4: rl_model_checking with all_optimal_dataset (Pmin) ==="
echo "If Pmin=Pmax, then all actions in the dataset are truly optimal"
echo ""

python cool_mc.py \
  --project_name="${PROJECT_NAME}" \
  --parent_run_id="last" \
  --task="rl_model_checking" \
  --prism_dir="../prism_files" \
  --prism_file_path="${PRISM_FILE}" \
  --constant_definitions="${CONSTANTS}" \
  --preprocessor="optimal_control;all_optimal_dataset;${PRISM_PATH};Pmax=? [ F jobs_done=2 ];${CONSTANTS}" \
  --prop="Pmin=? [ F jobs_done=2 ]" \
  --seed=42

echo ""
echo "=== Step 5: rl_model_checking with Rmin-optimal dataset (Pmax) ==="
echo "Rmin-optimal actions should also achieve Pmax=1"
echo ""

python cool_mc.py \
  --project_name="${PROJECT_NAME}" \
  --parent_run_id="last" \
  --task="rl_model_checking" \
  --prism_dir="../prism_files" \
  --prism_file_path="${PRISM_FILE}" \
  --constant_definitions="${CONSTANTS}" \
  --preprocessor="optimal_control;all_optimal_dataset;${PRISM_PATH};Rmin=? [ F jobs_done=2 ];${CONSTANTS}" \
  --prop="Pmax=? [ F jobs_done=2 ]" \
  --seed=42

echo ""
echo "=== Step 6: rl_model_checking with Rmin-optimal dataset (Pmin) ==="
echo "Rmin-optimal: Should have Pmin=Pmax=1 (all actions reach goal)"
echo ""

python cool_mc.py \
  --project_name="${PROJECT_NAME}" \
  --parent_run_id="last" \
  --task="rl_model_checking" \
  --prism_dir="../prism_files" \
  --prism_file_path="${PRISM_FILE}" \
  --constant_definitions="${CONSTANTS}" \
  --preprocessor="optimal_control;all_optimal_dataset;${PRISM_PATH};Rmin=? [ F jobs_done=2 ];${CONSTANTS}" \
  --prop="Pmin=? [ F jobs_done=2 ]" \
  --seed=42

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results interpretation:"
echo "- raw_dataset: Single optimal action per state -> deterministic policy"
echo "- all_optimal_dataset (Pmax): Multiple optimal actions -> permissive MDP"
echo "- all_optimal_dataset (Pmin vs Pmax):"
echo "    * If Pmin=Pmax=1: ALL actions in dataset guarantee reaching goal"
echo "    * If Pmin<Pmax: Some actions may lead to dead ends"
echo "- Rmin-optimal: Should have Pmin=Pmax=1 (efficient paths only)"
echo ""
