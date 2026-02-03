#!/bin/bash
# Example: Behavioral Cloning from Induced DTMC
#
# This example demonstrates how to:
# 1. Train an initial agent (PPO with raw_dataset BC)
# 2. Use the induced_dataset to extract state-action pairs from the induced DTMC
# 3. Train new BC agents (Neural Network and Decision Tree) using those state-action pairs
#
# The induced_dataset loads a previously trained agent, builds the induced DTMC
# via model checking, and extracts all reachable state-action pairs.

# ============================================================================
# Pipeline for jobs_done=5
# ============================================================================

# Step 1: Train initial PPO agent with raw_dataset behavioral cloning
echo "=== Step 1: Training initial PPO agent with raw_dataset BC (jobs_done=5) ==="
python cool_mc.py \
  --project_name="induced_dataset_example_5jobs" \
  --algorithm="ppo_agent" \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=5,MAX_FUEL=10" \
  --seed=128 \
  --layers=4 \
  --neurons=512 \
  --lr=0.0003 \
  --batch_size=64 \
  --num_episodes=100 \
  --eval_interval=100 \
  --gamma=0.99 \
  --reward_flag=1 \
  --wrong_action_penalty=0 \
  --prop="" \
  --max_steps=1000 \
  --behavioral_cloning="raw_dataset;../prism_files/transporter_with_rewards.prism;Pmax=? [ F jobs_done=5 ];MAX_JOBS=5,MAX_FUEL=10" \
  --bc_epochs=65 \
  --deploy=1

# Step 2: Model check the trained PPO agent to verify performance
echo "=== Step 2: Model checking the trained PPO agent (jobs_done=5) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="induced_dataset_example_5jobs" \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=5,MAX_FUEL=10" \
  --task="rl_model_checking" \
  --prop="P=? [ F jobs_done=5 ]" \
  --seed=42

# Read the project name and run_id from last_run.txt for the induced_dataset config
LAST_RUN=$(cat last_run.txt)
PROJECT_NAME=$(echo $LAST_RUN | cut -d',' -f1)
RUN_ID=$(echo $LAST_RUN | cut -d',' -f2)

echo "=== Using trained agent from: $PROJECT_NAME / $RUN_ID ==="

# Step 3a: Train a BC Neural Network agent using induced_dataset
echo "=== Step 3a: Training BC NN agent from induced DTMC (jobs_done=5) ==="
python cool_mc.py \
  --project_name="induced_bc_nn_example_5jobs" \
  --algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=5,MAX_FUEL=10" \
  --behavioral_cloning="induced_dataset;${PROJECT_NAME};${RUN_ID};P=? [ F jobs_done=5 ];MAX_JOBS=5,MAX_FUEL=10" \
  --bc_epochs=50 \
  --layers=3 \
  --neurons=64 \
  --lr=0.001 \
  --batch_size=32 \
  --num_episodes=1 \
  --prop="Pmax=? [ F jobs_done=5 ]" \
  --seed=42

# Step 3a-verify: Model check the BC NN agent
echo "=== Step 3a-verify: Model checking the BC NN agent (jobs_done=5) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="induced_bc_nn_example_5jobs" \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=5,MAX_FUEL=10" \
  --task="rl_model_checking" \
  --prop="P=? [ F jobs_done=5 ]" \
  --seed=42

# Step 3b: Train a BC Decision Tree agent using induced_dataset (interpretable!)
# For decision tree: --layers=max_depth (0=unlimited), --neurons=min_samples_leaf
echo "=== Step 3b: Training BC Decision Tree agent from induced DTMC (jobs_done=5) ==="
python cool_mc.py \
  --project_name="induced_bc_dt_example_5jobs" \
  --algorithm=bc_decision_tree_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=5,MAX_FUEL=10" \
  --behavioral_cloning="induced_dataset;${PROJECT_NAME};${RUN_ID};P=? [ F jobs_done=5 ];MAX_JOBS=5,MAX_FUEL=10" \
  --layers=0 \
  --neurons=1 \
  --num_episodes=1 \
  --prop="Pmax=? [ F jobs_done=5 ]" \
  --seed=42

# Step 3b-verify: Model check the BC Decision Tree agent
echo "=== Step 3b-verify: Model checking the BC Decision Tree agent (jobs_done=5) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="induced_bc_dt_example_5jobs" \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=5,MAX_FUEL=10" \
  --task="rl_model_checking" \
  --prop="P=? [ F jobs_done=5 ]" \
  --seed=42

echo "=== Done! ==="
