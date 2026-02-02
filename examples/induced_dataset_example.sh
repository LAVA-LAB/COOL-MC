#!/bin/bash
# Example: Behavioral Cloning from Induced DTMC
#
# This example demonstrates how to:
# 1. Train an initial agent (PPO with raw_dataset BC)
# 2. Use the induced_dataset to extract state-action pairs from the induced DTMC
# 3. Train a new BC agent using those state-action pairs
#
# The induced_dataset loads a previously trained agent, builds the induced DTMC
# via model checking, and extracts all reachable state-action pairs.

# Step 1: Train initial PPO agent with raw_dataset behavioral cloning
echo "=== Step 1: Training initial PPO agent with raw_dataset BC ==="
python cool_mc.py \
  --project_name="induced_dataset_example" \
  --algorithm="ppo_agent" \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
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
  --behavioral_cloning="raw_dataset;../prism_files/transporter_with_rewards.prism;Pmax=? [ F jobs_done=2 ];MAX_JOBS=2,MAX_FUEL=10" \
  --bc_epochs=65 \
  --deploy=1

# Step 2: Model check the trained agent to verify performance
echo "=== Step 2: Model checking the trained agent ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="induced_dataset_example" \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
  --task="rl_model_checking" \
  --prop="P=? [ F jobs_done=2 ]" \
  --seed=42

# Read the project name and run_id from last_run.txt for the induced_dataset config
# Format: project_name,run_id
LAST_RUN=$(cat last_run.txt)
PROJECT_NAME=$(echo $LAST_RUN | cut -d',' -f1)
RUN_ID=$(echo $LAST_RUN | cut -d',' -f2)

echo "=== Using trained agent from: $PROJECT_NAME / $RUN_ID ==="

# Step 3: Train a new BC agent using induced_dataset
# Config format: induced_dataset;project_name;run_id;property;constant_definitions
echo "=== Step 3: Training BC agent from induced DTMC ==="
python cool_mc.py \
  --project_name="induced_bc_example" \
  --algorithm=bc_nn_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
  --behavioral_cloning="induced_dataset;${PROJECT_NAME};${RUN_ID};P=? [ F jobs_done=2 ];MAX_JOBS=2,MAX_FUEL=10" \
  --bc_epochs=50 \
  --layers=3 \
  --neurons=64 \
  --lr=0.001 \
  --batch_size=32 \
  --num_episodes=1 \
  --prop="Pmax=? [ F jobs_done=2 ]" \
  --seed=42

# Step 4: Verify the BC agent via model checking
echo "=== Step 4: Model checking the BC agent ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="induced_bc_example" \
  --prism_dir="../prism_files" \
  --prism_file_path="transporter_with_rewards.prism" \
  --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
  --task="rl_model_checking" \
  --prop="P=? [ F jobs_done=2 ]" \
  --seed=42

echo "=== Done! ==="
