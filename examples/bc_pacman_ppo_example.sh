#!/bin/bash
# Example: Behavioral Cloning with PPO Agent on Pacman
#
# This example demonstrates:
# 1. Training a PPO agent with behavioral cloning (MAXSTEPS=15)
# 2. Model checking the trained policy with a larger horizon (MAXSTEPS=20)
#
# The scaling parameter MAXSTEPS controls how many steps the pacman can take.

# ============================================================================
# Step 1: Train PPO agent with behavioral cloning (MAXSTEPS=15)
# ============================================================================
echo "=== Step 1: Training PPO agent with raw_dataset BC (MAXSTEPS=20) ==="
python cool_mc.py \
  --project_name="bc_pacman_ppo_example" \
  --algorithm="ppo_agent" \
  --prism_dir="../prism_files" \
  --prism_file_path="pacman.prism" \
  --constant_definitions="MAXSTEPS=15" \
  --seed=42 \
  --layers=4 \
  --neurons=256 \
  --lr=0.0003 \
  --batch_size=64 \
  --num_episodes=100 \
  --eval_interval=100 \
  --gamma=0.99 \
  --reward_flag=1 \
  --wrong_action_penalty=0 \
  --prop="" \
  --max_steps=500 \
  --behavioral_cloning="raw_dataset_with_anon_labels;../prism_files/pacman.prism;Pmin=? [ F \"Crash\" ];MAXSTEPS=20" \
  --bc_epochs=100 \
  --deploy=1

# ============================================================================
# Step 2: Model check the trained PPO policy (MAXSTEPS=15)
# ============================================================================
echo "=== Step 2: Model checking trained PPO agent (MAXSTEPS=20) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="bc_pacman_ppo_example" \
  --prism_dir="../prism_files" \
  --prism_file_path="pacman.prism" \
  --constant_definitions="MAXSTEPS=20" \
  --task="rl_model_checking" \
  --prop="P=? [ F \"Crash\" ]" \
  --seed=42

# ============================================================================
# Step 3: Model check the trained PPO policy (MAXSTEPS=20)
# ============================================================================
echo "=== Step 3: Model checking trained PPO agent (MAXSTEPS=60) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="bc_pacman_ppo_example" \
  --prism_dir="../prism_files" \
  --prism_file_path="pacman.prism" \
  --constant_definitions="MAXSTEPS=21" \
  --task="rl_model_checking" \
  --prop="P=? [ F \"Crash\" ]" \
  --seed=42

echo "=== Done! ==="
echo "Training was performed with MAXSTEPS=15"
echo "Model checking was performed with MAXSTEPS=15 (same as training) and MAXSTEPS=20 (larger horizon)"
