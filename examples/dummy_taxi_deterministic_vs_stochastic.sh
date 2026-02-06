#!/bin/bash
# Dummy taxi example: deterministic (DQN) vs stochastic (PPO) agent
# Trains each for 100 episodes, then runs model checking.

set -e

echo "============================================"
echo "Part 1: Deterministic DQN Agent"
echo "============================================"

# Train deterministic DQN agent (100 episodes)
echo "Training DQN agent (100 episodes)..."
python cool_mc.py \
    --project_name="dummy_taxi_det_vs_stoch" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter.prism" \
    --seed=128 \
    --algorithm="dqn_agent" \
    --num_episodes=100 \
    --max_steps=200

# Model checking on deterministic agent
echo "Model checking (deterministic agent)..."
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="dummy_taxi_det_vs_stoch" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"empty\" ]"

echo ""
echo "============================================"
echo "Part 2: Stochastic PPO Agent"
echo "============================================"

# Train stochastic PPO agent (100 episodes)
echo "Training stochastic PPO agent (100 episodes)..."
python cool_mc.py \
    --project_name="dummy_taxi_det_vs_stoch" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter.prism" \
    --seed=128 \
    --algorithm="stochastic_ppo_agent" \
    --num_episodes=100 \
    --max_steps=200 \
    --wrong_action_penalty=0

# Model checking on stochastic agent (will convert induced MDP to DTMC)
echo "Model checking (stochastic agent)..."
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="dummy_taxi_det_vs_stoch" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"empty\" ]"

echo ""
echo "============================================"
echo "Done!"
echo "============================================"
