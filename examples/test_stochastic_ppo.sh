#!/bin/bash
# Test script for Stochastic PPO Agent with Behavioral Cloning
#
# This script demonstrates how to:
# 1. Use behavioral cloning to quickly learn an optimal stochastic policy
# 2. Run model checking on the stochastic agent
# 3. The model checker will automatically detect the stochastic agent
#    and convert the induced MDP to a DTMC
#
# The stochastic agent uses action probability distributions instead of
# deterministic action selection. During model checking:
# - All actions with probability > 0 are expanded during building
# - Non-determinism is resolved by combining transition probabilities
#   weighted by action probabilities: P(s'|s) = Σ_a P(a|s) * P(s'|s,a)
# - The result is a DTMC (Discrete-Time Markov Chain) instead of an MDP

echo "============================================"
echo "Stochastic PPO Agent - Behavioral Cloning"
echo "============================================"

# Step 1: Use behavioral cloning to quickly get an optimal stochastic policy
# The behavioral_cloning parameter generates an optimal dataset from the PRISM model
# and trains the stochastic PPO agent on it
#
# Note: wrong_action_penalty=0 for stochastic policies because:
#   - Stochastic sampling may pick unavailable actions (expected behavior)
#   - Environment substitutes with available action automatically
#   - No need to penalize the inherent randomness of the policy
echo ""
echo "Step 1: Training Stochastic PPO Agent with Behavioral Cloning..."
echo "  (Using optimal policy from PRISM model as training data)"
python cool_mc.py \
    --project_name="test_stochastic_ppo" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter_with_rewards.prism" \
    --seed=128 \
    --algorithm="stochastic_ppo_agent" \
    --behavioral_cloning="raw_dataset;../prism_files/transporter_with_rewards.prism;Pmax=? [ F jobs_done=2 ];MAX_JOBS=2,MAX_FUEL=10" \
    --bc_epochs=50 \
    --layers=3 \
    --neurons=128 \
    --lr=0.001 \
    --batch_size=32 \
    --num_episodes=1000 \
    --eval_interval=5 \
    --gamma=0.99 \
    --reward_flag=1 \
    --wrong_action_penalty=0 \
    --max_steps=500

echo ""
echo "============================================"
echo "Stochastic PPO Agent - Model Checking"
echo "============================================"

# Step 2: Run model checking on the trained stochastic agent
# The model checker will automatically:
# - Detect that the agent is a StochasticAgent
# - Expand all actions with probability > 0 during building
# - Convert the induced MDP to a DTMC
# - Run model checking on the DTMC
echo ""
echo "Step 2: Model Checking with Stochastic Agent..."
echo "  (Watch for 'Converting induced MDP to DTMC' message)"
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="test_stochastic_ppo" \
    --constant_definitions="MAX_JOBS=2,MAX_FUEL=10" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter_with_rewards.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F jobs_done=2 ]"

echo ""
echo "============================================"
echo "Test Complete!"
echo "============================================"
echo ""
echo "Key features demonstrated:"
echo "  - Behavioral cloning for fast optimal policy learning"
echo "  - Stochastic PPO extends StochasticAgent class"
echo "  - Implements action_probability_distribution(state) method"
echo "  - Policy samples actions stochastically (even during evaluation)"
echo "  - wrong_action_penalty=0 (no penalty for stochastic sampling)"
echo "  - Model checker builds MDP with all actions (prob > 0)"
echo "  - MDP is converted to DTMC before model checking"
echo "  - DTMC has single meta-action per state with combined probabilities"
echo "  - Probabilities renormalized when actions unavailable in states"
echo ""
echo "Why behavioral cloning?"
echo "  - Generates optimal dataset from PRISM model using model checking"
echo "  - Trains stochastic policy much faster than RL from scratch"
echo "  - Gets near-optimal policy in ~50 epochs vs 500+ episodes with RL"
echo ""
