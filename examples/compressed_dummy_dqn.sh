#!/bin/bash
# Compressed Dummy DQN Training and Verification
# This script trains a DQN agent on the compressed_dummy.prism model
# and then verifies its performance using model checking

echo "=========================================="
echo "Compressed Dummy DQN Training & Verification"
echo "=========================================="

# Training Phase
echo ""
echo "Phase 1: Training DQN Agent..."
echo "------------------------------------------"

python cool_mc.py \
    --task=safe_training \
    --project_name="compressed_dummy_dqn" \
    --algorithm=dqn_agent \
    --prism_file_path="compressed_dummy.prism" \
    --constant_definitions="" \
    --prop="" \
    --reward_flag=1 \
    --wrong_action_penalty=0 \
    --seed=128 \
    --epsilon=1.0 \
    --epsilon_min=0.01 \
    --epsilon_dec=0.995 \
    --layers=2 \
    --neurons=64 \
    --lr=0.001 \
    --batch_size=32 \
    --num_episodes=100 \
    --eval_interval=100 \
    --gamma=0.99 \
    --max_steps=10

echo ""
echo "Phase 2: Model Checking - Goal Reachability with Interpretation..."
echo "------------------------------------------"

# Verification Phase - Probability of reaching goal state (s=4) with decision tree interpretation
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="compressed_dummy_dqn" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="compressed_dummy.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"goal\" ]" \
    --interpreter="decision_tree" \


echo ""
echo "Phase 3: Model Checking - Bad State Reachability with Interpretation..."
echo "------------------------------------------"

# Verification Phase - Probability of reaching bad state (s=3) with decision tree interpretation
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="compressed_dummy_dqn" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="compressed_dummy.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"bad\" ]" \
    --interpreter="decision_tree" \


echo ""
echo "=========================================="
echo "Compressed Dummy DQN Experiment Complete!"
echo "=========================================="
