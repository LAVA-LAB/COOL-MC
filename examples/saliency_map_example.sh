#!/bin/bash
# Saliency Map Feature Importance Example
# Trains a DQN agent on the compressed_dummy model, then uses the
# saliency_map interpreter to rank features by gradient-based importance.
# Demonstrates both unfiltered and filtered (state subspace) analysis.

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Saliency Map Feature Importance Example"
echo "=========================================="

# Step 1: Train DQN agent
echo ""
echo "Phase 1: Training DQN Agent..."
echo "------------------------------------------"

python cool_mc.py \
    --task=safe_training \
    --project_name="saliency_map_example" \
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

# Step 2: Saliency map - all states
echo ""
echo "Phase 2: Saliency Map (all states)..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="saliency_map_example" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="compressed_dummy.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"goal\" ]" \
    --interpreter="saliency_map;;saliency_map_results.csv"

# Step 3: Saliency map - filtered subspace (feature a in [0, 0.5])
echo ""
echo "Phase 3: Saliency Map (filtered subspace: a=[0,0.5])..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="saliency_map_example" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="compressed_dummy.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"goal\" ]" \
    --interpreter="saliency_map;a=[0,0.5];saliency_map_filtered.csv"

echo ""
echo "=========================================="
echo "Saliency Map Example Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - All states: saliency_map_results.csv + saliency_map_results.png"
echo "  - Filtered:   saliency_map_filtered.csv + saliency_map_filtered.png"
echo ""
