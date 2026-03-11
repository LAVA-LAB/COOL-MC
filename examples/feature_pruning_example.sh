#!/bin/bash
# Feature Pruning Example
# Trains a DQN agent on the compressed_dummy model, then uses the
# feature_pruning interpreter to measure how pruning individual input
# features from the neural network affects the safety property.

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Feature Pruning Example"
echo "=========================================="

# Step 1: Train DQN agent
echo ""
echo "Phase 1: Training DQN Agent..."
echo "------------------------------------------"

python cool_mc.py \
    --task=safe_training \
    --project_name="feature_pruning_example" \
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

# Step 2: Feature Pruning - prune both features individually and combined
# Config format: "feature_pruning;feature1,feature2[;output_file.csv]"
echo ""
echo "Phase 2: Feature Pruning Analysis..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="feature_pruning_example" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="compressed_dummy.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"goal\" ]" \
    --interpreter="feature_pruning;a,b;feature_pruning_results.csv"

echo ""
echo "=========================================="
echo "Feature Pruning Example Complete!"
echo "=========================================="
echo ""
echo "Results saved to feature_pruning_results.csv"
echo ""
