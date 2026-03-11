#!/bin/bash
# ICU Sepsis PPO Agent Training & Feature Importance Ranking
# Pre-trains via behavioral cloning, then trains PPO for 10000 episodes,
# then performs feature importance ranking analysis.

cd "$(dirname "$0")/.."

echo "=========================================="
echo "ICU Sepsis PPO Training & Verification"
echo "=========================================="

# Step 1: Train PPO agent (BC pre-training + 10000 RL episodes)
echo ""
echo "Phase 1: Training PPO Agent (BC + RL)..."
echo "------------------------------------------"

python cool_mc.py \
    --project_name="icu_sepsis_ppo" \
    --algorithm="ppo_agent" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --constant_definitions="" \
    --seed=512 \
    --layers=4 \
    --neurons=256 \
    --lr=0.0003 \
    --batch_size=64 \
    --gamma=0.99 \
    --num_episodes=10000 \
    --eval_interval=100 \
    --max_steps=1000 \
    --reward_flag=1 \
    --wrong_action_penalty=0 \
    #--prop="Pmax=? [ F \"survival\" ]" \
    --behavioral_cloning="raw_dataset;../prism_files/icu_sepsis.prism;Pmax=? [ F \"survival\" ];" \
    --bc_epochs=300 \
    --deploy=0

# Step 2: Feature Importance Ranking - Survival Reachability
echo ""
echo "Phase 2: Feature Importance Ranking - Survival..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_ppo" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"survival\" ]" \
    --interpreter="feature_importance_ranking"

# Step 3: Feature Pruning - Key clinical features
echo ""
echo "Phase 3: Feature Pruning Analysis - Survival..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_ppo" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"survival\" ]" \
    --interpreter="feature_pruning;SOFA,Arterial_lactate,SpO2,GCS,SIRS;feature_pruning_results.csv"

# Step 4: Saliency Map - All states
echo ""
echo "Phase 4: Saliency Map Feature Importance (all states)..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_ppo" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"survival\" ]" \
    --interpreter="saliency_map;;saliency_map_results.csv"

# Step 5: Action Sensitivity - Fluids vs Vasopressors
echo ""
echo "Phase 5: Action Sensitivity (fluids vs vasopressors)..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_ppo" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="P=? [ F \"survival\" ]" \
    --interpreter="action_sensitivity;fluids,vasopressors;;action_sensitivity_results.csv"

echo ""
echo "=========================================="
echo "ICU Sepsis PPO Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Pre-trained PPO via behavioral cloning (300 epochs)"
echo "  - Trained PPO via reinforcement learning (10000 episodes)"
echo "  - Evaluated every 100 episodes against Pmax=? [ F survival ]"
echo "  - Feature importance ranking analysis (results in feature_importance_ranking_results.csv)"
echo "  - Feature pruning analysis (results in feature_pruning_results.csv)"
echo "  - Saliency map analysis (results in saliency_map_results.csv)"
echo "  - Action sensitivity analysis (results in action_sensitivity_results.csv)"
echo ""
