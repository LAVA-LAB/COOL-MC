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
    --seed=42 \
    --layers=4 \
    --neurons=512 \
    --lr=0.0003 \
    --batch_size=32 \
    --gamma=0.99 \
    --num_episodes=25000 \
    --eval_interval=1000 \
    --max_steps=1000 \
    --reward_flag=1 \
    --wrong_action_penalty=0 \
    --behavioral_cloning="raw_dataset;../prism_files/icu_sepsis.prism;Pmax=? [ F \"survival\" ];" \
    --bc_epochs=65 \
    --prop="Pmax=? [ F \"survival\" ]" \
    --postprocessor="post_shielding;all_optimal_dataset;../prism_files/icu_sepsis.prism;Pmax=? [ F \"survival\" ];" \
    --deploy=0


python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_ppo" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmax=? [ !"action_overlap" U "survival" ]' \
    --state_labeler="action_overlap;Pmax=? [ F \"survival\" ];Pmax=? [ F \"death\" ]"


python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_ppo" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmax=? [ "f0_v0" U "survival" ]' \
    --state_labeler="action_label"

python cool_mc.py --parent_run_id="last" \
    --project_name="icu_sepsis_ppo" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "survival" ]' \
    --interpreter='action_distribution;Pmax=? [ F "survival" ];Pmax=? [ F "death" ];action_distribution_results.csv'

echo ""
echo "=========================================="
echo "ICU Sepsis PPO Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Action overlap analysis (survival vs death)"
echo "  - Action label analysis (policy action sequences)"
echo "  - Action landscape analysis (Pmax vs Pmin alignment)"
echo ""
