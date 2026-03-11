#!/bin/bash
# Action Label State Labeler Example
#
# Demonstrates querying PCTL properties about specific action sequences
# chosen by the trained policy. Each state is labeled with the action
# the policy selects there, enabling temporal reasoning about treatment paths.
#
# ICU Sepsis actions: f{0-4}_v{0-4}
#   f = fluid level (0=none, 4=max)
#   v = vasopressor level (0=none, 4=max)

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Action Label Labeler - ICU Sepsis Example"
echo "=========================================="

# Step 1: Train a PPO agent on ICU Sepsis
echo ""
echo "Phase 1: Training PPO Agent..."
echo "------------------------------------------"

python cool_mc.py \
    --project_name="icu_sepsis_action_label" \
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
    --num_episodes=10000 \
    --eval_interval=1000 \
    --max_steps=1000 \
    --reward_flag=1 \
    --wrong_action_penalty=0 \
    --behavioral_cloning="raw_dataset;../prism_files/icu_sepsis.prism;Pmax=? [ F \"survival\" ];" \
    --bc_epochs=65 \
    --prop="Pmax=? [ F \"survival\" ]" \
    --deploy=0

# Step 2: Model checking with action labels
# Query: What is the probability of reaching survival while the policy
# keeps choosing "no treatment" (f0_v0)?
echo ""
echo "Phase 2: Checking P=? [ \"f0_v0\" U \"survival\" ]"
echo "  (Probability of reaching survival while policy chooses no-treatment)"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_action_label" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ "f0_v0" U "survival" ]' \
    --state_labeler="action_label"

# Step 3: Query a specific treatment action sequence
# Query: What is the probability of reaching survival while the policy
# keeps choosing max vasopressors with no fluids (f0_v4)?
echo ""
echo "Phase 3: Checking P=? [ \"f0_v4\" U \"survival\" ]"
echo "  (Probability of reaching survival while policy chooses max vasopressors)"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_action_label" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ "f0_v4" U "survival" ]' \
    --state_labeler="action_label"

# Step 4: Check probability of eventually reaching a state where the policy
# chooses a specific action
echo ""
echo "Phase 4: Checking P=? [ F \"f2_v2\" ]"
echo "  (Probability of eventually reaching a state where policy chooses moderate treatment)"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_action_label" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "f2_v2" ]' \
    --state_labeler="action_label"

# Step 5: Negated action sequence - avoid a specific treatment on the way to survival
echo ""
echo "Phase 5: Checking P=? [ !\"f0_v0\" U \"survival\" ]"
echo "  (Probability of reaching survival while NEVER choosing no-treatment)"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_action_label" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ !"f0_v0" U "survival" ]' \
    --state_labeler="action_label"

echo ""
echo "=========================================="
echo "Action Label Analysis Complete!"
echo "=========================================="
echo ""
echo "PCTL queries demonstrated:"
echo '  P=? [ "action" U "goal" ]     - Reach goal while policy keeps choosing action'
echo '  P=? [ !"action" U "goal" ]    - Reach goal while policy never chooses action'
echo '  P=? [ F "action" ]            - Eventually reach a state where policy chooses action'
echo ""
