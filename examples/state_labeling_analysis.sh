#!/bin/bash
# Example script for training a neural network policy and analyzing state labels
#
# This script demonstrates:
# 1. Training a small DQN agent on the transporter environment
# 2. Using rl_model_checking with state labelers to classify states
# 3. Checking properties involving custom labels from multiple labelers
#
# Available labelers:
#   - critical_state: Labels based on Q-value spread (max - min)
#   - top_two_gap: Labels based on gap between best and second-best action

# Configuration
PROJECT_NAME="state_labeling_analysis"
PRISM_DIR="../prism_files"
PRISM_FILE="transporter.prism"
CONSTANT_DEFS="MAX_JOBS=2,MAX_FUEL=10"
SEED=42

# Training parameters (small network for demo)
NUM_EPISODES=200
LAYERS=2
NEURONS=32

# State labeler thresholds
# critical_state: gap between max and min Q-values
CRITICAL_THRESHOLD=40
# top_two_gap: gap between max and second-max Q-values
CONFIDENCE_THRESHOLD=0.5

# Combined labeler config
LABELER_CONFIG="critical_state;threshold=$CRITICAL_THRESHOLD#top_two_gap;threshold=$CONFIDENCE_THRESHOLD"

echo "=========================================="
echo "Step 1: Training DQN Agent"
echo "=========================================="
python cool_mc.py \
    --task="safe_training" \
    --project_name="$PROJECT_NAME" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --constant_definitions="$CONSTANT_DEFS" \
    --algorithm="dqn_agent" \
    --num_episodes=$NUM_EPISODES \
    --layers=$LAYERS \
    --neurons=$NEURONS \
    --seed=$SEED

echo ""
echo "=========================================="
echo "Step 2: Model Checking with State Labels"
echo "=========================================="

# Property 1: Probability of reaching goal (baseline)
echo ""
echo "--- Property: P=? [ F \"empty\" ] ---"
python cool_mc.py \
    --task="rl_model_checking" \
    --parent_run_id="last" \
    --project_name="$PROJECT_NAME" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --constant_definitions="$CONSTANT_DEFS" \
    --prop="P=? [ F \"empty\" ]" \
    --state_labeler="$LABELER_CONFIG" \
    --seed=$SEED

# Property 2: Probability of reaching a critical state
echo ""
echo "--- Property: P=? [ F \"critical\" ] ---"
python cool_mc.py \
    --task="rl_model_checking" \
    --parent_run_id="last" \
    --project_name="$PROJECT_NAME" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --constant_definitions="$CONSTANT_DEFS" \
    --prop="P=? [ F \"critical\" ]" \
    --state_labeler="$LABELER_CONFIG" \
    --seed=$SEED

# Property 3: Probability of reaching a confident state
echo ""
echo "--- Property: P=? [ F \"confident\" ] ---"
python cool_mc.py \
    --task="rl_model_checking" \
    --parent_run_id="last" \
    --project_name="$PROJECT_NAME" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --constant_definitions="$CONSTANT_DEFS" \
    --prop="P=? [ F \"confident\" ]" \
    --state_labeler="$LABELER_CONFIG" \
    --seed=$SEED

# Property 4: Combined labels - confident AND non-critical (ideal states)
echo ""
echo "--- Property: P=? [ F \"confident\" & \"non_critical\" ] ---"
python cool_mc.py \
    --task="rl_model_checking" \
    --parent_run_id="last" \
    --project_name="$PROJECT_NAME" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --constant_definitions="$CONSTANT_DEFS" \
    --prop="P=? [ F \"confident\" & \"non_critical\" ]" \
    --state_labeler="$LABELER_CONFIG" \
    --seed=$SEED

# Property 5: Until property - staying non-critical until reaching goal
echo ""
echo "--- Property: P=? [ \"non_critical\" U \"empty\" ] ---"
python cool_mc.py \
    --task="rl_model_checking" \
    --parent_run_id="last" \
    --project_name="$PROJECT_NAME" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --constant_definitions="$CONSTANT_DEFS" \
    --prop="P=? [ \"non_critical\" U \"empty\" ]" \
    --state_labeler="$LABELER_CONFIG" \
    --seed=$SEED

# Property 6: Until property - not confident until confident
echo ""
echo "--- Property: P=? [ \"not_confident\" U \"confident\" ] ---"
python cool_mc.py \
    --task="rl_model_checking" \
    --parent_run_id="last" \
    --project_name="$PROJECT_NAME" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --constant_definitions="$CONSTANT_DEFS" \
    --prop="P=? [ \"not_confident\" U \"confident\" ]" \
    --state_labeler="$LABELER_CONFIG" \
    --seed=$SEED

echo ""
echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
echo ""
echo "State Labelers Used:"
echo ""
echo "  critical_state (threshold=$CRITICAL_THRESHOLD):"
echo "    - 'critical':     (max_Q - min_Q) < threshold (uncertain across all actions)"
echo "    - 'non_critical': (max_Q - min_Q) >= threshold (clear preference)"
echo ""
echo "  top_two_gap (threshold=$CONFIDENCE_THRESHOLD):"
echo "    - 'confident':     (max_Q - 2nd_max_Q) >= threshold (clear best action)"
echo "    - 'not_confident': (max_Q - 2nd_max_Q) < threshold (uncertain between top 2)"
echo ""
echo "Use 'mlflow ui' to view detailed results."
