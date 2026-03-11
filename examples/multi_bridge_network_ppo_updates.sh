#!/bin/bash
# Multi-Bridge Network PPO — Updated Experiments Only
#
# This script re-runs ONLY the phases that were updated or added.
# It reuses the existing trained model from multi_bridge_ppo.
# Run this AFTER the main multi_bridge_network_ppo.sh has trained the model.
#
# Updated phases:
#   Phase 5  — Budget sensitivity: B_MAX ±1 (was 6..10, now 9,10,11)
#   Phase 6  — Expected steps: T=? [F "budget_empty"] (was R{"steps"})
#   Phase 7  — Cycle-aware: P(budget_empty) (was P(failed))
#   Phase 13 — NEW: MinorMaint → MajorMaint, check P(budget_empty)

cd "$(dirname "$0")/.."

RESULTS_FILE="bridge_results_updates.txt"

# Capture all output to results file
exec > >(tee -a "$RESULTS_FILE") 2>&1
echo "=========================================="
echo "Multi-Bridge PPO — Updated Experiments"
echo "=========================================="
echo "Results logged to $RESULTS_FILE"

# ===========================================================
# PHASE 5 — Budget sensitivity: B_MAX ±1
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 5: Budget Sensitivity — B_MAX ±1"
echo "=========================================="
echo "Query: P=? [ F \"budget_empty\" ]  for B_MAX = 9, 10, 11"
echo ""
echo "Compares the probability that the budget is fully depleted when"
echo "the budget is one below (B_MAX=9), at training value (B_MAX=10),"
echo "and one above (B_MAX=11). Tests how sensitive the policy's spending"
echo "behavior is to slight budget changes around its training configuration."
echo "------------------------------------------"

for B in 9 10 11; do
    echo ""
    echo "Phase 5: P(budget_empty) with B_MAX=$B"
    echo "------------------------------------------"

    python cool_mc.py \
        --parent_run_id="last" \
        --project_name="multi_bridge_ppo" \
        --prism_dir="../prism_files" \
        --prism_file_path="multi-bridge-network3.prism" \
        --constant_definitions="B_MAX=$B" \
        --seed=42 \
        --task="rl_model_checking" \
        --prop='P=? [ F "budget_empty" ]'
done

# ===========================================================
# PHASE 7 — Cycle-aware behavior: P(budget_empty)
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 7: Cycle-Aware Behavior Detection"
echo "=========================================="
echo "Query: P=? [ F \"budget_empty\" ]  with cycle_year remapped to 0, 1, 2, 3"
echo ""
echo "Tests whether the policy has learned to anticipate budget reloads."
echo "Remaps cycle_year to a fixed value so the agent always 'thinks' it"
echo "is at a specific point in the 4-year budget cycle. If P(budget_empty)"
echo "differs between cy=0 (cycle start, budget just reloaded) and cy=3"
echo "(cycle end, budget about to reload), the policy is cycle-aware and"
echo "adjusts its spending behavior based on cycle position."
echo "------------------------------------------"

for CY in 0 1 2 3; do
    echo ""
    echo "Phase 7: Feature remap cycle_year=$CY — P=? [ F \"budget_empty\" ]"
    echo "------------------------------------------"

    python cool_mc.py \
        --parent_run_id="last" \
        --project_name="multi_bridge_ppo" \
        --prism_dir="../prism_files" \
        --prism_file_path="multi-bridge-network3.prism" \
        --constant_definitions="B_MAX=10" \
        --seed=42 \
        --task="rl_model_checking" \
        --prop='P=? [ F "budget_empty" ]' \
        --preprocessor="feature_remapping;cycle_year=[$CY]"
done

# ===========================================================
# PHASE 13 — Action Replacement: MinorMaint → MajorMaint
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 13: Action Replacement — MinorMaint → MajorMaint"
echo "=========================================="
echo "Method: --action_replace=\"1:2\" (level shorthand)"
echo "Query: P=? [ F \"budget_empty\" ]"
echo ""
echo "Every action containing MinorMaint (digit 1) on any bridge is"
echo "substituted with MajorMaint (digit 2), doubling the maintenance"
echo "cost for those actions. Tests whether the policy's minor repairs"
echo "are budget-critical — if P(budget_empty) increases significantly"
echo "compared to Phase 2e, the policy relies on cheap minor repairs"
echo "to stay within budget."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "budget_empty" ]' \
    --action_replace="1:2"

echo ""
echo "=========================================="
echo "All results saved to $RESULTS_FILE"
echo "=========================================="
