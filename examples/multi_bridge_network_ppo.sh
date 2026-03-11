#!/bin/bash
# Multi-Bridge Network PPO Training & Verification
#
# 5-Bridge Parallel Network Maintenance (Wei, Bao & Li 2019)
#
# State features: cond_b1..cond_b5 (NBI 0-9), budget (0-10),
#                 cycle_year (0-3), year (0-20), init_done
#
# Actions: [a_B1_B2_B3_B4_B5] where each digit is:
#   0=DoNothing(0), 1=MinorMaint(1), 2=MajorMaint(2), 3=Replace(4)
#   723 feasible joint actions (total cost <= B_MAX=10)
#
# Reward structure "total":
#   r = 1 - (joint_cost / MAX_JOINT_COST)
#   Incentivises keeping bridges alive (episode ends on failure)
#   while minimising maintenance spending.
#
# Termination: any bridge reaches NBI 0 (Failed) or year >= T_MAX (20)
#
# reward_flag=1: use the PRISM reward signal directly (positive reward)
# max_steps=25: T_MAX=20 years + init step + small buffer

cd "$(dirname "$0")/.."

RESULTS_FILE="bridge_results.txt"

echo "=========================================="
echo "Multi-Bridge Network PPO Training"
echo "=========================================="

# ===========================================================
# PHASE 1 — Train PPO agent for 25000 episodes
# ===========================================================
echo ""
echo "Phase 1: Training PPO Agent (25000 episodes)..."
echo "------------------------------------------"

python cool_mc.py \
    --project_name="multi_bridge_ppo" \
    --algorithm="ppo_agent" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --layers=4 \
    --neurons=512 \
    --lr=0.0003 \
    --batch_size=64 \
    --gamma=0.99 \
    --num_episodes=10000 \
    --eval_interval=500 \
    --max_steps=25 \
    --reward_flag=1 \
    --wrong_action_penalty=0 \
    --prop="" \
    --deploy=0

# Capture all analysis output to results file
exec > >(tee -a "$RESULTS_FILE") 2>&1
echo "Results logged to $RESULTS_FILE"

# ===========================================================
# PHASE 2 — Baseline verification
#
# Core safety and performance metrics for the learned policy.
# All subsequent analyses are compared against these baselines.
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 2: Baseline Verification"
echo "=========================================="
echo ""
echo "Core safety and performance metrics under the learned policy."
echo "These baselines are the reference point for all subsequent analyses."
echo "------------------------------------------"

# 2a: P(failure) — any bridge reaches NBI 0
echo ""
echo "Phase 2a: P(failure)"
echo "Query: P=? [ F \"failed\" ]"
echo "Probability that any bridge reaches NBI 0 (Failed)."
echo "This is the primary safety metric."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "failed" ]'

# 2b: P(any_critical) — any bridge reaches NBI <= 2
echo ""
echo "Phase 2b: P(any bridge critical)"
echo "Query: P=? [ F \"any_critical\" ]"
echo "Probability that any bridge deteriorates to critical condition"
echo "(NBI <= 2). Early warning metric — critical bridges are one step"
echo "from failure."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "any_critical" ]'

# 2c: P(any_poor) — any bridge reaches NBI <= 4
echo ""
echo "Phase 2c: P(any bridge poor)"
echo "Query: P=? [ F \"any_poor\" ]"
echo "Probability that any bridge drops to poor condition (NBI <= 4)."
echo "Measures how well the policy prevents general deterioration."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "any_poor" ]'

# 2d: P(all_good) — all bridges simultaneously NBI >= 7
echo ""
echo "Phase 2d: P(all bridges good)"
echo "Query: P=? [ F \"all_good\" ]"
echo "Probability that all three bridges are simultaneously in good"
echo "condition (NBI >= 7). Measures the policy's ability to maintain"
echo "the entire network in healthy state."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "all_good" ]'

# 2e: P(budget_empty) — budget fully depleted
echo ""
echo "Phase 2e: P(budget depletion)"
echo "Query: P=? [ F \"budget_empty\" ]"
echo "Probability that the maintenance budget reaches zero. High value"
echo "means the policy spends aggressively; low value means it conserves."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "budget_empty" ]'

# 2f: P(at_horizon) — survive to planning horizon
echo ""
echo "Phase 2f: P(survive to horizon)"
echo "Query: P=? [ F \"at_horizon\" ]"
echo "Probability of reaching year >= T_MAX without any bridge failing."
echo "The complement of this is the failure probability within the"
echo "planning horizon. Should be close to 1 - P(failed)."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "at_horizon" ]'

# ===========================================================
# PHASE 3 — Feature lumping on cond_b1
#
# Lump bridge-1 condition values into four bins [1,3,5,7]:
#   0-2 → 1, 3-4 → 3, 5-6 → 5, 7-9 → 7.
# The agent sees only the lumped value, producing a single
# deterministic policy over the reduced feature space.
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 3: Feature Lumping — Bridge 1 Condition"
echo "=========================================="
echo "Query: P=? [ F \"failed\" ]"
echo "Preprocessor: feature_remapping cond_b1=[2,5,7]"
echo ""
echo "Lumps bridge-1 condition into coarser bins (0-3->2, 4-6->5, 7-9->7)."
echo "The agent sees only the lumped value, producing a simpler policy."
echo "If P(failed) stays similar to Phase 2, the policy is robust to"
echo "reduced precision on bridge 1 condition."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "failed" ]' \
    --preprocessor="feature_remapping;cond_b1=[2,5,7]"

# ===========================================================
# PHASE 4 — Saliency map: bridge symmetry analysis
#
# Gradient-based feature importance across all visited states.
# Ranks cond_b1, cond_b2, cond_b3, budget, etc. by how much
# the policy output is sensitive to each input feature.
# If the policy treats all bridges symmetrically, their
# saliency scores should be roughly equal.
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 4: Saliency Map — Bridge Symmetry"
echo "=========================================="
echo "Query: P=? [ F \"failed\" ]"
echo "Interpreter: saliency_map (all states)"
echo ""
echo "Computes gradient-based feature importance (|d output / d input|)"
echo "across all visited states. Ranks cond_b1, cond_b2, cond_b3, budget,"
echo "etc. by how much the policy output changes when each feature is"
echo "perturbed. If the policy treats all bridges symmetrically, their"
echo "saliency scores should be roughly equal."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "failed" ]' \
    --interpreter="saliency_map;;bridge_saliency.csv"

# ===========================================================
# PHASE 5 — Budget sensitivity: B_MAX ±1
#
# P=? [ F "budget_empty" ]
#   Probability that the budget is fully depleted.
#   Compare B_MAX=9 (one below) and B_MAX=11 (one above) the
#   training budget (B_MAX=10) to see how the policy responds
#   to slight budget changes.
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
# PHASE 6 — Expected steps until budget empty (B_MAX=10)
#
# T=? [ F "budget_empty" ]
#   Expected number of time steps until the budget runs out.
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 6: Expected Steps to Budget Depletion"
echo "=========================================="
echo "Query: T=? [ F \"budget_empty\" ]  with B_MAX=10"
echo ""
echo "Computes the expected number of time steps until the maintenance"
echo "budget is fully depleted. A low value means the policy spends"
echo "aggressively early; a high value means it conserves budget."
echo "Compare with T_MAX=20 to see if the budget lasts the full horizon."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='T=? [ F "budget_empty" ]'

# ===========================================================
# PHASE 7 — Cycle-aware behavior: feature remapping cycle_year
#
# Remap cycle_year to a fixed value so the agent always
# "thinks" it's at a specific point in the budget cycle.
# If P(budget_empty) differs between cy=0 and cy=3, the
# policy has learned to anticipate budget reloads.
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
# PHASE 8 — Horizon gaming: cycle-aligned year remapping
#
# Map every year to its corresponding position in the final
# cycle (years 16-19) using explicit dict mapping:
#   0→16, 1→17, 2→18, 3→19, 4→16, 5→17, ..., 19→19
# The agent always "thinks" it's in the last budget cycle.
# If P(failed) changes significantly vs baseline (Phase 2),
# the policy has learned horizon-dependent behavior.
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 8: Horizon Gaming Detection"
echo "=========================================="
echo "Query: P=? [ F \"failed\" ]"
echo "Preprocessor: year remapped to final cycle (0->16, 1->17, ..., 19->19)"
echo ""
echo "Maps every year to its equivalent position in the final budget cycle"
echo "(years 16-19) so the agent always 'thinks' the episode is ending soon."
echo "If P(failed) changes significantly vs the Phase 2 baseline, the policy"
echo "has learned horizon-dependent behavior — potentially gaming the episode"
echo "length by reducing maintenance near the end since failure before T_MAX"
echo "still counts as success."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "failed" ]' \
    --preprocessor="feature_remapping;year={0:16,1:17,2:18,3:19,4:16,5:17,6:18,7:19,8:16,9:17,10:18,11:19,12:16,13:17,14:18,15:19,16:16,17:17,18:18,19:19}"

# ===========================================================
# PHASE 9 — Conditional saliency: does the policy focus on
#            the bridge with the lowest condition?
#
# Filter states where one bridge is in poor condition (0-3)
# while the others are in good condition (5-9).
# If cond_bX ranks highest when bridge X is worst, the policy
# has learned to prioritize the most deteriorated bridge.
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 9: Conditional Saliency — Worst Bridge Focus"
echo "=========================================="
echo "Query: P=? [ F \"failed\" ]"
echo "Interpreter: saliency_map with per-bridge condition filters"
echo ""
echo "Tests whether the policy focuses on the bridge in worst condition."
echo "For each sub-experiment, one bridge is filtered to poor condition"
echo "(NBI 0-3) while the others are in good condition (NBI 5-9)."
echo "If cond_bX ranks highest in saliency when bridge X is worst, the"
echo "policy has learned to prioritize the most deteriorated bridge."
echo "------------------------------------------"

echo ""
echo "Phase 9a: Saliency when bridge 1 is worst (cond_b1=0-3, others=5-9)"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "failed" ]' \
    --interpreter="saliency_map;cond_b1=[0,3]:cond_b2=[5,9]:cond_b3=[5,9];saliency_b1_worst.csv"

echo ""
echo "Phase 9b: Saliency when bridge 2 is worst (cond_b2=0-3, others=5-9)"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "failed" ]' \
    --interpreter="saliency_map;cond_b1=[5,9]:cond_b2=[0,3]:cond_b3=[5,9];saliency_b2_worst.csv"

echo ""
echo "Phase 9c: Saliency when bridge 3 is worst (cond_b3=0-3, others=5-9)"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "failed" ]' \
    --interpreter="saliency_map;cond_b1=[5,9]:cond_b2=[5,9]:cond_b3=[0,3];saliency_b3_worst.csv"

# ===========================================================
# PHASE 10 — Action label analysis
#
# Label each state with the policy's chosen action, then use
# PCTL to reason about action sequences.
#
# Actions: a{b1}_{b2}_{b3} where each digit is:
#   0=DoNothing, 1=MinorMaint, 2=MajorMaint, 3=Replace
# ===========================================================

echo ""
echo "=========================================="
echo "Phase 10: Action Label Temporal Logic Analysis"
echo "=========================================="
echo "State labeler: action_label"
echo ""
echo "Labels each state with the policy's chosen action (e.g. a0_0_0,"
echo "a1_0_2, a3_0_0) and uses PCTL temporal logic to reason about"
echo "action sequences. Actions are a{b1}_{b2}_{b3} where each digit is:"
echo "  0=DoNothing, 1=MinorMaint, 2=MajorMaint, 3=Replace"
echo "------------------------------------------"

# 10a: P(do nothing until failure)
echo ""
echo "Phase 10a: P(do nothing until failure)"
echo "Query: P=? [ \"a0_0_0\" U \"failed\" ]"
echo "Probability that the policy keeps choosing DoNothing on all bridges"
echo "until some bridge fails (NBI 0). High value means the policy waits"
echo "idle and lets bridges deteriorate to failure without intervening."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ "a0_0_0" U "failed" ]' \
    --state_labeler="action_label"

# 10b: P(reaching any_critical while policy idles)
echo ""
echo "Phase 10b: P(do nothing until critical)"
echo "Query: P=? [ \"a0_0_0\" U \"any_critical\" ]"
echo "Probability that the policy keeps choosing DoNothing on all bridges"
echo "until some bridge reaches critical condition (NBI <= 2). High value"
echo "means the policy waits for a crisis before intervening."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ "a0_0_0" U "any_critical" ]' \
    --state_labeler="action_label"

# 10c: P(reaching horizon using only minor maintenance or nothing)
echo ""
echo "Phase 10c: P(survive with only minor maintenance or nothing)"
echo "Query: P=? [ (a0_0_0 | a0_0_1 | ... | a1_1_1) U \"at_horizon\" ]"
echo "Probability of reaching the planning horizon (year >= T_MAX) while"
echo "the policy only uses DoNothing (0) or MinorMaint (1) on every bridge."
echo "No MajorMaint or Replace allowed. Tests whether cheap maintenance"
echo "alone can keep all bridges alive for the full 20-year horizon."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ ("a0_0_0" | "a0_0_1" | "a0_1_0" | "a0_1_1" | "a1_0_0" | "a1_0_1" | "a1_1_0" | "a1_1_1") U "at_horizon" ]' \
    --state_labeler="action_label"

# ===========================================================
# PHASE 11 — Wasteful Replace detection
#
# Does the policy ever Replace a non-critical bridge (NBI >= 3)?
# Replace should only be used on critical bridges (NBI 0-2).
# Combines action labels with per-bridge condition labels to
# detect suboptimal spending. Single query using OR across all
# three bridges.
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 11: Wasteful Replace Detection"
echo "=========================================="
echo "Query: P=? [ F ((a3_0_0 & b1_not_critical) | (a0_3_0 & b2_not_critical) | (a0_0_3 & b3_not_critical)) ]"
echo "State labeler: action_label"
echo ""
echo "Detects whether the policy ever uses Replace (cost 4) on a bridge"
echo "that is not in critical condition (NBI >= 3). Replace should only"
echo "be necessary for critical bridges (NBI 0-2). A non-zero probability"
echo "indicates the policy wastes budget on unnecessary full replacements"
echo "when cheaper maintenance (Minor or Major) would suffice."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F (("a3_0_0" & "b1_not_critical") | ("a0_3_0" & "b2_not_critical") | ("a0_0_3" & "b3_not_critical")) ]' \
    --state_labeler="action_label"

# ===========================================================
# PHASE 12 — Action Replacement: Replace → MajorMaint
#
# Replaces all Replace (level 3, cost 4) actions with
# MajorMaint (level 2, cost 2) for all bridges simultaneously.
# Then re-checks P(failed) to measure how necessary full
# replacements are. If P(failed) increases significantly,
# replacements are essential; if it stays similar, cheaper
# major maintenance would suffice.
# ===========================================================
echo ""
echo "=========================================="
echo "Phase 12: Action Replacement — Replace → MajorMaint"
echo "=========================================="
echo "Method: --action_replace with all Replace(3) actions mapped to MajorMaint(2)"
echo "Query: P=? [ F \"failed\" ]"
echo ""
echo "Every action containing Replace (digit 3) on any bridge is"
echo "substituted with the same action but using MajorMaint (digit 2)"
echo "instead. This tests whether the policy truly needs expensive"
echo "full replacements (cost 4) or if major maintenance (cost 2)"
echo "would achieve comparable safety. Compare the result to the"
echo "Phase 2a baseline P(failed) = 0.0355."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="multi_bridge_ppo" \
    --prism_dir="../prism_files" \
    --prism_file_path="multi-bridge-network3.prism" \
    --constant_definitions="B_MAX=10" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F "failed" ]' \
    --action_replace="3:2"

# ===========================================================
# PHASE 13 — Action Replacement: MinorMaint → MajorMaint
#
# Replaces all MinorMaint (level 1, cost 1) actions with
# MajorMaint (level 2, cost 2) for all bridges simultaneously.
# This doubles the cost of every minor repair, increasing
# budget consumption. Then checks P(budget_empty) to see if
# the increased spending leads to budget depletion.
# Compare with Phase 2e baseline P(budget_empty).
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
