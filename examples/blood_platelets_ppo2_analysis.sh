#!/bin/bash
# Blood Platelets Inventory — PPO Analysis Only (no training)
#
# Runs all analysis phases (2–9) against an already-trained
# blood_platelets_ppo2 agent. Skips Phase 1 (training).
#
# Prerequisites: run blood_platelets_ppo2.sh Phase 1 first,
# or ensure a blood_platelets_ppo2 project exists in MLflow.
#
# State variables (features):  d, x1, x2, x3, x4, x5, pend, ph
#   tinv = x1+x2+x3+x4+x5  (total inventory)
#
# Actions:  pr0 .. pr30  (produce 0 .. 30 aggregated units)
#   pr_k is only available in phase 0 when tinv + k <= MAXS.
#
# PRISM-defined labels (always present):
#   "empty"   = tinv=0     "full"  = tinv=MAXS
#   "monday"  = d=0        "weekend" = d>=5
#
# Observed policy action distribution:
#   Most-used actions: pr6 (15%), pr15 (12.7%), pr11 (12.6%),
#     pr14 (11.8%), pr28 (8.3%), pr7 (5.1%), pr10 (5.7%)
#   Never-used actions: pr0, pr2, pr18, pr20, pr22, pr23, pr25,
#     pr26, pr29, pr30
#
# All time-bounded properties use F<=200, matching max_steps=200.

cd "$(dirname "$0")/.."

ALL_FEATURES="d,x1,x2,x3,x4,x5,pend,ph"
RESULTS_FILE="blood_platelets_ppo2_analysis_results.txt"

printf "Blood Platelets PPO2 — Analysis Only\nRun: %s\n============================================\n" "$(date)" > "$RESULTS_FILE"
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "=========================================="
echo "Blood Platelets PPO2 — Analysis (no training)"
echo "  Using existing project: blood_platelets_ppo2"
echo "=========================================="

# ===========================================================
# PHASE 2 — Baseline safety check: stockout within 200 steps
#
# Verifies: What is the probability that inventory reaches
# zero (tinv=0) within the full episode horizon?
# This is the core safety metric for the trained policy.
# ===========================================================
echo ""
echo "Phase 2: Baseline stockout probability"
echo "  P=? [ F<=200 \"empty\" ]"
echo "  Verifying: probability of stockout within 200 steps"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F<=200 "empty" ]'

# ===========================================================
# PHASE 3 — Permutation importance: most-relied-on feature
#
# Labels each state with the feature whose permutation most
# often changes the policy action (imp_d, imp_x1, … imp_ph,
# or imp_none). The prop then checks how often the policy
# relies on x1 (oldest stock) as its primary decision driver.
# ===========================================================
echo ""
echo "Phase 3: Permutation importance — P=? [ F<=200 \"imp_x1\" ]"
echo "  Verifying: probability of reaching a state where x1"
echo "  (oldest stock) is the most important feature for the"
echo "  policy's decision. Also produces aggregate importance"
echo "  ranking across all 8 features."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F<=200 "imp_x1" ]' \
    --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=20;bp_permutation_importance.csv"

# ===========================================================
# PHASE 4 — Feature pruning: individual feature impact
#
# Zeroes EACH feature individually and re-runs model checking.
# The resulting delta in P[F<=200 "empty"] reveals which
# single input the policy relies on most to prevent stockouts.
#
# Uses mode "all" = prune each feature one at a time and
# compare against the unpruned baseline. This produces a
# ranking of features by their impact on the safety property.
# ===========================================================
echo ""
echo "Phase 4: Individual feature pruning — P=? [ F<=200 \"empty\" ]"
echo "  Verifying: for each feature, how much does zeroing it"
echo "  increase the stockout probability? Features with the"
echo "  largest delta are most critical for the policy's safety."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F<=200 "empty" ]' \
    --interpreter="feature_pruning;all;bp_feature_pruning.csv"

# ===========================================================
# PHASE 5 — Action label: ordering behaviour
#
# Labels every state with the action chosen by the policy.
# These queries use actions the policy ACTUALLY selects:
#
# 5a: pr6 — the most frequently chosen action (15% of states).
#     This is a conservative mid-range order. A high probability
#     of reaching a pr6 state confirms the policy favours moderate
#     restocking, keeping inventory in a middle band.
#
# 5b: pr28 — the largest order the policy ever places (8.3% of
#     states). Measures how often the policy triggers a large
#     emergency restock. If this state is rarely reached, the
#     policy avoids extreme ordering — a sign of smooth control.
# ===========================================================
echo ""
echo "Phase 5a: Action label — P=? [ F<=200 \"pr6\" ] (most common order)"
echo "  Verifying: probability of reaching a state where the policy"
echo "  places its most frequent order (6 units) within 200 steps."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F<=200 "pr6" ]' \
    --state_labeler="action_label"

echo ""
echo "Phase 5b: Action label — P=? [ F<=200 \"pr28\" ] (largest order)"
echo "  Verifying: probability of reaching a state where the policy"
echo "  places its largest order (28 units) within 200 steps."
echo "  A large batch restock — indicates low-inventory situations."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F<=200 "pr28" ]' \
    --state_labeler="action_label"

# ===========================================================
# PHASE 6 — Friday + permutation importance
#
# Asks: what is the probability of reaching a Friday state
# where the day-of-week feature is NOT the most important
# driver of the policy's decision? A high value suggests the
# policy systematically ignores the day on Fridays — potentially
# dangerous, since Friday precedes the low-demand weekend.
#
# "friday" is added via feature_range (d=4).
# !"imp_d" uses PRISM's built-in negation to match any state
# where d is NOT the most important feature (i.e. the state
# does not carry the "imp_d" label from permutation importance).
# Both labelers are chained with '#'.
# ===========================================================
echo ""
echo "Phase 6: Friday + permutation importance"
echo "  P=? [ F (\"friday\" & !\"imp_d\") ]"
echo "  Verifying: probability of reaching a Friday where the"
echo "  policy does NOT consider day-of-week as its primary"
echo "  decision driver. High value = policy ignores weekday"
echo "  context before the low-demand weekend."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F ("friday" & !"imp_d") ]' \
    --state_labeler="feature_range;d=[4,4]:friday#permutation_importance;${ALL_FEATURES};n_permutations=20;bp_perm_imp_friday.csv"

# ===========================================================
# PHASE 7 — Action label × existing PRISM labels
#
# These experiments combine action labels with the PRISM-defined
# "monday", "weekend", and "empty" labels to detect policy
# behaviours that risk wastage or shortage.
#
# All queries use actions the policy ACTUALLY selects, ensuring
# non-trivial results.
#
# 7a: Large weekend order risk
#     P=? [ F ("pr28" & "weekend") ]
#     pr28 is the largest order this policy uses (8.3% of states).
#     It is only available when tinv + 28 <= 30, i.e. tinv <= 2.
#     On weekends, demand drops (lam=1.75 / 3.25). If the policy
#     orders 28 units with near-empty stock on a weekend, the
#     incoming batch will far exceed weekend consumption, leading
#     to outdating wastage.
#
# 7b: Small Monday order risk
#     P=? [ F ("pr1" & "monday") ]
#     Monday has the highest demand (lam=6.5). pr1 is one of the
#     smallest orders this policy uses (1.6% of states). Ordering
#     just 1 unit on Monday is inadequate for Monday-Tuesday
#     demand, risking a mid-week shortage.
#
# 7c: Stockout without large restocking
#     P=? [ !"pr28" U "empty" ]
#     Checks whether the system can reach stockout (empty) while
#     the policy never chose pr28 (its largest order) beforehand.
#     Since pr28 IS used in 8.3% of states, a value < 1.0 means
#     large orders actively prevent some stockout paths.
#     A value = 1.0 means stockout is reachable even through paths
#     where the policy always orders large — the shortage is
#     demand-driven, not an ordering failure.
# ===========================================================
echo ""
echo "Phase 7a: Large weekend order — P=? [ F (\"pr28\" & \"weekend\") ]"
echo "  Verifying: does the policy place its largest order (28 units)"
echo "  on weekends when demand is low? pr28 requires tinv <= 2."
echo "  High value = wastage risk from overordering on low-demand days."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F ("pr28" & "weekend") ]' \
    --state_labeler="action_label"

echo ""
echo "Phase 7b: Small Monday order — P=? [ F (\"pr1\" & \"monday\") ]"
echo "  Verifying: does the policy place a minimal order (1 unit)"
echo "  on Monday (highest demand, lam=6.5)? High value = shortage"
echo "  risk from underordering on the busiest day."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F ("pr1" & "monday") ]' \
    --state_labeler="action_label"

echo ""
echo "Phase 7c: Stockout without large order — P=? [ !\"pr28\" U \"empty\" ]"
echo "  Verifying: can the system reach stockout while the policy"
echo "  never placed its largest order (pr28)? pr28 is used in 8.3%"
echo "  of states. Value < 1.0 means large orders prevent some"
echo "  stockout paths; value = 1.0 means stockout is demand-driven."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ !"pr28" U "empty" ]' \
    --state_labeler="action_label"

# ===========================================================
# PHASE 8 — MAXS sensitivity sweep (no retraining)
#
# The trained policy (MAXS=30) is held fixed and the PRISM
# model is re-built with different MAXS values to measure how
# the stockout probability changes as inventory capacity changes.
# No new agent is trained — only the model changes.
# ===========================================================
echo ""
echo "Phase 8: MAXS sensitivity sweep — P=? [ F<=200 \"empty\" ] (no retraining)"
echo "  Verifying: how does changing the maximum inventory capacity"
echo "  affect stockout probability while keeping the same policy?"
echo "  The policy was trained with MAXS=30."
echo "=========================================="

for MAXS_VAL in 30 35; do
    echo ""
    echo "  MAXS=${MAXS_VAL} ..."
    python cool_mc.py \
        --parent_run_id="last" \
        --project_name="blood_platelets_ppo2" \
        --prism_dir="../prism_files" \
        --prism_file_path="blood_platelets_inventory.prism" \
        --constant_definitions="MAXS=${MAXS_VAL}" \
        --seed=42 \
        --task="rl_model_checking" \
        --prop='P=? [ F<=200 "empty" ]'
done

# ===========================================================
# PHASE 9 — Action-replacement counterfactual
#
# Counterfactual: pr14 → pr6
#   pr14 is the 4th most common action (11.8% of states, 709
#   states). It requires tinv + 14 <= 30, i.e. tinv <= 16.
#   pr6 is always available when pr14 is (tinv + 6 <= 30 holds
#   trivially whenever tinv + 14 <= 30), so no silent fallback.
#
#   Business question for inventory managers:
#     "What if we reduce medium-to-large orders (14 units) down
#      to modest orders (6 units)? Does the policy's decision
#      to order 14 instead of 6 actually protect against
#      stockouts, or is it unnecessarily aggressive?"
#
# Ground truth first, counterfactual second, for direct comparison.
#   9a — Ground truth  P[F<=200 "empty"] (stockout baseline)
#   9b — Ground truth  P[F<=200 "full"]  (wastage baseline)
#   9c — Counterfactual P[F<=200 "empty"] pr14 → pr6
#        If higher than 9a: pr14 orders actively prevent stockouts.
#   9d — Counterfactual P[F<=200 "full"]  pr14 → pr6
#        If lower than 9b: reducing to pr6 keeps inventory lower.
# ===========================================================
echo ""
echo "Phase 9: Action replacement counterfactual (pr14 -> pr6)"
echo "  Verifying: what happens to stockout/wastage probability if"
echo "  every pr14 order (14 units, 11.8% of states) is replaced"
echo "  with pr6 (6 units)? Tests whether the policy's medium-large"
echo "  orders are necessary or overly aggressive."
echo "=========================================="

echo ""
echo "Phase 9a: Ground truth — P=? [ F<=200 \"empty\" ] (no replacement)"
echo "  Baseline stockout probability for comparison."
echo "------------------------------------------"
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F<=200 "empty" ]'

echo ""
echo "Phase 9b: Ground truth — P=? [ F<=200 \"full\" ] (no replacement)"
echo "  Baseline wastage probability (inventory at MAXS) for comparison."
echo "------------------------------------------"
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F<=200 "full" ]'

echo ""
echo "Phase 9c: Counterfactual — P=? [ F<=200 \"empty\" ] with pr14->pr6"
echo "  If higher than 9a: pr14 orders actively prevent stockouts."
echo "  If same as 9a: reducing to pr6 does not increase shortage risk."
echo "------------------------------------------"
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F<=200 "empty" ]' \
    --action_replace="pr14:pr6"

echo ""
echo "Phase 9d: Counterfactual — P=? [ F<=200 \"full\" ] with pr14->pr6"
echo "  If lower than 9b: reducing orders keeps inventory below ceiling."
echo "  If same as 9b: smaller orders do not reduce wastage risk."
echo "------------------------------------------"
python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ F<=200 "full" ]' \
    --action_replace="pr14:pr6"

# ===========================================================
echo ""
echo "=========================================="
echo "Blood Platelets PPO2 — Analysis Complete!"
echo "=========================================="
echo ""
echo "Summary of results:"
echo "  Ph2   — Baseline P[F<=200 empty]: stockout probability (MAXS=30)"
echo "  Ph3   — bp_permutation_importance.csv: most-relied-on feature per state"
echo "          P[F<=200 imp_x1]: reachability of x1-dominated decisions"
echo "  Ph4   — bp_feature_pruning.csv: individual feature pruning impact on"
echo "          P[F<=200 empty] — ranks which features prevent stockouts"
echo "  Ph5a  — P[F<=200 pr6]:  probability of most-common order (6 units)"
echo "  Ph5b  — P[F<=200 pr28]: probability of largest order (28 units)"
echo "  Ph6   — bp_perm_imp_friday.csv: permutation importance on Fridays"
echo "          P[F (friday & !imp_d)]: policy ignores day-of-week on Fridays"
echo "  Ph7a  — P[F (pr28 & weekend)]: large order on low-demand weekend"
echo "  Ph7b  — P[F (pr1  & monday)]:  tiny order on high-demand Monday"
echo "  Ph7c  — P[!pr28 U empty]: stockout without the policy's largest order"
echo "  Ph8   — MAXS sweep {30,35}: P[F<=200 empty] capacity sensitivity"
echo "  Ph9a  — Ground truth P[F<=200 empty] (stockout baseline)"
echo "  Ph9b  — Ground truth P[F<=200 full]  (wastage baseline)"
echo "  Ph9c  — P[F<=200 empty] with pr14->pr6: stockout if orders reduced"
echo "  Ph9d  — P[F<=200 full]  with pr14->pr6: wastage if orders reduced"
echo ""
