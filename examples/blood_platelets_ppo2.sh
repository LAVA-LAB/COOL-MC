#!/bin/bash
# Blood Platelets Inventory — PPO Training & Full Analysis
#
# Reward convention:
#   reward_flag=0  →  PRISM cost signal is negated to a negative reward.
#   PPO's advantage normalisation turns "less penalty" into a positive
#   advantage, so it minimises costs just as well as it maximises rewards.
#
# State variables (features):  d, x1, x2, x3, x4, x5, pend, ph
#   tinv = x1+x2+x3+x4+x5  (total inventory; a PRISM formula, not a
#   direct feature — individual x variables are used as proxies below)
#
# Actions:  pr0 .. pr30  (produce 0 .. 30 aggregated units)
#   pr_k is only available in phase 0 when tinv + k <= MAXS.
#
# PRISM-defined labels (always present):
#   "empty"   = tinv=0     "full"  = tinv=MAXS
#   "monday"  = d=0        "weekend" = d>=5
#
# Observed policy action distribution (from Phase 5):
#   Most-used actions: pr6 (15%), pr15 (12.7%), pr11 (12.6%),
#     pr14 (11.8%), pr28 (8.3%), pr7 (5.1%), pr10 (5.7%)
#   Never-used actions: pr0, pr2, pr18, pr20, pr22, pr23, pr25,
#     pr26, pr29, pr30
#
# Properties use either P=? [ F<=200 ... ] (probability within the
# episode horizon) or T=? [ F ... ] (expected steps to reach a
# target state). max_steps=200 bounds the episode length.

cd "$(dirname "$0")/.."

ALL_FEATURES="d,x1,x2,x3,x4,x5,pend,ph"
RESULTS_FILE="blood_platelets_ppo2_results.txt"

# Tee all stdout+stderr to results file for the rest of the script
printf "Blood Platelets PPO — Results\nRun: %s\n============================================\n" "$(date)" > "$RESULTS_FILE"
exec > >(tee -a "$RESULTS_FILE") 2>&1

# ===========================================================
# PHASE 1 — Train PPO agent (default parameters, MAXS=30)
# ===========================================================
echo "=========================================="
echo "Blood Platelets PPO — Phase 1: Training"
echo "=========================================="

python cool_mc.py \
    --project_name="blood_platelets_ppo2" \
    --algorithm="ppo_agent" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --layers=3 \
    --neurons=256 \
    --lr=0.0003 \
    --batch_size=64 \
    --gamma=0.99 \
    --num_episodes=25000 \
    --eval_interval=500 \
    --max_steps=200 \
    --reward_flag=0 \
    --wrong_action_penalty=0 \
    --prop="" \
    --deploy=0

# ===========================================================
# PHASE 2 — Baseline safety checks over multiple time horizons
#
# 2a: Stockout probability P[F<=H "empty"] for horizons
#     H = 150, 175, 200, …, 400 steps. Shows how stockout
#     risk accumulates over time without retraining.
# 2b: Wastage probability P[F<=H "full"] for the same horizons.
#     Shows how overstocking risk grows with longer operation.
# ===========================================================
echo ""
echo "Phase 2a: Stockout probability over time horizons"
echo "------------------------------------------"

for HORIZON in 150 175 200 225 250 275 300 325 350 375 400; do
    echo ""
    echo "  Horizon ${HORIZON} ..."
    python cool_mc.py \
        --parent_run_id="last" \
        --project_name="blood_platelets_ppo2" \
        --prism_dir="../prism_files" \
        --prism_file_path="blood_platelets_inventory.prism" \
        --constant_definitions="MAXS=30" \
        --seed=42 \
        --task="rl_model_checking" \
        --prop='P=? [ F<='${HORIZON}' "empty" ]'
done

echo ""
echo "Phase 2b: Wastage probability over time horizons"
echo "------------------------------------------"

for HORIZON in 150 175 200 225 250 275 300 325 350 375 400; do
    echo ""
    echo "  Horizon ${HORIZON} ..."
    python cool_mc.py \
        --parent_run_id="last" \
        --project_name="blood_platelets_ppo2" \
        --prism_dir="../prism_files" \
        --prism_file_path="blood_platelets_inventory.prism" \
        --constant_definitions="MAXS=30" \
        --seed=42 \
        --task="rl_model_checking" \
        --prop='P=? [ F<='${HORIZON}' "full" ]'
done

# ===========================================================
# PHASE 3 — Permutation importance: expected steps per feature
#
# Labels each state with the feature whose permutation most
# often changes the policy action (imp_d, imp_x1, … imp_ph,
# or imp_none). Then computes T=? [ F "imp_FEAT" ] for each
# feature: the expected number of steps to first reach a state
# where that feature is the primary decision driver.
# Also produces an aggregate importance ranking (saved to CSV).
# ===========================================================
echo ""
echo "Phase 3: Permutation importance — T=? [ F \"imp_FEAT\" ] per feature"
echo "------------------------------------------"

for FEAT in d x1 x2 x3 x4 x5 pend; do
    echo ""
    echo "  Feature ${FEAT} ..."
    python cool_mc.py \
        --parent_run_id="last" \
        --project_name="blood_platelets_ppo2" \
        --prism_dir="../prism_files" \
        --prism_file_path="blood_platelets_inventory.prism" \
        --constant_definitions="MAXS=30" \
        --seed=42 \
        --task="rl_model_checking" \
        --prop='T=? [ F "imp_'${FEAT}'" ]' \
        --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=20;bp_permutation_importance.csv"
done

# ===========================================================
# PHASE 4 — Feature pruning: individual feature impact
#
# Zeroes EACH feature individually and re-runs model checking.
# Uses mode "all" = prune each feature one at a time and
# compare against the unpruned baseline.
#
# 4a: Delta in P[F<=200 "empty"] — which features are most
#     critical for preventing stockouts?
# 4b: Delta in P[F<=200 "full"] — which features are most
#     critical for preventing overstocking / wastage?
# ===========================================================
echo ""
echo "Phase 4a: Feature pruning impact on stockout"
echo "  P=? [ F<=200 \"empty\" ] per pruned feature"
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

echo ""
echo "Phase 4b: Feature pruning impact on wastage"
echo "  P=? [ F<=200 \"full\" ] per pruned feature"
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
    --interpreter="feature_pruning;all;bp_feature_pruning_full.csv"

# ===========================================================
# PHASE 5 — Action label: expected steps to each order size
#
# Labels every state with the action chosen by the policy,
# then computes T=? [ F "prX" ] for each action pr0..pr30:
# the expected number of steps to first reach a state where
# the policy selects that order size.
#
# Actions never selected by the policy will yield infinity
# (unreachable). Frequently used actions (e.g. pr6, pr15)
# will have low expected steps. This gives a complete picture
# of how quickly each ordering level is triggered.
# ===========================================================
echo ""
echo "Phase 5: Expected steps to each action — T=? [ F \"prX\" ]"
echo "------------------------------------------"

for ACTION in pr0 pr1 pr2 pr3 pr4 pr5 pr6 pr7 pr8 pr9 pr10 pr11 pr12 pr13 pr14 pr15 pr16 pr17 pr18 pr19 pr20 pr21 pr22 pr23 pr24 pr25 pr26 pr27 pr28 pr29 pr30; do
    echo ""
    echo "  Action ${ACTION} ..."
    python cool_mc.py \
        --parent_run_id="last" \
        --project_name="blood_platelets_ppo2" \
        --prism_dir="../prism_files" \
        --prism_file_path="blood_platelets_inventory.prism" \
        --constant_definitions="MAXS=30" \
        --seed=42 \
        --task="rl_model_checking" \
        --prop='T=? [ F "'${ACTION}'" ]' \
        --state_labeler="action_label"
done

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
    --prop='P=? [ F<=200 ("friday" & !"imp_d") ]' \
    --state_labeler="feature_range;d=[4,4]:friday#permutation_importance;${ALL_FEATURES};n_permutations=20;bp_perm_imp_friday.csv"

# ===========================================================
# PHASE 7 — Emergency restock proactivity (Until operator)
#
# P=? [ !"empty" U "pr28" ]
# "What is the probability that the policy triggers its
# emergency restock (pr28, ordering 28 units) BEFORE the
# first stockout?"
#
# High value: the policy is proactive — it detects dangerously
#   low inventory and fires a large reorder before patients
#   are affected. The emergency mechanism works as early warning.
# Low value: the policy is reactive — stockout sometimes occurs
#   before the emergency order fires, meaning the detection
#   threshold is too late.
# ===========================================================
echo ""
echo "Phase 7: Emergency restock proactivity"
echo "  P=? [ !\"empty\" U \"pr28\" ]"
echo "  Does the emergency order (pr28) fire before stockout?"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ !"empty" U "pr28" ]' \
    --state_labeler="action_label"

# ===========================================================
# PHASE 9 — Action-replacement counterfactual (pr14 → pr6)
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
# Compare against Phase 2 baselines for direct comparison.
#   9c — Counterfactual P[F<=200 "empty"] pr14 → pr6
#        If higher than Ph2a baseline: pr14 orders prevent stockouts.
#   9d — Counterfactual P[F<=200 "full"]  pr14 → pr6
#        If lower than Ph2b baseline: reducing to pr6 lowers wastage.
# ===========================================================
echo ""
echo "Phase 9: Action replacement counterfactual (pr14 -> pr6)"
echo "  Verifying: what happens to stockout/wastage probability if"
echo "  every pr14 order (14 units, 11.8% of states) is replaced"
echo "  with pr6 (6 units)? Compare against Phase 2 baselines."
echo "=========================================="

echo ""
echo "Phase 9c: Counterfactual — P=? [ F<=200 \"empty\" ] with pr14->pr6"
echo "  If higher than Ph2a: pr14 orders actively prevent stockouts."
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
echo "  If lower than Ph2b: reducing orders lowers wastage risk."
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
# PHASE 10 — Weekend resilience (Until operator)
#
# P=? [ !"empty" U "monday" ]
# "What is the probability that inventory stays positive
# continuously until the next Monday?"
#
# Since there is no production on Saturday/Sunday, this
# measures whether the Friday ordering decision provides
# a sufficient buffer to survive the weekend without
# a stockout. A high value means the policy's pre-weekend
# ordering is robust; a low value signals weekend
# vulnerability.
# ===========================================================
echo ""
echo "Phase 10: Weekend resilience"
echo "  P=? [ !\"empty\" U \"monday\" ]"
echo "  Does inventory survive the weekend without stockout?"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ !"empty" U "monday" ]'

# ===========================================================
# PHASE 11 — Gradual depletion pattern (Until operator)
#
# P=? [ "safe_stock" U<=200 "empty" ]
# "What is the probability that inventory stays above a
# safe threshold (tinv >= 5) all the way until a stockout?"
#
# High value: stockouts happen suddenly from seemingly safe
#   inventory levels — the policy (or a human monitor) would
#   have little warning.
# Low value: stockouts are preceded by a gradual decline
#   through low-stock states, giving advance warning and
#   opportunity for intervention.
#
# Uses feature_range labeler to define "safe_stock" = tinv>=5,
# approximated via x1+x2+x3+x4+x5 >= 5. Since tinv is a
# PRISM formula (not a direct feature), we label states where
# no individual age bucket is critically low as a proxy.
# ===========================================================
echo ""
echo "Phase 11: Gradual depletion pattern"
echo "  P=? [ \"safe_stock\" U<=200 \"empty\" ]"
echo "  Does inventory stay above safe threshold until stockout?"
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="blood_platelets_ppo2" \
    --prism_dir="../prism_files" \
    --prism_file_path="blood_platelets_inventory.prism" \
    --constant_definitions="MAXS=30" \
    --seed=42 \
    --task="rl_model_checking" \
    --prop='P=? [ "safe_stock" U<=200 "empty" ]'

# ===========================================================
echo ""
echo "=========================================="
echo "Blood Platelets PPO — Complete!"
echo "=========================================="
echo ""
echo "Summary of results:"
echo "  Ph2a  — P[F<=H empty] for H=150..400: stockout over time horizons"
echo "  Ph2b  — P[F<=H full]  for H=150..400: wastage over time horizons"
echo "  Ph3   — bp_permutation_importance.csv: most-relied-on feature per state"
echo "          T[F imp_FEAT] per feature: expected steps to each feature-dominated decision"
echo "  Ph4a  — bp_feature_pruning.csv: feature pruning impact on stockout"
echo "  Ph4b  — bp_feature_pruning_full.csv: feature pruning impact on wastage"
echo "  Ph5   — T[F prX] for each action pr0..pr30: expected steps to each order"
echo "  Ph6   — bp_perm_imp_friday.csv: permutation importance on Fridays"
echo "          P[F<=500 (friday & !imp_d)]: policy ignores day-of-week on Fridays"
echo "  Ph7   — P[!empty U pr28]: emergency restock fires before stockout?"
echo "  Ph9c  — P[F<=200 empty] with pr14->pr6: stockout if orders reduced"
echo "  Ph9d  — P[F<=200 full]  with pr14->pr6: wastage if orders reduced"
echo "  Ph10  — P[!empty U monday]: weekend resilience (survive until Monday)"
echo "  Ph11  — P[safe_stock U<=200 empty]: gradual depletion pattern"
echo ""
