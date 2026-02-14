#!/bin/bash
# ICU Sepsis PPO - Part A: Optimal Permissive Policy Analysis (full MDP)
#
# PURPOSE: Part A explores the theoretical limits of the environment itself.
# It answers: "What is the best/worst that ANY treatment strategy could
# achieve?" by considering ALL possible treatment policies, not just ours.
#
# This is like asking: given this patient population and disease dynamics,
# what are the absolute best and worst outcomes that treatment decisions
# can produce? This lets us benchmark our trained policy against the
# theoretical optimum and understand which adverse events are unavoidable
# vs. preventable with better treatment.
#
# KEY CONCEPT: Each experiment uses "Pmax" (best possible strategy) or
# "Pmin" (worst possible strategy). These search over ALL conceivable
# treatment strategies -- not just the one we trained. Importantly, Pmax
# and Pmin each find a DIFFERENT strategy, so Pmax[survival] + Pmax[death]
# does NOT sum to 100% (see Section 1 comments for full explanation).
#
# Assumes the PPO agent is already trained (referenced via last_run.txt).
# Outputs results to RESULTS_A.TXT.
#
# Sections:
#   1. MDP bounds: best/worst-case survival
#   2. Optimal subgroups + avoidability: high SOFA, combined severity, low BP
#   3. Action overlap: survival vs death optimal action conflicts
#   4. Action distribution: survival vs death optimal action frequencies

cd "$(dirname "$0")/.."

PROJECT="icu_sepsis_ppo"
PRISM_DIR="../prism_files"
PRISM_FILE="icu_sepsis.prism"
RESULTS="RESULTS_A.TXT"

# Severity subgroups based on percentile thresholds (NOT clinical cutoffs).
# Each labels ~25% of reachable states, defining relative severity rankings:
#   high_sofa:     top 25% SOFA scores    = most severe organ dysfunction
#   high_lactate:  top 25% lactate levels  = most severe tissue hypoperfusion
#   hypotension:   bottom 25% mean BP      = lowest blood pressure states
HIGH_SOFA="SOFA=[p75,p100]:high_sofa"
HIGH_LACTATE="Arterial_lactate=[p75,p100]:high_lactate"
LOW_BP="MeanBP=[p0,p25]:hypotension"

# Clear results file
> "$RESULTS"

log() {
    echo "$1" | tee -a "$RESULTS"
}

run_analysis() {
    python cool_mc.py "$@" 2>&1 | tee -a "$RESULTS"
    echo "" >> "$RESULTS"
}

log "=========================================================================="
log "ICU Sepsis PPO - Part A: Optimal Permissive Policy Analysis (full MDP)"
log "=========================================================================="
log "Date: $(date)"
log ""

# ==========================================================================
# 1. MDP BOUNDS: Best and worst case survival
#
# CONTEXT: Before evaluating our specific trained policy (Part B), we first
# establish the theoretical limits: what are the absolute best and worst
# outcomes that ANY treatment strategy could achieve in this patient
# population? This tells us the range of what is possible and sets the
# benchmark against which we evaluate our trained policy.
#
# NOTE: Pmax and Pmin each find a DIFFERENT optimal strategy. Pmax finds the
# strategy that maximises survival; Pmin finds the one that minimises it.
# Because they represent different strategies, Pmax[survival] + Pmin[survival]
# does NOT necessarily equal 100%. To get complementary probabilities, use:
#   Pmax[survival] = 1 - Pmin[death]   (same strategy, opposite outcome)
# ==========================================================================
log "=========================================================================="
log "1. MDP BOUNDS: Best-case and worst-case survival under any policy"
log "=========================================================================="
log ""

# QUESTION: What is the highest possible survival rate achievable by any
#           treatment strategy in this patient population?
# WHY IT MATTERS: This is the theoretical upper bound. If our trained policy
#   matches this value, it is provably optimal -- no other strategy can do better.
# RESULT KEY: Compare this value to the trained policy survival from Part B (5a).
#   If they match, the trained policy already achieves the best possible outcome.
log "--- 1a. Pmax=? [ F \"survival\" ] (best-case survival under any policy) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmax=? [ F "survival" ]'

# QUESTION: What is the lowest possible survival rate under the worst
#           conceivable treatment strategy?
# WHY IT MATTERS: This is the theoretical lower bound -- the outcome if every
#   treatment decision were made as poorly as possible. The gap between this
#   and Pmax (1a) shows how much treatment decisions actually matter: a large
#   gap means treatment choices have a strong impact on patient outcomes.
# RESULT KEY: (Pmax - Pmin) = the "treatment influence range" on survival.
log "--- 1b. Pmin=? [ F \"survival\" ] (worst-case survival under any policy) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmin=? [ F "survival" ]'

# ==========================================================================
# 2. OPTIMAL SEVERITY SUBGROUP ANALYSIS (full MDP)
#
# CONTEXT: Patients pass through states of varying clinical severity during
# their ICU stay. We define severity subgroups using percentile thresholds
# (top/bottom 25%) on clinical indicators like SOFA score, lactate, and
# blood pressure. This section asks: can ANY treatment strategy avoid these
# severe states, or are some inevitable regardless of treatment?
#
# Comparing these theoretical bounds to the trained policy results (Part B,
# Section 7) reveals whether our policy is optimal for these subgroups too,
# or whether there is room for improvement.
# ==========================================================================
log "=========================================================================="
log "2. OPTIMAL SEVERITY SUBGROUP ANALYSIS (full MDP)"
log "=========================================================================="
log ""

# QUESTION: What is the highest possible probability of a patient surviving
#           while never entering a state of severe organ dysfunction
#           (top 25% SOFA scores)?
# WHY IT MATTERS: This is the best any treatment strategy can achieve for
#   "clean" survival -- recovery without the patient's organs deteriorating
#   severely. Compare to the trained policy result in Part B (7b) to see if
#   the trained policy matches this upper bound.
# RESULT KEY: High value = it IS possible to keep patients stable and alive.
#   Compare to Part B 7b (trained policy) and 2b below (worst-case).
log "--- 2a. Pmax=? [ !\"high_sofa\" U \"survival\" ] (BEST-case: survive without top-25% SOFA) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmax=? [ !"high_sofa" U "survival" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}"

# QUESTION: Under the worst possible treatment strategy, what is the lowest
#           probability of surviving without severe organ dysfunction?
# WHY IT MATTERS: This is the floor -- the worst that bad treatment decisions
#   could cause in terms of "clean" survival. The gap between 2a and 2b shows
#   how much treatment choices influence whether patients experience severe
#   organ dysfunction on their path to recovery.
# RESULT KEY: (2a - 2b) = the range of treatment influence on SOFA-free survival.
log "--- 2b. Pmin=? [ !\"high_sofa\" U \"survival\" ] (WORST-case: survive without top-25% SOFA) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmin=? [ !"high_sofa" U "survival" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}"

# QUESTION: What is the minimum probability that a patient dies without ever
#           reaching severe organ dysfunction (top 25% SOFA)?
# WHY IT MATTERS: Some patients may die even if their SOFA score never becomes
#   severely elevated. This measures the irreducible death rate among patients
#   who appear relatively stable by SOFA criteria. A non-zero value means that
#   even the best treatment cannot prevent all deaths in the lower-severity group.
# RESULT KEY: If > 0, some deaths occur without severe organ failure and are
#   unavoidable -- possibly due to other clinical factors not captured by SOFA.
log "--- 2c. Pmin=? [ !\"high_sofa\" U \"death\" ] (BEST-case: minimize deaths from below top-25% SOFA) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmin=? [ !"high_sofa" U "death" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}"

# QUESTION: Is it possible for ANY treatment strategy to completely prevent
#           a patient from ever experiencing low blood pressure (bottom 25% BP)?
# WHY IT MATTERS: Hypotension is a dangerous complication in sepsis. If this
#   probability is > 0 even under the best strategy, it means some degree of
#   hypotension is unavoidable in this patient population regardless of treatment.
# RESULT KEY: If = 0, hypotension can be fully prevented by optimal treatment.
#   If > 0, some hypotension episodes are structurally inevitable.
log "--- 2d. Pmin=? [ F \"hypotension\" ] (minimum probability of bottom-25% BP under any policy) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmin=? [ F "hypotension" ]' \
    --state_labeler="feature_range;${LOW_BP}"

# QUESTION: Under the worst treatment strategy, what is the maximum probability
#           that a patient develops BOTH severe organ dysfunction (top 25% SOFA)
#           AND elevated lactate (top 25%)?
# WHY IT MATTERS: The combination of high SOFA + high lactate indicates a
#   critically ill patient with both organ failure and tissue oxygen deprivation.
#   This worst-case bound shows how dangerous bad treatment could be.
# RESULT KEY: Compare to 2f (best-case). The gap shows how much treatment
#   decisions influence the risk of reaching this dangerous combined state.
log "--- 2e. Pmax=? [ F \"high_sofa\" & \"high_lactate\" ] (WORST-case: reach combined high severity) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmax=? [ F "high_sofa" & "high_lactate" ]' \
    --state_labeler="feature_range;${HIGH_SOFA};${HIGH_LACTATE}"

# QUESTION: Under the best treatment strategy, what is the minimum probability
#           that a patient develops BOTH severe organ dysfunction AND elevated
#           lactate?
# WHY IT MATTERS: If this is > 0, even optimal treatment cannot fully prevent
#   patients from reaching this critical combined state. If = 0, it means the
#   right treatment choices can always keep patients from simultaneous organ
#   failure and tissue hypoperfusion.
# RESULT KEY: If > 0, combined critical illness is partly unavoidable.
#   Compare to Part B (trained policy) to see how close the policy is to this bound.
log "--- 2f. Pmin=? [ F \"high_sofa\" & \"high_lactate\" ] (BEST-case: minimize combined high severity) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmin=? [ F "high_sofa" & "high_lactate" ]' \
    --state_labeler="feature_range;${HIGH_SOFA};${HIGH_LACTATE}"

# QUESTION: Can ANY treatment strategy completely prevent a patient from ever
#           reaching a state of severe organ dysfunction (top 25% SOFA)?
# WHY IT MATTERS: SOFA score is a primary indicator of organ failure severity
#   in ICU patients. If even the best strategy cannot prevent high SOFA, it
#   means some degree of organ deterioration is inherent to the disease process.
# RESULT KEY: If > 0, severe organ dysfunction is partly unavoidable.
#   The value represents the minimum "exposure" to high-severity states.
log "--- 2g. Pmin=? [ F \"high_sofa\" ] (minimum probability of reaching top-25% SOFA under any policy) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmin=? [ F "high_sofa" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}"

# QUESTION: Can ANY treatment strategy prevent a patient from ever reaching
#           BOTH severe organ dysfunction AND low blood pressure simultaneously?
# WHY IT MATTERS: High SOFA + hypotension together indicate a patient in
#   multi-organ failure with hemodynamic instability -- an extremely dangerous
#   combination. If this minimum probability is 0, proper treatment can always
#   prevent this combined crisis.
# RESULT KEY: If = 0, the combined crisis is fully avoidable with right treatment.
#   If > 0, some patients will inevitably reach this state.
log "--- 2h. Pmin=? [ F \"high_sofa\" & \"hypotension\" ] (minimum probability of top-25% SOFA + bottom-25% BP) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmin=? [ F "high_sofa" & "hypotension" ]' \
    --state_labeler="feature_range;${HIGH_SOFA};${LOW_BP}"

# QUESTION: What is the minimum probability that a patient dies while their
#           blood pressure never drops to the lowest 25%?
# WHY IT MATTERS: This identifies deaths that occur without overt hypotension.
#   A non-zero value reveals that some patients die from causes other than
#   hemodynamic collapse -- possibly infection, organ failure from other causes,
#   or factors not reflected in blood pressure alone.
# RESULT KEY: If > 0, low blood pressure is NOT a prerequisite for death --
#   some deaths are irreducible even with stable blood pressure.
log "--- 2i. Pmin=? [ !\"hypotension\" U \"death\" ] (minimize deaths without reaching bottom-25% BP) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmin=? [ !"hypotension" U "death" ]' \
    --state_labeler="feature_range;${LOW_BP}"

# ==========================================================================
# 3. ACTION OVERLAP: Survival vs death optimal action conflicts (full MDP)
#
# CONTEXT: For each patient state, there is a best action for maximising
# survival and a best action for maximising death. In some states, these
# are the SAME action -- meaning the "best" and "worst" treatment coincide.
# These are "conflict states" where the treatment decision is ambiguous:
# the action that gives the patient the best chance of survival also happens
# to be the one a "death-maximising" adversary would choose. This section
# asks how often patients must pass through such ambiguous states.
# ==========================================================================
log "=========================================================================="
log "3. ACTION OVERLAP: Survival vs death conflicts (full MDP)"
log "=========================================================================="
log ""

# QUESTION: What is the maximum probability that a patient survives while
#           passing ONLY through states where the best survival action and
#           best death action are the same ("conflict states")?
# WHY IT MATTERS: In conflict states, the optimal treatment is ambiguous --
#   the same action simultaneously maximises both survival and death probability.
#   A high value here means that patients can survive even when every treatment
#   decision along the way was in one of these ambiguous states.
#   A low value means that survival typically requires passing through states
#   where survival-optimal and death-optimal actions clearly differ, i.e.,
#   where treatment decisions have an unambiguous impact on the outcome.
# RESULT KEY: High = survival is possible even through ambiguous states.
#   Low = clear, decisive treatment choices are needed for survival.
log "--- 3a. Pmax=? [ \"action_overlap\" U \"survival\" ] (survive through conflicting states) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --preprocessor="allow_all_actions" \
    --prop='Pmax=? [ "action_overlap" U "survival" ]' \
    --state_labeler="action_overlap;Pmax=? [ F \"survival\" ];Pmax=? [ F \"death\" ]"

# ==========================================================================
# 4. ACTION DISTRIBUTION: Survival vs death optimal action frequencies
#
# CONTEXT: Each treatment action (combination of IV fluids and vasopressors)
# can be optimal for survival in some states and optimal for causing death in
# others. This section computes, across all patient states, how often each
# action is chosen by the survival-maximising vs. death-maximising strategy.
# The output is a CSV table that can be used directly for paper figures.
# ==========================================================================
log "=========================================================================="
log "4. ACTION DISTRIBUTION: Survival vs death optimal action comparison"
log "=========================================================================="
log ""

# QUESTION: For each treatment action (IV fluid / vasopressor combination),
#           in how many patient states is it the best action for survival vs.
#           the best action for causing death?
# WHY IT MATTERS: If an action appears frequently in both the survival and
#   death strategies, it means that action's effect is highly state-dependent --
#   helpful in some patient conditions, harmful in others. Actions that appear
#   predominantly in the survival strategy are "reliably beneficial", while
#   actions predominantly in the death strategy are "reliably harmful".
#   This directly informs clinical interpretation of the trained policy's
#   treatment preferences.
# RESULT KEY: Output CSV with per-action counts. Actions with high survival
#   frequency and low death frequency are the safest treatment choices.
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "survival" ]' \
    --interpreter='action_distribution;Pmax=? [ F "survival" ];Pmax=? [ F "death" ];icu_analysis_action_distribution.csv'

# ==========================================================================
# SUMMARY
# ==========================================================================
log "=========================================================================="
log "PART A COMPLETE"
log "=========================================================================="
log ""
log "All results saved to: $RESULTS"
log ""
log "Sections:"
log "  1. Pmax/Pmin survival bounds"
log "  2. Optimal subgroups + avoidability (high SOFA, combined severity, low BP)"
log "  3. Action overlap (survival vs death conflicts)"
log "  4. Action distribution comparison"
log ""
