#!/bin/bash
# ICU Sepsis PPO - Part B: Trained Policy Analysis (induced DTMC)
#
# PURPOSE: Part A established the theoretical limits (best/worst outcomes
# under ANY treatment strategy). Part B evaluates our specific trained PPO
# policy: what does it actually achieve, and how does it compare to those
# theoretical bounds?
#
# KEY CONCEPT: Because the trained policy makes a fixed treatment decision
# in each state, the model becomes a Markov chain (DTMC) with no remaining
# choices. All probabilities here reflect a single, specific strategy.
# Unlike Part A, P[survival] + P[death] = 100% (same fixed policy).
#
# Assumes the PPO agent is already trained (referenced via last_run.txt).
# Outputs results to RESULTS_B.TXT.
#
# Sections:
#   5. Baseline survival/death probabilities
#   6. Action label: treatment reachability & trajectories
#   7. Severity subgroups: high SOFA, low BP, elevated lactate (percentile-based)
#   8. Treatment appropriateness: severity-treatment matching

cd "$(dirname "$0")/.."

PROJECT="icu_sepsis_ppo"
PRISM_DIR="../prism_files"
PRISM_FILE="icu_sepsis.prism"
RESULTS="RESULTS_B.TXT"

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
log "ICU Sepsis PPO - Part B: Trained Policy Analysis (induced DTMC)"
log "=========================================================================="
log "Date: $(date)"
log ""

# ==========================================================================
# 5. BASELINE: Survival and death probability
#
# CONTEXT: These are the most fundamental questions about our trained policy:
# what fraction of patients survive vs. die under its treatment decisions?
# Since the policy is fixed, this is a single Markov chain and the two
# probabilities must sum to 100%.
# ==========================================================================
log "=========================================================================="
log "5. BASELINE: Survival and death probability under trained policy"
log "=========================================================================="
log ""

# QUESTION: What is the survival probability under the trained policy?
# WHY IT MATTERS: This is the primary performance metric of our trained policy.
#   Compare to the theoretical upper bound from Part A (1a) to determine whether
#   the policy is optimal or if there is room for improvement.
# RESULT KEY: Compare to Part A 1a (Pmax). If they match, the policy is provably
#   optimal. The gap (Part A 1a - this value) is the "optimality gap".
log "--- 5a. P=? [ F \"survival\" ] ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "survival" ]'

# QUESTION: What is the death probability under the trained policy?
# WHY IT MATTERS: The complement of survival. Under a fixed policy,
#   P[death] = 1 - P[survival]. This value should match that identity.
#   Compare to Part A 1b (Pmin survival) to see how far above the worst case
#   our policy sits.
# RESULT KEY: Should equal (1 - 5a). Compare to Part A worst-case bounds.
log "--- 5b. P=? [ F \"death\" ] ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "death" ]'

# ==========================================================================
# 6. ACTION LABEL: Treatment reachability & trajectories
#
# CONTEXT: The trained policy assigns a specific treatment action (combination
# of IV fluids and vasopressors) to each patient state. This section examines
# which treatments the policy actually uses along patient trajectories. The
# action labels follow the format f{fluid_level}_v{vasopressor_level} where
# levels range from 0 (none) to 4 (maximum dose).
# ==========================================================================
log "=========================================================================="
log "6. ACTION LABEL: Treatment reachability under trained policy"
log "=========================================================================="
log ""

# QUESTION: What is the probability that a patient survives while only
#           receiving the "no treatment" action (f0_v0) at every step?
# WHY IT MATTERS: This reveals whether the policy has a "watchful waiting"
#   pathway -- patients who recover without any active intervention. A high
#   value suggests some patients are stable enough to survive without treatment;
#   a low value means the policy actively treats most patients.
# RESULT KEY: High = significant "no treatment needed" pathway exists.
#   Low = the policy intervenes for most patients before they reach survival.
log "--- 6a. P=? [ \"f0_v0\" U \"survival\" ] (survive via no-treatment path) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ "f0_v0" U "survival" ]' \
    --state_labeler="action_label"

# QUESTION: What is the probability that the policy prescribes maximum IV
#           fluids with no vasopressors at some point during the patient's stay?
# WHY IT MATTERS: Maximum fluid resuscitation without vasopressors is a specific
#   treatment strategy. Knowing how often the policy reaches this state reveals
#   whether the policy prefers aggressive fluid management alone.
# RESULT KEY: The frequency with which the trained policy uses a "fluids only"
#   aggressive strategy. Compare to 6c (max everything) to see treatment preferences.
log "--- 6b. P=? [ F \"f4_v0\" ] (reach max fluids, no vasopressors) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "f4_v0" ]' \
    --state_labeler="action_label"

# QUESTION: What is the probability that the policy prescribes maximum IV
#           fluids AND maximum vasopressors at some point?
# WHY IT MATTERS: This is the most aggressive possible treatment. How often
#   the policy reaches this "all out" treatment reveals the proportion of
#   patients who become severely enough ill to need maximum intervention.
# RESULT KEY: High = many patients require the most aggressive treatment.
#   Low = the policy manages most patients with less intensive treatment.
log "--- 6c. P=? [ F \"f4_v4\" ] (reach max fluids + max vasopressors) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "f4_v4" ]' \
    --state_labeler="action_label"

# ==========================================================================
# 7. SEVERITY SUBGROUP ANALYSIS: feature_range labeler
#
# CONTEXT: How does the trained policy perform for patients who reach different
# severity levels? We use percentile-based thresholds to define severity
# subgroups and then measure reachability and outcomes within each subgroup.
# Compare these results to Part A Section 2 (theoretical bounds) to determine
# whether the trained policy is optimal for each subgroup.
# ==========================================================================
log "=========================================================================="
log "7. SEVERITY SUBGROUP ANALYSIS (trained policy)"
log "=========================================================================="
log ""

# --- 7a-c. High SOFA (top 25%): reachability, safe survival, deaths from lower-severity ---

# QUESTION: Under the trained policy, what is the probability that a patient
#           reaches a state with severe organ dysfunction (top 25% SOFA)?
# WHY IT MATTERS: This measures how often the trained policy allows patients
#   to deteriorate to severe organ dysfunction. Compare to Part A 2g (minimum
#   achievable probability under any policy) to see whether this is avoidable.
# RESULT KEY: Compare to Part A 2g. If they match, severe SOFA is unavoidable.
#   If this is higher, the trained policy could potentially prevent some of these.
log "--- 7a. P=? [ F \"high_sofa\" ] (probability of reaching a top-25% SOFA state) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}"

# QUESTION: Under the trained policy, what is the probability that a patient
#           survives without ever entering severe organ dysfunction?
# WHY IT MATTERS: This is the "clean survival" rate -- recovery without severe
#   complications. Compare to Part A 2a (theoretical best) and 2b (theoretical
#   worst) to benchmark the trained policy's performance.
# RESULT KEY: Compare to Part A 2a. If they match, the trained policy achieves
#   the best possible "clean survival" rate.
log "--- 7b. P=? [ !\"high_sofa\" U \"survival\" ] (survive without entering top-25% SOFA) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ !"high_sofa" U "survival" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}"

# QUESTION: Under the trained policy, what is the probability that a patient
#           dies without ever reaching severe organ dysfunction?
# WHY IT MATTERS: These are deaths among patients who appeared relatively
#   stable by SOFA criteria -- potentially unexpected deaths. Compare to
#   Part A 2c (irreducible minimum) to see if these deaths are preventable.
# RESULT KEY: Compare to Part A 2c. If they match, these deaths are unavoidable
#   regardless of treatment. If higher, better treatment could reduce them.
log "--- 7c. P=? [ !\"high_sofa\" U \"death\" ] (die without reaching top-25% SOFA) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ !"high_sofa" U "death" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}"

# --- 7d-f. Low BP (bottom 25%): reachability, safe survival, deaths from higher BP ---

# QUESTION: Under the trained policy, what is the probability that a patient
#           experiences low blood pressure (bottom 25%)?
# WHY IT MATTERS: Hypotension is a critical complication in sepsis. Compare to
#   Part A 2d to determine if the trained policy could reduce hypotension episodes.
# RESULT KEY: Compare to Part A 2d. Gap = potentially preventable hypotension.
log "--- 7d. P=? [ F \"hypotension\" ] (probability of reaching a bottom-25% blood pressure state) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "hypotension" ]' \
    --state_labeler="feature_range;${LOW_BP}"

# QUESTION: Under the trained policy, what is the probability that a patient
#           survives without ever experiencing low blood pressure?
# WHY IT MATTERS: Survival without hypotension is a desirable outcome.
#   This measures how often the trained policy achieves "stable" recovery.
# RESULT KEY: Higher is better. Compare to Part A bounds for the achievable range.
log "--- 7e. P=? [ !\"hypotension\" U \"survival\" ] (survive without entering bottom-25% BP) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ !"hypotension" U "survival" ]' \
    --state_labeler="feature_range;${LOW_BP}"

# QUESTION: Under the trained policy, what is the probability that a patient
#           dies without ever experiencing low blood pressure?
# WHY IT MATTERS: Deaths with stable blood pressure suggest causes other than
#   hemodynamic collapse. Compare to Part A 2i (irreducible minimum).
# RESULT KEY: Compare to Part A 2i. If they match, these deaths are unavoidable.
log "--- 7f. P=? [ !\"hypotension\" U \"death\" ] (die without reaching bottom-25% BP) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ !"hypotension" U "death" ]' \
    --state_labeler="feature_range;${LOW_BP}"

# --- 7g-h. Combined severity ---

# QUESTION: Under the trained policy, what is the probability that a patient
#           reaches BOTH severe organ dysfunction AND elevated lactate?
# WHY IT MATTERS: This dangerous combination indicates simultaneous organ
#   failure and tissue oxygen deprivation. Compare to Part A 2e/2f bounds.
# RESULT KEY: Compare to Part A 2f (minimum achievable). Gap = potentially
#   preventable combined critical illness.
log "--- 7g. P=? [ F \"high_sofa\" & \"high_lactate\" ] (reach top-25% SOFA AND top-25% lactate) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" & "high_lactate" ]' \
    --state_labeler="feature_range;${HIGH_SOFA};${HIGH_LACTATE}"

# QUESTION: Under the trained policy, what is the probability that a patient
#           reaches BOTH severe organ dysfunction AND low blood pressure?
# WHY IT MATTERS: High SOFA + hypotension is an extremely dangerous state.
#   Compare to Part A 2h to see if the trained policy could prevent this.
# RESULT KEY: Compare to Part A 2h. If they match, this combined crisis
#   is unavoidable. If higher, there may be room for improvement.
log "--- 7h. P=? [ F \"high_sofa\" & \"hypotension\" ] (reach top-25% SOFA AND bottom-25% BP) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" & "hypotension" ]' \
    --state_labeler="feature_range;${HIGH_SOFA};${LOW_BP}"

# ==========================================================================
# 8. TREATMENT APPROPRIATENESS: severity subgroup + action_label combined
#
# CONTEXT: Does the trained policy match treatment intensity to patient
# severity? Clinically, sicker patients should receive more aggressive
# treatment. This section combines severity subgroups with the policy's
# chosen actions to check whether the policy follows this principle.
# ==========================================================================
log "=========================================================================="
log "8. TREATMENT APPROPRIATENESS (trained policy)"
log "=========================================================================="
log ""

# --- 8a-c. Top-25% SOFA: treatment escalation ladder ---

# QUESTION: When a patient has severe organ dysfunction (top 25% SOFA), how
#           often does the trained policy prescribe NO treatment?
# WHY IT MATTERS: Giving no treatment to the most severely ill patients would
#   be clinically concerning. A low probability here is reassuring.
# RESULT KEY: Low = the policy correctly avoids withholding treatment from
#   the sickest patients. High = potential safety concern.
log "--- 8a. P=? [ F \"high_sofa\" & \"f0_v0\" ] (top-25% SOFA gets NO treatment) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" & "f0_v0" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}#action_label"

# QUESTION: When a patient has severe organ dysfunction, how often does the
#           trained policy prescribe maximum fluids without vasopressors?
# WHY IT MATTERS: Fluid resuscitation alone may be insufficient for severe
#   organ dysfunction. Compare to 8c to see the policy's treatment ladder.
# RESULT KEY: Part of the treatment escalation picture for severe patients.
log "--- 8b. P=? [ F \"high_sofa\" & \"f4_v0\" ] (top-25% SOFA gets max fluids only) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" & "f4_v0" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}#action_label"

# QUESTION: When a patient has severe organ dysfunction, how often does the
#           trained policy prescribe maximum treatment (fluids + vasopressors)?
# WHY IT MATTERS: Ideally, the most severely ill patients should receive the
#   most aggressive treatment. A high value here confirms clinical appropriateness.
# RESULT KEY: High = the policy appropriately escalates treatment for severe cases.
#   Compare 8a < 8b < 8c to verify treatment escalation with severity.
log "--- 8c. P=? [ F \"high_sofa\" & \"f4_v4\" ] (top-25% SOFA gets max treatment) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" & "f4_v4" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}#action_label"

# --- 8d-f. Bottom-25% BP: treatment ladder ---

# QUESTION: When a patient has dangerously low blood pressure, how often does
#           the trained policy prescribe NO treatment?
# WHY IT MATTERS: Hypotension typically requires active intervention
#   (vasopressors). Withholding treatment here would be clinically inappropriate.
# RESULT KEY: Low = the policy avoids leaving hypotensive patients untreated.
log "--- 8d. P=? [ F \"hypotension\" & \"f0_v0\" ] (bottom-25% BP gets NO treatment) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "hypotension" & "f0_v0" ]' \
    --state_labeler="feature_range;${LOW_BP}#action_label"

# QUESTION: When a patient has low blood pressure, how often does the trained
#           policy prescribe maximum fluids without vasopressors?
# WHY IT MATTERS: For hypotension, vasopressors are often the first-line
#   treatment. Giving only fluids may be suboptimal.
# RESULT KEY: Part of the treatment escalation picture for hypotensive patients.
log "--- 8e. P=? [ F \"hypotension\" & \"f4_v0\" ] (bottom-25% BP gets max fluids) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "hypotension" & "f4_v0" ]' \
    --state_labeler="feature_range;${LOW_BP}#action_label"

# QUESTION: When a patient has low blood pressure, how often does the trained
#           policy prescribe the most aggressive treatment (fluids + vasopressors)?
# WHY IT MATTERS: Vasopressors are specifically indicated for hypotension.
#   A high probability confirms the policy uses the clinically appropriate treatment.
# RESULT KEY: High = clinically appropriate response to hypotension.
log "--- 8f. P=? [ F \"hypotension\" & \"f4_v4\" ] (bottom-25% BP gets vasopressors + fluids) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "hypotension" & "f4_v4" ]' \
    --state_labeler="feature_range;${LOW_BP}#action_label"

# --- 8g. Over-treatment check ---

# QUESTION: How often does the trained policy prescribe maximum treatment
#           to patients who are NOT in the most severe category (below top-25% SOFA)?
# WHY IT MATTERS: Over-treating lower-severity patients wastes resources and
#   may cause harm (e.g., fluid overload). A low value means the policy
#   reserves aggressive treatment for truly severe cases.
# RESULT KEY: Low = the policy avoids over-treating milder cases.
#   High = the policy may be overly aggressive with less severe patients.
log "--- 8g. P=? [ F \"not_high_sofa\" & \"f4_v4\" ] (lower-severity patients get max treatment) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "not_high_sofa" & "f4_v4" ]' \
    --state_labeler="feature_range;${HIGH_SOFA}#action_label"

# --- 8h. Most severe patients (top-25% SOFA + top-25% lactate) with no treatment ---

# QUESTION: When a patient has BOTH severe organ dysfunction AND elevated
#           lactate (the most critically ill), how often does the policy
#           prescribe no treatment?
# WHY IT MATTERS: This is the most dangerous patient subgroup. Any instances of
#   no treatment here would be a serious safety concern for the policy.
# RESULT KEY: Should be very low or zero. Any non-trivial value is a red flag.
log "--- 8h. P=? [ F \"high_sofa\" & \"high_lactate\" & \"f0_v0\" ] (most severe patients: no treatment) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" & "high_lactate" & "f0_v0" ]' \
    --state_labeler="feature_range;${HIGH_SOFA};${HIGH_LACTATE}#action_label"

# --- 8i-k. High SOFA + low BP (top-25% SOFA & bottom-25% BP): treatment ladder ---

# QUESTION: When a patient has BOTH severe organ dysfunction AND low blood
#           pressure, how often does the policy prescribe no treatment?
# WHY IT MATTERS: This is an extremely dangerous combination requiring urgent
#   intervention. No treatment here is a critical safety failure.
# RESULT KEY: Should be zero or near-zero.
log "--- 8i. P=? [ F \"high_sofa\" & \"hypotension\" & \"f0_v0\" ] (high SOFA + low BP: no treatment) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" & "hypotension" & "f0_v0" ]' \
    --state_labeler="feature_range;${HIGH_SOFA};${LOW_BP}#action_label"

# QUESTION: When a patient has BOTH severe organ dysfunction AND low blood
#           pressure, how often does the policy prescribe the most aggressive
#           treatment (maximum fluids + vasopressors)?
# WHY IT MATTERS: This is the clinically expected response -- the sickest
#   patients with hemodynamic instability should receive maximum support.
# RESULT KEY: High = clinically appropriate. This should be the dominant
#   treatment for this patient subgroup.
log "--- 8j. P=? [ F \"high_sofa\" & \"hypotension\" & \"f4_v4\" ] (high SOFA + low BP: vasopressors + fluids) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" & "hypotension" & "f4_v4" ]' \
    --state_labeler="feature_range;${HIGH_SOFA};${LOW_BP}#action_label"

# QUESTION: When a patient has BOTH severe organ dysfunction AND low blood
#           pressure, how often does the policy prescribe fluids only (no
#           vasopressors)?
# WHY IT MATTERS: Hypotension typically requires vasopressors. Giving only
#   fluids to patients with both organ dysfunction and hypotension may be
#   clinically insufficient. Compare to 8j to assess treatment appropriateness.
# RESULT KEY: Should be lower than 8j. If higher, the policy may under-treat
#   the most critically ill hypotensive patients.
log "--- 8k. P=? [ F \"high_sofa\" & \"hypotension\" & \"f4_v0\" ] (high SOFA + low BP: fluids only, no vasopressors) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "high_sofa" & "hypotension" & "f4_v0" ]' \
    --state_labeler="feature_range;${HIGH_SOFA};${LOW_BP}#action_label"

# ==========================================================================
# SUMMARY
# ==========================================================================
log "=========================================================================="
log "PART B COMPLETE"
log "=========================================================================="
log ""
log "All results saved to: $RESULTS"
log ""
log "Sections:"
log "  5. Baseline survival/death"
log "  6. Treatment reachability (action labels)"
log "  7. Severity subgroups: high SOFA, low BP, elevated lactate (percentile-based)"
log "  8. Treatment appropriateness: severity-treatment matching"
log ""
