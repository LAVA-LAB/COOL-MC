#!/bin/bash
# ICU Sepsis PPO - Post-Training Analysis
#
# Assumes the PPO agent is already trained (referenced via last_run.txt).
# Outputs all results to RESULTS.TXT.
#
# Analysis sections:
#   PART A: TRAINED POLICY ANALYSIS (induced DTMC)
#     1. Baseline survival/death probabilities
#     2. Action label: treatment reachability & trajectories
#     3. Clinical subgroups: SOFA, hypotension, multi-organ distress, septic shock
#     4. Treatment appropriateness: escalation, vasopressors, septic shock ladder
#   PART B: OPTIMAL PERMISSIVE POLICY ANALYSIS (full MDP)
#     5. MDP bounds: best/worst-case survival
#     6. Optimal subgroups + avoidability: organ failure, septic shock, hypotension
#     7. Action overlap: survival vs death optimal action conflicts
#     8. Action distribution: survival vs death optimal action frequencies

cd "$(dirname "$0")/.."

PROJECT="icu_sepsis_ppo"
PRISM_DIR="../prism_files"
PRISM_FILE="icu_sepsis.prism"
RESULTS="RESULTS.TXT"

# Clinical feature ranges (percentile-based, works with any normalization)
# These label ~25% of reachable states each, ensuring non-trivial PCTL results.
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
log "ICU Sepsis PPO - Comprehensive Analysis"
log "=========================================================================="
log "Date: $(date)"
log ""

# ##########################################################################
# PART A: TRAINED POLICY ANALYSIS (induced DTMC)
# ##########################################################################
log "##########################################################################"
log "PART A: TRAINED POLICY ANALYSIS (induced DTMC)"
log "##########################################################################"
log ""

# ==========================================================================
# 1. BASELINE: Survival and death probability
# ==========================================================================
log "=========================================================================="
log "1. BASELINE: Survival and death probability under trained policy"
log "=========================================================================="
log ""

log "--- 1a. P=? [ F \"survival\" ] ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "survival" ]'

log "--- 1b. P=? [ F \"death\" ] ---"
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
# 2. ACTION LABEL: Treatment reachability & trajectories
# ==========================================================================
log "=========================================================================="
log "2. ACTION LABEL: Treatment reachability under trained policy"
log "=========================================================================="
log ""

log "--- 2a. P=? [ \"f0_v0\" U \"survival\" ] (survive via no-treatment path) ---"
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

log "--- 2b. P=? [ F \"f4_v0\" ] (reach max fluids, no vasopressors) ---"
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

log "--- 2c. P=? [ F \"f4_v4\" ] (reach max fluids + max vasopressors) ---"
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
# 3. CLINICAL SUBGROUP ANALYSIS: feature_range labeler
#    Uses percentile-based ranges to identify clinically meaningful subgroups,
#    then queries temporal properties about outcomes.
# ==========================================================================
log "=========================================================================="
log "3. CLINICAL SUBGROUP ANALYSIS (trained policy)"
log "=========================================================================="
log ""

# --- 3a-c. High organ failure (SOFA): reachability, safe survival, unexpected deaths ---
log "--- 3a. P=? [ F \"high_sofa\" ] (probability of reaching high organ failure) ---"
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

log "--- 3b. P=? [ !\"high_sofa\" U \"survival\" ] (survive while avoiding high SOFA) ---"
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

log "--- 3c. P=? [ !\"high_sofa\" U \"death\" ] (die without high SOFA — unexpected deaths) ---"
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

# --- 3d-f. Hypotension: reachability, safe survival, unexpected deaths ---
log "--- 3d. P=? [ F \"hypotension\" ] (probability of reaching hypotensive state) ---"
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

log "--- 3e. P=? [ !\"hypotension\" U \"survival\" ] (survive without hypotension) ---"
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

log "--- 3f. P=? [ !\"hypotension\" U \"death\" ] (die without hypotension — unexpected deaths) ---"
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

# --- 3g. Multi-organ distress ---
log "--- 3g. P=? [ F \"high_sofa\" & \"high_lactate\" ] (reach high SOFA AND high lactate) ---"
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

# --- 3h. Septic shock (SOFA + hypotension): baseline for MDP comparison ---
log "--- 3h. P=? [ F \"high_sofa\" & \"hypotension\" ] (reach septic shock state) ---"
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
# 4. TREATMENT APPROPRIATENESS: feature_range + action_label combined
#    Does the policy prescribe the RIGHT treatment for the RIGHT patient?
#    Combines clinical subgroups with the policy's chosen action.
# ==========================================================================
log "=========================================================================="
log "4. TREATMENT APPROPRIATENESS (trained policy)"
log "=========================================================================="
log ""

# --- 4a-c. High SOFA: treatment escalation ladder ---
log "--- 4a. P=? [ F \"high_sofa\" & \"f0_v0\" ] (high SOFA gets NO treatment) ---"
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

log "--- 4b. P=? [ F \"high_sofa\" & \"f4_v0\" ] (high SOFA gets max fluids only) ---"
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

log "--- 4c. P=? [ F \"high_sofa\" & \"f4_v4\" ] (high SOFA gets max treatment) ---"
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

# --- 4d-f. Hypotension: treatment ladder ---
log "--- 4d. P=? [ F \"hypotension\" & \"f0_v0\" ] (hypotension gets NO treatment) ---"
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

log "--- 4e. P=? [ F \"hypotension\" & \"f4_v0\" ] (hypotension gets max fluids) ---"
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

log "--- 4f. P=? [ F \"hypotension\" & \"f4_v4\" ] (hypotension gets vasopressors + fluids) ---"
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

# --- 4g. Over-treatment check ---
log "--- 4g. P=? [ F \"not_high_sofa\" & \"f4_v4\" ] (non-critical patients get max treatment) ---"
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

# --- 4h. Most critical patients (high SOFA + high lactate) with no treatment ---
log "--- 4h. P=? [ F \"high_sofa\" & \"high_lactate\" & \"f0_v0\" ] (critical patients: no treatment) ---"
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

# --- 4i-j. Septic shock (SOFA + hypotension): treatment appropriateness ---
log "--- 4i. P=? [ F \"high_sofa\" & \"hypotension\" & \"f0_v0\" ] (septic shock: no treatment) ---"
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

log "--- 4j. P=? [ F \"high_sofa\" & \"hypotension\" & \"f4_v4\" ] (septic shock: vasopressors + fluids) ---"
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

log "--- 4k. P=? [ F \"high_sofa\" & \"hypotension\" & \"f4_v0\" ] (septic shock: fluids only, no vasopressors) ---"
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

# ##########################################################################
# PART B: OPTIMAL PERMISSIVE POLICY ANALYSIS (full MDP via allow_all_actions)
# ##########################################################################
log "##########################################################################"
log "PART B: OPTIMAL PERMISSIVE POLICY ANALYSIS (full MDP)"
log "##########################################################################"
log ""

# ==========================================================================
# 5. MDP BOUNDS: Best and worst case survival
# ==========================================================================
log "=========================================================================="
log "5. MDP BOUNDS: Best-case and worst-case survival under any policy"
log "=========================================================================="
log ""

log "--- 5a. Pmax=? [ F \"survival\" ] (best-case survival under any policy) ---"
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

log "--- 5b. Pmin=? [ F \"survival\" ] (worst-case survival under any policy) ---"
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
# 6. OPTIMAL CLINICAL SUBGROUP ANALYSIS (full MDP)
#    Compares best/worst-case achievable outcomes for clinical subgroups
#    against the trained policy results from Section 3.
# ==========================================================================
log "=========================================================================="
log "6. OPTIMAL CLINICAL SUBGROUP ANALYSIS (full MDP)"
log "=========================================================================="
log ""

# --- 6a-b. SOFA avoidance: Pmax and Pmin bounds ---
log "--- 6a. Pmax=? [ !\"high_sofa\" U \"survival\" ] (BEST-case: survive avoiding high SOFA) ---"
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

log "--- 6b. Pmin=? [ !\"high_sofa\" U \"survival\" ] (WORST-case: survive avoiding high SOFA) ---"
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

# --- 6c. Can ANY policy reduce unexpected deaths (deaths without high SOFA)? ---
log "--- 6c. Pmin=? [ !\"high_sofa\" U \"death\" ] (BEST-case: minimize unexpected deaths) ---"
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

# --- 6d. Is hypotension avoidable under ANY policy? ---
log "--- 6d. Pmin=? [ F \"hypotension\" ] (minimum probability of hypotension under any policy) ---"
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

# --- 6e-f. Combined critical state: Pmax/Pmin bounds ---
log "--- 6e. Pmax=? [ F \"high_sofa\" & \"high_lactate\" ] (BEST-case: reach critical state) ---"
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

log "--- 6f. Pmin=? [ F \"high_sofa\" & \"high_lactate\" ] (WORST-case: reach critical state) ---"
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

# --- 6g. Is organ failure avoidable under ANY policy? ---
log "--- 6g. Pmin=? [ F \"high_sofa\" ] (minimum probability of organ failure under any policy) ---"
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

# --- 6h. Is septic shock avoidable under ANY policy? ---
log "--- 6h. Pmin=? [ F \"high_sofa\" & \"hypotension\" ] (minimum probability of septic shock) ---"
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

# --- 6i. Can ANY policy reduce deaths without hypotension? ---
log "--- 6i. Pmin=? [ !\"hypotension\" U \"death\" ] (minimize deaths without hypotension) ---"
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
# 7. ACTION OVERLAP: Survival vs death optimal action conflicts (full MDP)
# ==========================================================================
log "=========================================================================="
log "7. ACTION OVERLAP: Survival vs death conflicts (full MDP)"
log "=========================================================================="
log ""

log "--- 7a. Pmax=? [ \"action_overlap\" U \"survival\" ] (survive through conflicting states) ---"
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
# 8. ACTION DISTRIBUTION: Survival vs death optimal action frequencies
# ==========================================================================
log "=========================================================================="
log "8. ACTION DISTRIBUTION: Survival vs death optimal action comparison"
log "=========================================================================="
log ""

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
log "ANALYSIS COMPLETE"
log "=========================================================================="
log ""
log "All results saved to: $RESULTS"
log ""
log "PART A — Trained Policy (induced DTMC):"
log "  1. Baseline survival/death"
log "  2. Treatment reachability (action labels)"
log "  3. Clinical subgroups: SOFA, hypotension, multi-organ distress, septic shock"
log "  4. Treatment appropriateness: escalation, vasopressors, septic shock ladder"
log ""
log "PART B — Optimal Permissive Policies (full MDP):"
log "  5. Pmax/Pmin survival bounds"
log "  6. Optimal subgroups + avoidability (organ failure, septic shock, hypotension)"
log "  7. Action overlap (survival vs death conflicts)"
log "  8. Action distribution comparison"
log ""
