#!/bin/bash
# ICU Sepsis PPO - Permutation Feature Importance Analysis
#
# Loads the last trained PPO policy and labels each reachable state with
# the feature whose permutation most frequently changes the policy's action.
#
# Evaluates the top-3 most important and bottom-3 least important features
# from the feature pruning importance ranking (feature_importance_pruning.csv):
#
#   Top 3:    input_4hourly, GCS, PaO2_FiO2
#   Bottom 3: mechvent, Temp_C, gender
#
# For each feature set, queries:
#   - P=? [ F "imp_{feature}" ]             — probability of reaching a state
#                                              where this feature drives the action
#   - P=? [ "imp_{feature}" U "survival" ]  — survive through feature-driven states
#   - P=? [ "imp_{feature}" U "death" ]     — die through feature-driven states

cd "$(dirname "$0")/.."

PROJECT="icu_sepsis_ppo"
PRISM_DIR="../prism_files"
PRISM_FILE="icu_sepsis.prism"
RESULTS="RESULTS_PERMUTATION_IMPORTANCE.TXT"

# Features from feature_importance_pruning.csv
TOP_FEATURES="input_4hourly,GCS,PaO2_FiO2"
BOTTOM_FEATURES="mechvent,Temp_C,gender"
ALL_FEATURES="${TOP_FEATURES},${BOTTOM_FEATURES}"

N_PERMUTATIONS=20

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
log "ICU Sepsis PPO - Permutation Feature Importance Analysis"
log "=========================================================================="
log "Date: $(date)"
log ""
log "Features evaluated:"
log "  Top 3 (most important):  ${TOP_FEATURES}"
log "  Bottom 3 (least important): ${BOTTOM_FEATURES}"
log "  Permutations per feature: ${N_PERMUTATIONS}"
log ""

# ##########################################################################
# 1. PERMUTATION IMPORTANCE LABELING: All 6 features
#    Labels each state with imp_{feature} for the most important feature
# ##########################################################################
log "=========================================================================="
log "1. PERMUTATION IMPORTANCE: Baseline labeling with all 6 features"
log "=========================================================================="
log ""

log "--- 1a. P=? [ F \"survival\" ] with permutation importance labels ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "survival" ]' \
    --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=${N_PERMUTATIONS};icu_analysis_permutation_importance.csv"

# ##########################################################################
# 2. TOP-3 FEATURE QUERIES: input_4hourly, GCS, PaO2_FiO2
# ##########################################################################
log "=========================================================================="
log "2. TOP-3 FEATURES: Reachability and outcome queries"
log "=========================================================================="
log ""

for FEATURE in input_4hourly GCS PaO2_FiO2; do
    log "--- 2. Feature: ${FEATURE} ---"
    log ""

    log "--- P=? [ F \"imp_${FEATURE}\" ] (reach state where ${FEATURE} drives action) ---"
    run_analysis \
        --parent_run_id="last" \
        --project_name="$PROJECT" \
        --constant_definitions="" \
        --prism_dir="$PRISM_DIR" \
        --prism_file_path="$PRISM_FILE" \
        --seed=128 \
        --task="rl_model_checking" \
        --prop="Pmax=? [ F \"imp_${FEATURE}\" ]" \
        --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=${N_PERMUTATIONS}"

    log "--- P=? [ \"imp_${FEATURE}\" U \"survival\" ] (survive through ${FEATURE}-driven states) ---"
    run_analysis \
        --parent_run_id="last" \
        --project_name="$PROJECT" \
        --constant_definitions="" \
        --prism_dir="$PRISM_DIR" \
        --prism_file_path="$PRISM_FILE" \
        --seed=128 \
        --task="rl_model_checking" \
        --prop="Pmax=? [ \"imp_${FEATURE}\" U \"survival\" ]" \
        --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=${N_PERMUTATIONS}"

    log "--- P=? [ \"imp_${FEATURE}\" U \"death\" ] (die through ${FEATURE}-driven states) ---"
    run_analysis \
        --parent_run_id="last" \
        --project_name="$PROJECT" \
        --constant_definitions="" \
        --prism_dir="$PRISM_DIR" \
        --prism_file_path="$PRISM_FILE" \
        --seed=128 \
        --task="rl_model_checking" \
        --prop="Pmax=? [ \"imp_${FEATURE}\" U \"death\" ]" \
        --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=${N_PERMUTATIONS}"
done

# ##########################################################################
# 3. BOTTOM-3 FEATURE QUERIES: mechvent, Temp_C, gender
# ##########################################################################
log "=========================================================================="
log "3. BOTTOM-3 FEATURES: Reachability and outcome queries"
log "=========================================================================="
log ""

for FEATURE in mechvent Temp_C gender; do
    log "--- 3. Feature: ${FEATURE} ---"
    log ""

    log "--- P=? [ F \"imp_${FEATURE}\" ] (reach state where ${FEATURE} drives action) ---"
    run_analysis \
        --parent_run_id="last" \
        --project_name="$PROJECT" \
        --constant_definitions="" \
        --prism_dir="$PRISM_DIR" \
        --prism_file_path="$PRISM_FILE" \
        --seed=128 \
        --task="rl_model_checking" \
        --prop="Pmax=? [ F \"imp_${FEATURE}\" ]" \
        --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=${N_PERMUTATIONS}"

    log "--- P=? [ \"imp_${FEATURE}\" U \"survival\" ] (survive through ${FEATURE}-driven states) ---"
    run_analysis \
        --parent_run_id="last" \
        --project_name="$PROJECT" \
        --constant_definitions="" \
        --prism_dir="$PRISM_DIR" \
        --prism_file_path="$PRISM_FILE" \
        --seed=128 \
        --task="rl_model_checking" \
        --prop="Pmax=? [ \"imp_${FEATURE}\" U \"survival\" ]" \
        --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=${N_PERMUTATIONS}"

    log "--- P=? [ \"imp_${FEATURE}\" U \"death\" ] (die through ${FEATURE}-driven states) ---"
    run_analysis \
        --parent_run_id="last" \
        --project_name="$PROJECT" \
        --constant_definitions="" \
        --prism_dir="$PRISM_DIR" \
        --prism_file_path="$PRISM_FILE" \
        --seed=128 \
        --task="rl_model_checking" \
        --prop="Pmax=? [ \"imp_${FEATURE}\" U \"death\" ]" \
        --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=${N_PERMUTATIONS}"
done

# ##########################################################################
# 4. COMPARATIVE: Top features vs bottom features combined
# ##########################################################################
log "=========================================================================="
log "4. COMPARATIVE: Top-importance vs bottom-importance regions"
log "=========================================================================="
log ""

log "--- 4a. P=? [ F \"imp_none\" ] (reach state where NO feature changes the action) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ F "imp_none" ]' \
    --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=${N_PERMUTATIONS}"

log "--- 4b. P=? [ \"imp_none\" U \"survival\" ] (survive through insensitive states) ---"
run_analysis \
    --parent_run_id="last" \
    --project_name="$PROJECT" \
    --constant_definitions="" \
    --prism_dir="$PRISM_DIR" \
    --prism_file_path="$PRISM_FILE" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop='Pmax=? [ "imp_none" U "survival" ]' \
    --state_labeler="permutation_importance;${ALL_FEATURES};n_permutations=${N_PERMUTATIONS}"

# ##########################################################################
# SUMMARY
# ##########################################################################
log "=========================================================================="
log "ANALYSIS COMPLETE"
log "=========================================================================="
log ""
log "All results saved to: $RESULTS"
log "Per-state importance scores: icu_analysis_permutation_importance.csv"
log ""
log "Sections:"
log "  1. Baseline labeling (all 6 features)"
log "  2. Top-3 features: input_4hourly, GCS, PaO2_FiO2"
log "  3. Bottom-3 features: mechvent, Temp_C, gender"
log "  4. Comparative queries (imp_none)"
log ""
