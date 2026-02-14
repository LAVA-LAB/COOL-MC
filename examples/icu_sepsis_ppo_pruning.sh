#!/bin/bash
# ICU Sepsis PPO - Feature Pruning Experiments (stored separately)
#
# Results: All features showed <0.7% change due to post-shielding correcting errors.
# Ranking: GCS (-0.69%) > Arterial_lactate (-0.31%) > SOFA (-0.29%) > Platelets_count (-0.21%)
#          > Creatinine (-0.09%) > HR (-0.08%) > SpO2 (-0.08%) > MeanBP (-0.03%)
#          > input_total (-0.01%) > max_dose_vaso (-0.01%)
#
# These experiments were moved out of icu_sepsis_ppo_analysis.sh because
# post-shielding makes the policy robust to individual feature removal,
# so no feature provides meaningful discrimination.

cd "$(dirname "$0")/.."

PROJECT="icu_sepsis_ppo"
PRISM_DIR="../prism_files"
PRISM_FILE="icu_sepsis.prism"
RESULTS="PRUNING_RESULTS.TXT"

> "$RESULTS"

log() {
    echo "$1" | tee -a "$RESULTS"
}

run_analysis() {
    python cool_mc.py "$@" 2>&1 | tee -a "$RESULTS"
    echo "" >> "$RESULTS"
}

log "=========================================================================="
log "ICU Sepsis PPO - Feature Pruning Experiments"
log "=========================================================================="
log "Date: $(date)"
log ""

for FEAT in \
    gender mechvent max_dose_vaso re_admission age Weight_kg \
    GCS HR SysBP MeanBP DiaBP RR Temp_C FiO2_1 \
    Potassium Sodium Chloride Glucose Magnesium Calcium \
    Hb WBC_count Platelets_count PTT PT \
    Arterial_pH paO2 paCO2 Arterial_BE HCO3 Arterial_lactate \
    SOFA SIRS Shock_Index PaO2_FiO2 cumulated_balance SpO2 \
    BUN Creatinine SGOT SGPT Total_bili INR \
    input_total input_4hourly output_total output_4hourly; do
    log "--- Pruning ${FEAT} ---"
    OUTFILE=$(echo "$FEAT" | tr '[:upper:]' '[:lower:]')
    run_analysis \
        --parent_run_id="last" \
        --project_name="$PROJECT" \
        --constant_definitions="" \
        --prism_dir="$PRISM_DIR" \
        --prism_file_path="$PRISM_FILE" \
        --seed=128 \
        --task="rl_model_checking" \
        --prop='Pmax=? [ F "survival" ]' \
        --interpreter="feature_pruning;${FEAT};icu_analysis_pruning_${OUTFILE}.csv"
done

log "=========================================================================="
log "PRUNING COMPLETE"
log "=========================================================================="
