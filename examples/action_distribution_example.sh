#!/bin/bash
# Action Distribution Example (ICU Sepsis)
# Uses a trained ICU sepsis agent to compare optimal action distributions
# across two PCTL properties: survival vs death.

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Action Distribution Example (ICU Sepsis)"
echo "=========================================="

# Action distribution: survival vs death
echo ""
echo "Action Distribution: Survival vs Death..."
echo "------------------------------------------"

python cool_mc.py \
    --parent_run_id="last" \
    --project_name="icu_sepsis_ppo" \
    --constant_definitions="" \
    --prism_dir="../prism_files" \
    --prism_file_path="icu_sepsis.prism" \
    --seed=128 \
    --task="rl_model_checking" \
    --prop="Pmax=? [ F \"survival\" ]" \
    --interpreter="action_distribution;Pmax=? [ F \"survival\" ];Pmax=? [ F \"death\" ];action_distribution_results.csv"

echo ""
echo "=========================================="
echo "Action Distribution Example Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - CSV:  rl_model_checking/action_distribution_results.csv"
echo "  - Plot: rl_model_checking/action_distribution_results.png"
echo ""
