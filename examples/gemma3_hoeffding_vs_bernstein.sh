#!/bin/bash
# Probabilistic model checking of LLM-based policies, using the two
# i.i.d. concentration bounds from the paper:
#   * Hoeffding                 (distribution-free)
#   * Empirical Bernstein       (Maurer-Pontil 2009, variance-adaptive)
#
# For each environment listed in ENVIRONMENTS, runs four experiments:
#   1. Single policy gemma3:1b verified with Hoeffding.
#   2. Single policy gemma3:1b verified with Empirical Bernstein.
#   3. Pairwise (gemma3:1b vs gemma3:4b) joint comparison with
#      Hoeffding (alpha split across both policies via num_models=K=2).
#   4. Same pairwise comparison with Empirical Bernstein.
#
# Each experiment's stdout/stderr is tee'd to its own log file under
# logs/llm_policy_verification_<timestamp>/.
#
# Configurable env vars (with defaults):
#   ALPHA         : joint failure probability   (default: 0.1)
#   SAMPLES       : per-state policy queries N  (default: 20)
#   MODEL_A       : first Ollama model          (default: gemma3:1b)
#   MODEL_B       : second Ollama model         (default: gemma3:4b)
#   TEMPERATURE_A : sampling temperature for MODEL_A in the pairwise
#                   comparison (default: 0.7)
#   TEMPERATURE_B : sampling temperature for MODEL_B in the pairwise
#                   comparison (default: 0.7)
#   SINGLE_TEMPERATURES : space-separated list of temperatures to sweep
#                   over for the single-policy runs of MODEL_A. Each
#                   value yields its own pair of single-policy
#                   experiments (Hoeffding + Empirical Bernstein), so
#                   the bounds can be compared like-for-like across
#                   policy sharpness. (default: "0 0.2 0.4 0.6 0.8")
#   OLLAMA_HOST   : Ollama URL                  (auto-detected)
#   SEED          : random seed                 (default: 128)
#   PRISM_DIR     : path to PRISM files         (default: ../prism_files)
#   LOG_DIR       : output directory            (default: logs/llm_policy_verification_<timestamp>)
#   ENVIRONMENTS  : space-separated list of environment tags to run.
#                   Recognised: "frozen_lake", "wolf_goat_cabbage".
#                   Default: "frozen_lake wolf_goat_cabbage".
set -euo pipefail

cd "$(dirname "$0")/.."

ALPHA="${ALPHA:-0.1}"
SAMPLES="${SAMPLES:-20}"
MODEL_A="${MODEL_A:-gemma3:1b}"
MODEL_B="${MODEL_B:-gemma3:4b}"
TEMPERATURE_A="${TEMPERATURE_A:-0.7}"
TEMPERATURE_B="${TEMPERATURE_B:-0.7}"
SINGLE_TEMPERATURES="${SINGLE_TEMPERATURES:-0 0.2 0.4 0.6 0.8}"
SEED="${SEED:-128}"
PRISM_DIR="${PRISM_DIR:-../prism_files}"
ENVIRONMENTS="${ENVIRONMENTS:-frozen_lake wolf_goat_cabbage}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/llm_policy_verification_${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"
echo "Per-experiment logs will be written under: ${LOG_DIR}"

probe_host() {
    python - "$1" <<'PY' 2>/dev/null
import socket, sys, urllib.parse
u = urllib.parse.urlparse(sys.argv[1])
try:
    socket.create_connection((u.hostname, u.port or 11434), timeout=2).close()
except Exception:
    sys.exit(1)
PY
}

if [ -z "${OLLAMA_HOST:-}" ]; then
    for candidate in "http://host.docker.internal:11434" "http://172.17.0.1:11434" "http://localhost:11434"; do
        if probe_host "$candidate"; then
            OLLAMA_HOST="$candidate"
            break
        fi
    done
fi

if [ -z "${OLLAMA_HOST:-}" ]; then
    echo "ERROR: could not reach an Ollama server."
    echo "Start the container with --add-host=host.docker.internal:host-gateway"
    echo "or set OLLAMA_HOST=http://<host>:11434 and re-run."
    exit 1
fi

echo "Using OLLAMA_HOST=${OLLAMA_HOST}"

# ---------------------------------------------------------------- env configs

# Each environment is described by four variables:
#   ENV_<tag>_PRISM_FILE  - PRISM model filename
#   ENV_<tag>_PROMPT      - prompt template path (relative to PRISM_DIR)
#   ENV_<tag>_PROP        - PCTL property
#   ENV_<tag>_CONSTANTS   - PRISM constant_definitions (may be empty)
ENV_frozen_lake_PRISM_FILE="frozen_lake.prism"
ENV_frozen_lake_PROMPT="${PRISM_DIR}/frozen_lake_prompt.txt"
ENV_frozen_lake_PROP='P=? [ F "at_frisbee" ]'
ENV_frozen_lake_CONSTANTS="control=0.3333,start_position=0"

ENV_wolf_goat_cabbage_PRISM_FILE="wolf_goat_cabbage.prism"
ENV_wolf_goat_cabbage_PROMPT="${PRISM_DIR}/wolf_goat_cabbage_prompt.txt"
ENV_wolf_goat_cabbage_PROP='P=? [ F "won" ]'
ENV_wolf_goat_cabbage_CONSTANTS=""

# ---------------------------------------------------------------- run helpers

run_single() {
    local tag="$1"
    local bound="$2"
    local temperature="$3"
    local prism_file="$4"
    local prompt="$5"
    local prop="$6"
    local constants="$7"
    local policy="ollama_agent;model=${MODEL_A};temperature=${temperature};prompt_file=${prompt};host=${OLLAMA_HOST};timeout=120;keep_alive=30s;debug_first_n=2"
    local env_dir="${LOG_DIR}/${tag}"
    local temp_tag="T${temperature//./_}"
    mkdir -p "${env_dir}"
    local log_file="${env_dir}/single_${MODEL_A//[:\/]/_}_${temp_tag}_${bound}.log"

    {
        echo "=========================================="
        echo " [${tag}] Single-policy IDTMC verification"
        echo "   policy           : ${MODEL_A}  (T=${temperature})"
        echo "   bound            : ${bound}"
        echo "   joint confidence : $(python -c "print(1-${ALPHA})") (alpha=${ALPHA})"
        echo "   samples / state  : ${SAMPLES}"
        echo "   prism            : ${prism_file}"
        echo "   property         : ${prop}"
        echo "   constants        : ${constants}"
        echo "   log file         : ${log_file}"
        echo "=========================================="

        local cmd=(
            python cool_mc.py
            --task="rl_model_checking"
            --project_name="${tag}_${temp_tag}_${bound}"
            --algorithm="${policy}"
            --prism_dir="${PRISM_DIR}"
            --prism_file_path="${prism_file}"
            --prop="${prop}"
            --transition_updater="${bound};alpha=${ALPHA};samples=${SAMPLES};seed=${SEED}"
            --seed="${SEED}"
        )
        if [ -n "${constants}" ]; then
            cmd+=(--constant_definitions="${constants}")
        fi
        "${cmd[@]}"
    } 2>&1 | tee "${log_file}"
}

run_compare() {
    local tag="$1"
    local bound="$2"
    local prism_file="$3"
    local prompt="$4"
    local prop="$5"
    local constants="$6"
    local policy_a="ollama_agent;model=${MODEL_A};temperature=${TEMPERATURE_A};prompt_file=${prompt};host=${OLLAMA_HOST};timeout=120;keep_alive=30s;debug_first_n=2"
    local policy_b="ollama_agent;model=${MODEL_B};temperature=${TEMPERATURE_B};prompt_file=${prompt};host=${OLLAMA_HOST};timeout=120;keep_alive=30s;debug_first_n=2"
    local env_dir="${LOG_DIR}/${tag}"
    mkdir -p "${env_dir}"
    local log_file="${env_dir}/compare_${MODEL_A//[:\/]/_}_vs_${MODEL_B//[:\/]/_}_${bound}.log"

    {
        echo "=========================================="
        echo " [${tag}] Two-policy joint IDTMC comparison"
        echo "   policy A         : ${MODEL_A}  (T=${TEMPERATURE_A})"
        echo "   policy B         : ${MODEL_B}  (T=${TEMPERATURE_B})"
        echo "   bound            : ${bound}"
        echo "   joint confidence : $(python -c "print(1-${ALPHA})") (alpha=${ALPHA})"
        echo "   samples / policy : ${SAMPLES}"
        echo "   prism            : ${prism_file}"
        echo "   property         : ${prop}"
        echo "   constants        : ${constants}"
        echo "   log file         : ${log_file}"
        echo "   Note: compare_idtmcs.py passes num_models=K=2 to the"
        echo "   transition updater, so each per-action CI is built at"
        echo "   alpha_per = alpha / (|Act| * |S| * 2). Both IDTMCs"
        echo "   jointly contain their true induced DTMCs with probability"
        echo "   at least 1 - alpha."
        echo "=========================================="

        pushd rl_model_checking > /dev/null
        local cmd=(
            python compare_idtmcs.py
            --prism_dir="${PRISM_DIR}"
            --prism_file_path="${prism_file}"
            --prop="${prop}"
            --alpha="${ALPHA}"
            --samples="${SAMPLES}"
            --seed="${SEED}"
            --bound="${bound}"
            --algorithms="${policy_a}#${policy_b}"
            --labels="${MODEL_A//[:\/]/_},${MODEL_B//[:\/]/_}"
            --project_name="compare_${tag}_${bound}"
        )
        if [ -n "${constants}" ]; then
            cmd+=(--constant_definitions="${constants}")
        fi
        "${cmd[@]}"
        popd > /dev/null
    } 2>&1 | tee "${log_file}"
}

# ---------------------------------------------------------------- driver

resolve_env_var() {
    # Indirect lookup of ENV_<tag>_<field>; aborts if missing.
    local tag="$1"
    local field="$2"
    local var="ENV_${tag}_${field}"
    if [ -z "${!var+x}" ]; then
        echo "ERROR: unknown environment '${tag}' (no ${var} defined)."
        exit 1
    fi
    echo "${!var}"
}

for tag in ${ENVIRONMENTS}; do
    echo
    echo "############################################################"
    echo "# Environment: ${tag}"
    echo "############################################################"
    prism_file=$(resolve_env_var "${tag}" "PRISM_FILE")
    prompt=$(resolve_env_var "${tag}" "PROMPT")
    prop=$(resolve_env_var "${tag}" "PROP")
    constants=$(resolve_env_var "${tag}" "CONSTANTS")

    for temperature in ${SINGLE_TEMPERATURES}; do
        run_single "${tag}" hoeffding           "${temperature}" "${prism_file}" "${prompt}" "${prop}" "${constants}"
        run_single "${tag}" empirical_bernstein "${temperature}" "${prism_file}" "${prompt}" "${prop}" "${constants}"
    done
    run_compare "${tag}" hoeffding           "${prism_file}" "${prompt}" "${prop}" "${constants}"
    run_compare "${tag}" empirical_bernstein "${prism_file}" "${prompt}" "${prop}" "${constants}"
done

echo
echo "All experiments completed."
echo "Logs are in: ${LOG_DIR}"
find "${LOG_DIR}" -type f -name '*.log' | sort
