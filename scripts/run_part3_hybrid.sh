#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Part 3 - GARCH-hybrid deep-learning models
#
# Models  : GARCHLSTM, GARCHGRU, GARCHTransformer
# Horizons: 1day, 5day, 22day
#
# Runs the existing training pipeline through its CLI:
#   python -m src.experiments.run_experiments --models <MODEL> --horizons <H>
#
# Tuned hyperparameters are loaded automatically by the runner through
# src/utils/best_params.py::load_best_params, which reads
#   results/best_params/best_params_<study>_<horizon>.json
# A missing study file makes the run fail; it never falls back to defaults.
#
# Fitted Keras models are written by results_manager.py::save_model to
#   results/models/<horizon>/<index>/<Model>.keras
#
# This wrapper does not tune and does not commit anything.
# =============================================================================

# --- Resolve the repository root, whatever the current directory is ---------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=============================================================="
echo "  PART 3 - GARCH-HYBRID DEEP-LEARNING MODELS"
echo "=============================================================="
echo "Repository root : ${REPO_ROOT}"

# --- Require the existing training entry point ------------------------------
ENTRY_POINT="src/experiments/run_experiments.py"
if [[ ! -f "${ENTRY_POINT}" ]]; then
    echo "ERROR: Missing training entry point: ${ENTRY_POINT}" >&2
    exit 1
fi

# --- Activate the existing virtual environment ------------------------------
# Override with:  VF_VENV=/path/to/venv ./scripts/run_part3_hybrid.sh
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    VENV_CANDIDATES=()
    if [[ -n "${VF_VENV:-}" ]]; then
        VENV_CANDIDATES+=("${VF_VENV}")
    fi
    VENV_CANDIDATES+=("${HOME}/vf-venv" "${REPO_ROOT}/.venv" "${REPO_ROOT}/venv")

    ACTIVATED=""
    for CANDIDATE in "${VENV_CANDIDATES[@]}"; do
        if [[ -f "${CANDIDATE}/bin/activate" ]]; then
            # shellcheck disable=SC1091
            source "${CANDIDATE}/bin/activate"
            ACTIVATED="${CANDIDATE}"
            break
        fi
    done

    if [[ -z "${ACTIVATED}" ]]; then
        echo "ERROR: No virtual environment found." >&2
        echo "       Checked: ${VENV_CANDIDATES[*]}" >&2
        echo "       Create the GPU environment with: bash setup_wsl.sh" >&2
        echo "       Or set: export VF_VENV=/path/to/venv" >&2
        exit 1
    fi
    echo "Virtual environment : ${ACTIVATED}"
else
    echo "Virtual environment : ${VIRTUAL_ENV} (already active)"
fi

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: 'python' not found after activating the environment." >&2
    exit 1
fi

echo "Python              : $(python --version 2>&1)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

# --- Models and horizons ----------------------------------------------------
MODELS=("GARCHLSTM" "GARCHGRU" "GARCHTransformer")
HORIZONS=("1day" "5day" "22day")

# --- Report GPU visibility (informational) ----------------------------------
echo "--------------------------------------------------------------"
python - <<'PYTHON_EOF'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow          :", tf.__version__)
print("GPUs visible        :", len(gpus))
for gpu in gpus:
    print("  -", gpu.name)
if not gpus:
    print("  (no GPU detected - training will run on CPU)")
PYTHON_EOF

# --- Require the GARCH-augmented sequence data ------------------------------
GARCH_DATA_ROOT="data/garch_augmented_sequence_datasets"
if [[ ! -d "${GARCH_DATA_ROOT}" ]]; then
    echo "ERROR: Missing GARCH-augmented sequence data: ${GARCH_DATA_ROOT}" >&2
    exit 1
fi
echo "--------------------------------------------------------------"
echo "GARCH sequence data : ${GARCH_DATA_ROOT}"

# --- Preflight: every tuned study file must be present and readable ---------
# Fails before any training starts, not part-way through the grid.
echo "--------------------------------------------------------------"
echo "Checking saved best parameters..."
python - "${MODELS[@]}" -- "${HORIZONS[@]}" <<'PYTHON_EOF'
import sys

from src.utils.best_params import tuning_coverage

separator = sys.argv.index("--")
models = sys.argv[1:separator]
horizons = sys.argv[separator + 1:]

coverage = tuning_coverage(models, horizons)
missing = []

for (model, horizon), status in coverage.items():
    if status["present"] and status["readable"]:
        print(f"  [ok]      {model} {horizon} -> {status['path']}")
    else:
        reason = "bad" if status["present"] else "missing"
        print(f"  [{reason}] {model} {horizon} -> {status['path']}")
        missing.append(f"{model} {horizon}")

if missing:
    print("", file=sys.stderr)
    print(
        "ERROR: tuned parameters are missing or unreadable for: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    print(
        "Run the corresponding tuning scripts before this wrapper.",
        file=sys.stderr,
    )
    sys.exit(1)

print("All required best-parameter files are present and readable.")
PYTHON_EOF

TOTAL=$(( ${#MODELS[@]} * ${#HORIZONS[@]} ))
echo "--------------------------------------------------------------"
echo "Models              : ${MODELS[*]}"
echo "Horizons            : ${HORIZONS[*]}"
echo "Training runs       : ${TOTAL}"
echo "--------------------------------------------------------------"

# --- One process per (model, horizon) so a failure stops the run at once ----
STEP=0
for HORIZON in "${HORIZONS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        STEP=$(( STEP + 1 ))
        echo ""
        echo "[${STEP}/${TOTAL}] RUNNING model=${MODEL} horizon=${HORIZON}"
        echo "--------------------------------------------------------------"

        python -m src.experiments.run_experiments \
            --models "${MODEL}" \
            --horizons "${HORIZON}"

        echo "[${STEP}/${TOTAL}] DONE    model=${MODEL} horizon=${HORIZON}"
    done
done

echo ""
echo "=============================================================="
echo "  PART 3 COMPLETE"
echo "=============================================================="
echo "Fitted models : results/models/<horizon>/<index>/<Model>.keras"
echo "Metrics       : results/metrics/metrics.csv"
echo "Predictions   : results/predictions/predictions_<horizon>.csv"
