#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Part 1 - Baselines and classical models
#
# Models  : HistoricalVolatility, Persistence, GARCH, EGARCH, GJRGARCH,
#           LinearRegression, RandomForest, XGBoost
# Horizons: 1day, 5day, 22day
#
# Runs the existing training pipeline through its CLI:
#   python -m src.experiments.run_experiments --models <MODEL> --horizons <H>
#
# Fitted models are written by src/utils/results_manager.py::save_model to
#   results/models/<horizon>/<index>/<Model>.joblib
#
# This wrapper does not tune, does not evaluate separately, and does not
# commit anything.
# =============================================================================

# --- Resolve the repository root, whatever the current directory is ---------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=============================================================="
echo "  PART 1 - BASELINES AND CLASSICAL MODELS"
echo "=============================================================="
echo "Repository root : ${REPO_ROOT}"

# --- Require the existing training entry point ------------------------------
ENTRY_POINT="src/experiments/run_experiments.py"
if [[ ! -f "${ENTRY_POINT}" ]]; then
    echo "ERROR: Missing training entry point: ${ENTRY_POINT}" >&2
    exit 1
fi

# --- Activate the existing virtual environment ------------------------------
# Override with:  VF_VENV=/path/to/venv ./scripts/run_part1_baselines.sh
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
        echo "       Create one with: bash setup_wsl.sh" >&2
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

# --- Models and horizons ----------------------------------------------------
MODELS=(
    "HistoricalVolatility"
    "Persistence"
    "GARCH"
    "EGARCH"
    "GJRGARCH"
    "LinearRegression"
    "RandomForest"
    "XGBoost"
)

HORIZONS=("1day" "5day" "22day")

TOTAL=$(( ${#MODELS[@]} * ${#HORIZONS[@]} ))
echo "Models              : ${#MODELS[@]}"
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
echo "  PART 1 COMPLETE"
echo "=============================================================="
echo "Fitted models : results/models/<horizon>/<index>/<Model>.joblib"
echo "Metrics       : results/metrics/metrics.csv"
echo "Predictions   : results/predictions/predictions_<horizon>.csv"
echo ""
echo "NOTE: HistoricalVolatility and Persistence are rule-based and have no"
echo "      fitted estimator; the runner prints [NO MODEL FILE] for them."
