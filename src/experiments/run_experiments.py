""" the structure of this file:
1. Imports
2. Constants
   - Horizons
   - Indices
   - Models
3. main()
4. Run all experiments
5. Save results
6. Entry point """
"""
Main Experiment Runner

Runs all volatility forecasting experiments.

Workflow:
1. Load data
2. Loop through forecast horizons
3. Loop through indices
4. Loop through models
5. Train
6. Validate
7. Test
8. Save metrics
9. Save predictions
"""

from src.utils.data_loader import load_split
from src.utils.model_registry import get_model
from src.utils.results_manager import (
    save_metrics,
    save_predictions,
)
# =============================================================================
# Forecast Horizons
# =============================================================================

HORIZONS = [
    "1day",
    "5day",
    "22day",
]

# =============================================================================
# Market Indices
# =============================================================================

INDICES = [
    "^GSPC",
    "^IXIC",
    "^FTSE",
    "^GDAXI",
    "^N225",
    "^HSI",
    "000001.SS",
]

# =============================================================================
# Models
# =============================================================================

MODELS = [
    "HistoricalVolatility",
    "Persistence",

    "GARCH",
    "EGARCH",
    "GJRGARCH",

    "LinearRegression",
    "RandomForest",
    "XGBoost",

    "LSTM",
    "GRU",
    "Transformer",

    "GARCHLSTM",
    "GARCHGRU",
    "GARCHTransformer",
]
def main():

    print("=" * 60)
    print("VOLATILITY FORECASTING EXPERIMENTS")
    print("=" * 60)

    run_all_experiments()

    print("\nAll experiments completed successfully.")

def run_all_experiments():

    for horizon in HORIZONS:

        print(f"\nLoading {horizon} dataset...")

        data = load_split(horizon)

        X_train = data["X_train"]
        X_validation = data["X_validation"]
        X_test = data["X_test"]

        y_train = data["y_train"]
        y_validation = data["y_validation"]
        y_test = data["y_test"]

        for index in INDICES:

            print(f"\nIndex: {index}")

            target = f"{index}_target"

            y_train_index = y_train[target]
            y_validation_index = y_validation[target]
            y_test_index = y_test[target]

            for model_name in MODELS:

                print(f"Running {model_name}...")

                # Training happens here

                # Saving happens here
# entry point
if __name__ == "__main__":
    main()