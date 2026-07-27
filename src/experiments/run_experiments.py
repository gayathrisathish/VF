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