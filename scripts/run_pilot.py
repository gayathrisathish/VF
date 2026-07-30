"""
Driver to run a targeted pilot benchmark by overriding the constants
in the production experiment runner. Does not modify the original
runner source.

Configuration (per request):
- Horizons: ["1day"]
- Indices: ["^GSPC"]
- Models: use default (all 14 registered models)

Saves output printed to stdout and relies on the runner's save_metrics/save_predictions
for persistence.
"""
import os
import sys
import time
import traceback

# Ensure project root is on PYTHONPATH so `src` is importable when running this script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments import run_experiments


def main():
    start_time = time.time()

    # Override runner configuration for the pilot
    run_experiments.HORIZONS = ["1day"]
    run_experiments.INDICES = ["^GSPC"]

    print("Starting pilot benchmark: Horizons=", run_experiments.HORIZONS, "Indices=", run_experiments.INDICES)

    try:
        summary = run_experiments.run_all_experiments()
    except Exception as e:
        print("Run failed with exception:")
        traceback.print_exc()
        raise

    elapsed = time.time() - start_time
    print("\nPilot summary:")
    print(summary)
    print(f"Elapsed time: {elapsed:.2f} seconds")

    return summary, elapsed


if __name__ == '__main__':
    main()
