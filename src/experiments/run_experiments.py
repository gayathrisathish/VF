"""
Main experiment runner.

Runs all:
- horizons
- indices
- models
"""

HORIZONS = [
    "1day",
    "5day",
    "22day"
]

INDICES = [
    "^GSPC",
    "^IXIC",
    "^FTSE",
    "^GDAXI",
    "^N225",
    "^HSI",
    "000001.SS"
]

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
    "GARCHTransformer"
]


def main():

    for horizon in HORIZONS:

        print(f"\nRunning {horizon}")

        for index in INDICES:

            print(f"   {index}")

            for model in MODELS:

                print(f"      {model}")

                # load data

                # train

                # validate

                # test

                # save predictions

                # save metrics


if __name__ == "__main__":
    main()