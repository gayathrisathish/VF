python3 - <<'PY'
from src.utils.best_params import load_best_params

for model in ["LSTM", "GRU", "Transformer"]:
    for horizon in ["1day", "5day", "22day"]:
        try:
            p = load_best_params(model, horizon)
            print(f"[OK] {model} {horizon}: {p}")
        except Exception as e:
            print(f"[BAD] {model} {horizon}: {e}")
PY
