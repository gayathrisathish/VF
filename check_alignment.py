import pandas as pd

garch = pd.read_csv("results/predictions/garch_forecast_h1.csv", parse_dates=["Date"])
target = pd.read_csv("data/features/features_1day.csv", parse_dates=["Date"])[["Date", "^GSPC_target"]]

merged = garch.merge(target, on="Date")
print(merged.corr())

# also check a few small shifts to see if correlation improves — if shifting target by 
# 1-3 days gives meaningfully higher correlation than the unshifted version, that's a sign
# of a real offset bug, not just natural GARCH lag
for shift in [-3, -2, -1, 0, 1, 2, 3]:
    shifted_corr = merged["garch_forecast_h1"].corr(merged["^GSPC_target"].shift(shift))
    print(f"shift={shift}: corr={shifted_corr:.4f}")