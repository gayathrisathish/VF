import yfinance as yf

tickers = [
    "^GSPC",      # S&P 500
    "^IXIC",      # NASDAQ Composite
    "^FTSE",      # FTSE 100
    "^GDAXI",     # DAX
    "^N225",      # Nikkei 225
    "^HSI",       # Hang Seng
    "000001.SS"   # Shanghai Composite
]

data = yf.download(
    tickers,
    start="2005-01-01",
    end="2025-12-31",
    group_by="ticker",
    auto_adjust=True
)
data.to_csv("data/raw/global_indices.csv")