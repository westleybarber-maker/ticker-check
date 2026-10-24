# Ticker Check

Type a ticker and get a chart with support/resistance, trend estimates, indicator readings,
and how the stock reacted to its last 2 years of earnings compared with the index.

## Setup (once)

1. Install Python 3.10+ from python.org.
2. Put `app.py`, `analysis.py` and `requirements.txt` in one folder.
3. Open a terminal in that folder and run:

```
pip install -r requirements.txt
```

## Run

```
streamlit run app.py
```

It opens in your browser. Type a ticker in the sidebar:
- US: `AAPL`, `NVDA`, `MSFT`
- FTSE: add `.L`, e.g. `VOD.L`, `BP.L`, `AZN.L`

The index is chosen automatically (S&P 500 for US, FTSE 100 for `.L`), or pick one in the sidebar.

## Notes

- Data comes from Yahoo Finance via yfinance: free, no key, no daily cap, refreshed every 15 minutes.
- Earnings moves are measured from the close before results. "Open gap" is the overnight/pre-market
  reaction; "Pre-market" uses Yahoo's hourly pre-market data where it exists (US only).
- FTSE earnings history is often missing on Yahoo; the rest of the page still works.
- yfinance is unofficial, so if Yahoo changes something it may need `pip install -U yfinance`.
- For information only. Not financial advice.
