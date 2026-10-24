"""Ticker Check - type a ticker, get levels, trend, indicators and earnings reactions.

Run:  streamlit run app.py
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import analysis as an

st.set_page_config(page_title="Ticker Check", page_icon="📈", layout="wide")

INDEXES = {"S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "FTSE 100": "^FTSE", "FTSE 250": "^FTMC"}
PREMARKET_PROXY = {"^GSPC": "SPY", "^NDX": "QQQ"}  # indices have no pre-market, ETFs do
RANGES = {"6 months": 126, "1 year": 252, "2 years": 504}
GREEN, RED = "#1a9850", "#d73027"


# ------------------------------------------------------------------ data
def _clean_daily(h):
    if h is None or h.empty:
        return pd.DataFrame()
    h = h.copy()
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    h.index = h.index.normalize()
    h = h[~h.index.duplicated(keep="last")]
    return h[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


@st.cache_data(ttl=900, show_spinner=False)
def load_daily(symbol):
    return _clean_daily(yf.Ticker(symbol).history(period="5y", interval="1d", auto_adjust=True))


@st.cache_data(ttl=3600, show_spinner=False)
def load_intraday(symbol):
    """Hourly bars incl. pre-market for ~2 years (Yahoo's limit for hourly data)."""
    try:
        end = pd.Timestamp.today().normalize() + timedelta(days=1)
        h = yf.Ticker(symbol).history(start=end - timedelta(days=725), end=end,
                                      interval="60m", prepost=True, auto_adjust=False)
        if h.empty:
            return None
        h.index = (h.index.tz_localize("America/New_York") if h.index.tz is None
                   else h.index.tz_convert("America/New_York"))
        return h[["Close"]]
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings(symbol):
    cols = ["date", "timing", "eps_est", "eps_act", "surprise"]
    try:
        ed = yf.Ticker(symbol).get_earnings_dates(limit=16)
    except Exception:
        ed = None
    if ed is None or len(ed) == 0:
        return pd.DataFrame(columns=cols)
    rows = []
    for ts, r in ed.iterrows():
        ts = pd.Timestamp(ts)
        naive = ts.tz_localize(None) if ts.tzinfo else ts
        rows.append({
            "date": naive.normalize(),
            "timing": an.classify_timing(ts),
            "eps_est": r.get("EPS Estimate", np.nan),
            "eps_act": r.get("Reported EPS", np.nan),
            "surprise": r.get("Surprise(%)", np.nan),
        })
    return pd.DataFrame(rows, columns=cols).drop_duplicates("date").sort_values("date")


@st.cache_data(ttl=86400, show_spinner=False)
def load_info(symbol):
    tk = yf.Ticker(symbol)
    name = currency = None
    try:
        currency = tk.fast_info.get("currency")
    except Exception:
        pass
    try:
        name = tk.info.get("shortName")
    except Exception:
        pass
    return name, currency


# ---------------------------------------------------------------- charts
def price_chart(d, supports, resistances, earn_dates):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.55, 0.12, 0.15, 0.18])
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="Price", showlegend=False), 1, 1)
    for col, name, color, dash in [("SMA20", "20-day", "#9e9e9e", "dot"),
                                   ("SMA50", "50-day", "#1f77b4", "solid"),
                                   ("SMA200", "200-day", "#ff7f0e", "solid")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=name,
                                 line=dict(color=color, width=1.3, dash=dash)), 1, 1)
    for lv in supports:
        fig.add_hline(y=lv["level"], line=dict(color=GREEN, width=1, dash="dash"), row=1, col=1,
                      annotation_text=f"S {lv['level']:,.2f}", annotation_position="bottom right")
    for lv in resistances:
        fig.add_hline(y=lv["level"], line=dict(color=RED, width=1, dash="dash"), row=1, col=1,
                      annotation_text=f"R {lv['level']:,.2f}", annotation_position="top right")

    ed = [x for x in earn_dates if d.index[0] <= x <= d.index[-1]]
    if ed:
        ys = [d["High"].asof(x) * 1.04 for x in ed]
        fig.add_trace(go.Scatter(x=ed, y=ys, mode="markers+text", text=["E"] * len(ed),
                                 textposition="top center", name="Earnings",
                                 marker=dict(symbol="triangle-down", size=9, color="#7b1fa2")), 1, 1)

    vol_col = [GREEN if c >= o else RED for o, c in zip(d["Open"], d["Close"])]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vol_col, showlegend=False, name="Volume"), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["VolAvg20"], line=dict(color="#555", width=1),
                             showlegend=False, name="Avg volume"), 2, 1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], line=dict(color="#7b1fa2", width=1.3),
                             showlegend=False, name="RSI"), 3, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color="#999", width=1, dash="dot"), row=3, col=1)

    hist = d["MACD_hist"].fillna(0)
    fig.add_trace(go.Bar(x=d.index, y=hist, marker_color=[GREEN if v >= 0 else RED for v in hist],
                         showlegend=False, name="MACD hist"), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], line=dict(color="#1f77b4", width=1.2),
                             showlegend=False, name="MACD"), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_signal"], line=dict(color="#ff7f0e", width=1.2),
                             showlegend=False, name="Signal"), 4, 1)

    fig.update_layout(height=800, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.03, x=0))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    return fig


def relative_chart(stock, index, n, ticker, idx_name):
    s = stock["Close"].tail(n)
    i = index["Close"].reindex(s.index, method="ffill")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] - 1, name=ticker, line=dict(width=2)))
    fig.add_trace(go.Scatter(x=s.index, y=i / i.iloc[0] - 1, name=idx_name, line=dict(width=1.5, dash="dot")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat="+.0%",
                      hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# --------------------------------------------------------------- styling
def _colour(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"color: {GREEN}" if v > 0 else f"color: {RED}"


def _map(styler, func, subset):
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(func, subset=subset)


def style_moves(df, move_cols):
    sty = df.style.format({c: "{:+.1%}" for c in move_cols}, na_rep="–")
    extra = {"EPS est.": "{:.2f}", "EPS actual": "{:.2f}", "Surprise %": "{:+.1f}%"}
    sty = sty.format({k: v for k, v in extra.items() if k in df.columns}, na_rep="–")
    return _map(sty, _colour, move_cols)


# ------------------------------------------------------------------- app
st.sidebar.title("📈 Ticker Check")
ticker = st.sidebar.text_input("Ticker", "AAPL", help="US: AAPL, NVDA. FTSE: add .L, e.g. VOD.L, BP.L").strip().upper()
index_choice = st.sidebar.selectbox("Compare against", ["Auto"] + list(INDEXES))
chart_range = st.sidebar.radio("Chart range", list(RANGES), index=1, horizontal=True)
try_premarket = st.sidebar.checkbox("Fetch real pre-market moves (US only, slower)", value=True)
st.sidebar.caption("Data from Yahoo Finance via yfinance, refreshed every 15 minutes. Not financial advice.")

if not ticker:
    st.stop()

idx_sym = INDEXES[index_choice] if index_choice != "Auto" else ("^FTSE" if ticker.endswith(".L") else "^GSPC")
idx_name = next(k for k, v in INDEXES.items() if v == idx_sym)

with st.spinner(f"Loading {ticker}..."):
    stock = load_daily(ticker)
    index = load_daily(idx_sym)
    earn = load_earnings(ticker)
    name, currency = load_info(ticker)

if stock.empty or len(stock) < 30:
    st.error(f"No price data found for **{ticker}**. Check the ticker - FTSE stocks need .L (e.g. VOD.L).")
    st.stop()

stock = an.add_indicators(stock)
index = an.add_indicators(index) if not index.empty else index
supports, resistances = an.support_resistance(stock)
trend = an.trend_estimate(stock)
idx_trend = an.trend_estimate(index) if len(index) > 30 else {k: {"label": "n/a", "confidence": 0, "reasons": []} for k in trend}

today = pd.Timestamp.today().normalize()
past_earn = earn[(earn["date"] < today) & (earn["date"] >= today - pd.Timedelta(days=731))]
future_earn = earn[earn["date"] >= today]
next_earn = future_earn["date"].min() if len(future_earn) else None
days_to_earn = int((next_earn - today).days) if next_earn is not None else None

# ---- header
last, prev = stock.iloc[-1], stock.iloc[-2]
yr = stock.tail(252)
cur = f" {currency}" if currency else ""
st.title(ticker + (f" · {name}" if name else ""))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"{last['Close']:,.2f}{cur}", f"{last['Close'] / prev['Close'] - 1:+.2%}")
pos = (last["Close"] - yr["Low"].min()) / (yr["High"].max() - yr["Low"].min())
c2.metric("52-week range", f"{yr['Low'].min():,.2f} – {yr['High'].max():,.2f}", f"{pos:.0%} of the way up", delta_color="off")
if len(stock) > 63 and len(index) > 63:
    s3 = last["Close"] / stock["Close"].iloc[-64] - 1
    i3 = index["Close"].iloc[-1] / index["Close"].iloc[-64] - 1
    c3.metric("3 months", f"{s3:+.1%}", f"{s3 - i3:+.1%} vs {idx_name}")
c4.metric("Next earnings", next_earn.strftime("%d %b %Y") if next_earn is not None else "Unknown",
          f"in {days_to_earn} days" if days_to_earn is not None else None, delta_color="off")

# ---- snapshot
label, score, items = an.scorecard(stock, index if len(index) else stock, supports, resistances,
                                   trend, idx_trend, days_to_earn)
colour = {"Favourable": GREEN, "Unfavourable": RED}.get(label, "#e6a700")
st.subheader("Snapshot")
st.markdown(f"<div style='font-size:1.4rem;font-weight:600;color:{colour}'>Entry conditions: {label} "
            f"<span style='font-size:1rem;color:#888'>(score {score:+d})</span></div>", unsafe_allow_html=True)
icons = {"+": "✅", "-": "❌", "!": "⚠️"}
st.markdown("\n".join(f"- {icons[s]} {t}" for s, t in items) or "_No strong signals either way._")
st.caption("A summary of the signals below to check against your own plan - not a prediction or advice.")

# ---- chart
st.subheader("Chart")
n = RANGES[chart_range]
st.plotly_chart(price_chart(stock.tail(n), supports, resistances, list(past_earn["date"])), use_container_width=True)

# ---- levels
st.subheader("Support & resistance")
st.caption("Price levels where the stock has repeatedly turned over the last year. More touches = stronger level.")


def levels_table(levels):
    if not levels:
        return pd.DataFrame({"Level": ["None found"]})
    return pd.DataFrame([{"Level": f"{l['level']:,.2f}", "Distance": f"{l['distance']:+.1%}",
                          "Touches": l["touches"], "Last tested": l["last_touch"].strftime("%d %b %Y")}
                         for l in levels])


lc, rc = st.columns(2)
lc.markdown("**Resistance (above)**")
lc.dataframe(levels_table(resistances), hide_index=True, use_container_width=True)
rc.markdown("**Support (below)**")
rc.dataframe(levels_table(supports), hide_index=True, use_container_width=True)

# ---- trend
st.subheader("Trend estimate")
tc1, tc2 = st.columns(2)
for col, title, tr in [(tc1, ticker, trend), (tc2, idx_name, idx_trend)]:
    col.markdown(f"**{title}**")
    for horizon, t in tr.items():
        col.markdown(f"{horizon}: **{t['label']}** · {t['confidence']:.0%} of signals agree")
        if t["reasons"]:
            col.caption(" · ".join(t["reasons"]))

# ---- indicators
st.subheader("Indicators")
st.dataframe(an.indicator_readings(stock), hide_index=True, use_container_width=True)

# ---- earnings
st.subheader("Earnings reactions (last 2 years)")
if past_earn.empty:
    st.info("Yahoo has no earnings history for this ticker - common for FTSE stocks. Everything else above still applies.")
else:
    meta, sm, im = an.earnings_reactions(stock, index, past_earn)
    if meta.empty:
        st.info("Not enough price history around the earnings dates.")
    else:
        if try_premarket and not ticker.endswith(".L"):
            with st.spinner("Fetching pre-market data..."):
                intr = load_intraday(ticker)
                proxy = PREMARKET_PROXY.get(idx_sym)
                intr_idx = load_intraday(proxy) if proxy else None
            sm.insert(0, "Pre-market", [an.premarket_move(intr, b, r) for b, r in zip(meta["_base"], meta["_react"])])
            im.insert(0, "Pre-market", [an.premarket_move(intr_idx, b, r) for b, r in zip(meta["_base"], meta["_react"])])
        diff = sm - im
        move_cols = list(sm.columns)
        info = meta[["Timing", "EPS est.", "EPS actual", "Surprise %"]]

        t1, t2, t3 = st.tabs([ticker, idx_name + " (same days)", f"{ticker} minus index"])
        with t1:
            st.dataframe(style_moves(info.join(sm), move_cols), use_container_width=True)
        with t2:
            st.dataframe(style_moves(meta[["Timing"]].join(im), move_cols), use_container_width=True)
        with t3:
            st.dataframe(style_moves(meta[["Timing"]].join(diff), move_cols), use_container_width=True)

        st.markdown("**Across these results**")
        summ = an.summary_stats(sm, im)
        signed = ["Average move", "Median move", "Average index move", "Average vs index"]
        sty = summ.style.format("{:+.1%}", subset=pd.IndexSlice[signed, :], na_rep="–")
        sty = sty.format("{:.0%}", subset=pd.IndexSlice[["Times up", "Beat the index"], :], na_rep="–")
        st.dataframe(sty, use_container_width=True)
        st.caption("All moves are measured from the close before the results. Open gap = next open (the overnight / "
                   "pre-market reaction). Pre-market = last pre-market price before the bell, where Yahoo has it. "
                   "Day 1 = close of the first day after results. 1 month ≈ 21 trading days. Blank = not happened yet.")

# ---- vs index
st.subheader(f"Performance vs {idx_name}")
if len(index):
    st.plotly_chart(relative_chart(stock, index, n, ticker, idx_name), use_container_width=True)

st.caption("For information only. Not financial advice - decisions and risk are yours.")
