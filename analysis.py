"""Calculations for Ticker Check. No network access in here - app.py fetches the data."""
import numpy as np
import pandas as pd

# (label, trading days after the reaction day). Day 1 = close of the first trading day
# that reflects the results.
WINDOWS = [("Day 1", 0), ("Day 2", 1), ("Day 5", 4), ("Day 10", 9), ("1 month", 20)]


# ---------------------------------------------------------------- indicators
def add_indicators(df):
    df = df.copy()
    c = df["Close"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["RSI"] = 100 - 100 / (1 + gain / loss)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    return df


def _slope(s, n):
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    return s.iloc[-1] / s.iloc[-1 - n] - 1


def _gt(a, b):
    if pd.isna(a) or pd.isna(b):
        return None
    return bool(a > b)


def _bars_since_flip(s):
    s = s.dropna()
    if len(s) < 2:
        return None
    flips = s.ne(s.shift())
    flips.iloc[0] = False
    pos = np.flatnonzero(flips.to_numpy())
    return None if len(pos) == 0 else len(s) - 1 - pos[-1]


# ---------------------------------------------------- support and resistance
def support_resistance(df, lookback=252, window=5, max_levels=4):
    """Swing highs/lows over the lookback, clustered into levels. Returns (supports, resistances)."""
    d = df.tail(lookback)
    n = len(d)
    price = float(df["Close"].iloc[-1])
    if n < window * 2 + 1:
        return [], []

    highs, lows = d["High"].to_numpy(), d["Low"].to_numpy()
    pts = set()
    for i in range(window, n - window):
        if highs[i] >= highs[i - window:i + window + 1].max():
            pts.add((float(highs[i]), d.index[i]))
        if lows[i] <= lows[i - window:i + window + 1].min():
            pts.add((float(lows[i]), d.index[i]))
    pts.add((float(d["High"].max()), d["High"].idxmax()))
    pts.add((float(d["Low"].min()), d["Low"].idxmin()))

    atr = df["ATR14"].iloc[-1] if "ATR14" in df else np.nan
    tol = max(0.5 * atr, 0.01 * price) if pd.notna(atr) else 0.015 * price

    clusters = []
    for p, dt in sorted(pts, key=lambda x: x[0]):
        if clusters and p - clusters[-1]["mean"] <= tol:
            c = clusters[-1]
            c["prices"].append(p)
            c["dates"].append(dt)
            c["mean"] = float(np.mean(c["prices"]))
        else:
            clusters.append({"prices": [p], "dates": [dt], "mean": p})

    levels = [
        {
            "level": c["mean"],
            "touches": len(c["prices"]),
            "last_touch": max(c["dates"]),
            "distance": c["mean"] / price - 1,
        }
        for c in clusters
    ]

    def pick(cands):
        cands = sorted(cands, key=lambda l: abs(l["distance"]))
        strong = [l for l in cands if l["touches"] >= 2][:max_levels]
        if len(strong) < max_levels:
            strong += [l for l in cands if l not in strong][: max_levels - len(strong)]
        return sorted(strong, key=lambda l: abs(l["distance"]))

    supports = pick([l for l in levels if l["level"] < price])
    resistances = pick([l for l in levels if l["level"] > price])
    return supports, resistances


# ------------------------------------------------------------------- trend
def trend_estimate(df):
    L = df.iloc[-1]
    c = L["Close"]

    def judge(checks):
        valid = [(ok, yes if ok else no) for ok, yes, no in checks if ok is not None]
        if not valid:
            return {"label": "Not enough data", "confidence": 0.0, "reasons": []}
        score = sum(1 if ok else -1 for ok, _ in valid)
        ratio = score / len(valid)
        if ratio >= 0.6:
            label = "Up"
        elif ratio > 0:
            label = "Leaning up"
        elif ratio == 0:
            label = "Sideways"
        elif ratio > -0.6:
            label = "Leaning down"
        else:
            label = "Down"
        return {"label": label, "confidence": abs(ratio), "reasons": [t for _, t in valid]}

    s20, s50, s200 = _slope(df["SMA20"], 5), _slope(df["SMA50"], 10), _slope(df["SMA200"], 20)
    return {
        "Short (weeks)": judge([
            (_gt(c, L["SMA20"]), "Price above 20-day average", "Price below 20-day average"),
            (_gt(s20, 0), "20-day average rising", "20-day average falling"),
            (_gt(L["MACD_hist"], 0), "MACD above its signal line", "MACD below its signal line"),
        ]),
        "Medium (months)": judge([
            (_gt(c, L["SMA50"]), "Price above 50-day average", "Price below 50-day average"),
            (_gt(s50, 0), "50-day average rising", "50-day average falling"),
            (_gt(L["MACD"], 0), "MACD above zero", "MACD below zero"),
        ]),
        "Long (6m+)": judge([
            (_gt(c, L["SMA200"]), "Price above 200-day average", "Price below 200-day average"),
            (_gt(L["SMA50"], L["SMA200"]), "50-day above 200-day (golden cross)", "50-day below 200-day (death cross)"),
            (_gt(s200, 0), "200-day average rising", "200-day average falling"),
        ]),
    }


def indicator_readings(df):
    L = df.iloc[-1]
    c = L["Close"]
    out = []

    for col, name in [("SMA50", "50-day average"), ("SMA200", "200-day average")]:
        if pd.notna(L[col]):
            p = c / L[col] - 1
            out.append((name, f"{L[col]:,.2f}", f"Price {abs(p):.1%} {'above' if p >= 0 else 'below'}"))

    if pd.notna(L["SMA50"]) and pd.notna(L["SMA200"]):
        s = (df["SMA50"] > df["SMA200"]).where(df["SMA200"].notna())
        state = "Golden cross (50 above 200)" if L["SMA50"] > L["SMA200"] else "Death cross (50 below 200)"
        ago = _bars_since_flip(s)
        if ago is not None and ago <= 30:
            state += f", crossed {ago} days ago"
        out.append(("50 vs 200-day", "", state))

    r = L["RSI"]
    if pd.notna(r):
        if r < 30:
            zone = "Oversold - selling may be stretched"
        elif r < 45:
            zone = "Weak momentum"
        elif r <= 55:
            zone = "Neutral"
        elif r <= 70:
            zone = "Strong momentum"
        else:
            zone = "Overbought - may be stretched"
        out.append(("RSI (14)", f"{r:.0f}", zone))

    if pd.notna(L["MACD"]):
        above = L["MACD"] > L["MACD_signal"]
        txt = "Above signal (bullish momentum)" if above else "Below signal (bearish momentum)"
        ago = _bars_since_flip(df["MACD"] > df["MACD_signal"])
        if ago is not None and ago <= 5:
            txt += f", crossed {ago} days ago"
        out.append(("MACD (12,26,9)", f"{L['MACD']:.2f}", txt))

    if pd.notna(L["VolAvg20"]) and L["VolAvg20"] > 0:
        v = L["Volume"] / L["VolAvg20"]
        txt = "Heavy volume" if v > 1.5 else "Light volume" if v < 0.6 else "Normal volume"
        out.append(("Volume vs 20-day avg", f"{v:.1f}x", txt))

    if pd.notna(L["ATR14"]):
        out.append(("Typical daily range (ATR)", f"{L['ATR14']:,.2f}", f"About {L['ATR14'] / c:.1%} of price"))

    return pd.DataFrame(out, columns=["Indicator", "Value", "Reading"])


# ---------------------------------------------------------------- earnings
def classify_timing(ts):
    """ts is in the exchange's local time."""
    ts = pd.Timestamp(ts)
    mins = ts.hour * 60 + ts.minute
    if mins == 0:
        return "Unknown"
    if mins < 9 * 60 + 30:
        return "Before open"
    if mins >= 16 * 60:
        return "After close"
    return "During session"


def earnings_reactions(stock, index, earnings):
    """earnings: columns date (naive day), timing, eps_est, eps_act, surprise.
    Returns (meta, stock_moves, index_moves), all indexed by earnings date."""
    dates = stock.index
    metas, s_rows, i_rows = [], [], []
    for e in earnings.itertuples():
        d = pd.Timestamp(e.date)
        r = dates.searchsorted(d, side="right" if e.timing == "After close" else "left")
        if r == 0 or r >= len(dates):
            continue
        base_date, react_date = dates[r - 1], dates[r]
        base = stock["Close"].iloc[r - 1]
        ib = index["Close"].asof(base_date) if len(index) else np.nan

        sm, im = {}, {}
        sm["Open gap"] = stock["Open"].iloc[r] / base - 1
        io = index["Open"].get(react_date, np.nan) if len(index) else np.nan
        im["Open gap"] = io / ib - 1 if pd.notna(io) and pd.notna(ib) else np.nan
        for name, off in WINDOWS:
            p = r + off
            if p < len(dates):
                sm[name] = stock["Close"].iloc[p] / base - 1
                ic = index["Close"].asof(dates[p]) if len(index) else np.nan
                im[name] = ic / ib - 1 if pd.notna(ic) and pd.notna(ib) else np.nan
            else:
                sm[name] = im[name] = np.nan

        metas.append({
            "Earnings date": d.date(),
            "Timing": e.timing,
            "EPS est.": e.eps_est,
            "EPS actual": e.eps_act,
            "Surprise %": e.surprise,
            "_base": base_date,
            "_react": react_date,
        })
        s_rows.append(sm)
        i_rows.append(im)

    if not metas:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    meta = pd.DataFrame(metas).set_index("Earnings date")
    sm = pd.DataFrame(s_rows, index=meta.index)
    im = pd.DataFrame(i_rows, index=meta.index)
    order = meta.index[::-1]  # newest first
    return meta.loc[order], sm.loc[order], im.loc[order]


def premarket_move(intraday, base_date, react_date):
    """Last pre-market price on the reaction day vs the regular close on the day before.
    intraday: hourly bars incl. pre/post market, tz America/New_York."""
    if intraday is None or len(intraday) == 0:
        return np.nan
    idx = intraday.index
    mins = idx.hour * 60 + idx.minute
    day = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    pre = intraday[(day == react_date) & (mins < 570)]
    reg = intraday[(day == base_date) & (mins >= 570) & (mins < 960)]
    if pre.empty or reg.empty:
        return np.nan
    return pre["Close"].iloc[-1] / reg["Close"].iloc[-1] - 1


def summary_stats(stock_moves, idx_moves):
    diff = stock_moves - idx_moves
    return pd.DataFrame({
        "Average move": stock_moves.mean(),
        "Median move": stock_moves.median(),
        "Times up": (stock_moves > 0).sum() / stock_moves.notna().sum(),
        "Average index move": idx_moves.mean(),
        "Average vs index": diff.mean(),
        "Beat the index": (diff > 0).sum() / diff.notna().sum(),
    }).T


# --------------------------------------------------------------- scorecard
def scorecard(df, idx_df, supports, resistances, trend, idx_trend, days_to_earnings):
    items = []  # (signal, text) signal in "+", "-", "!"
    up = ("Up", "Leaning up")
    down = ("Down", "Leaning down")

    for horizon in ("Long (6m+)", "Medium (months)"):
        lab = trend[horizon]["label"]
        if lab in up:
            items.append(("+", f"{horizon.split()[0]} trend: {lab.lower()}"))
        elif lab in down:
            items.append(("-", f"{horizon.split()[0]} trend: {lab.lower()}"))

    ilab = idx_trend["Medium (months)"]["label"]
    if ilab in up:
        items.append(("+", f"Index medium trend {ilab.lower()} - market tailwind"))
    elif ilab in down:
        items.append(("-", f"Index medium trend {ilab.lower()} - market headwind"))

    L = df.iloc[-1]
    if pd.notna(L["RSI"]):
        if L["RSI"] < 30:
            items.append(("+", f"RSI {L['RSI']:.0f}: oversold, bounce potential"))
        elif L["RSI"] > 70:
            items.append(("-", f"RSI {L['RSI']:.0f}: overbought, chasing risk"))

    if pd.notna(L["MACD_hist"]) and len(df) > 1:
        prev = df["MACD_hist"].iloc[-2]
        if L["MACD_hist"] > 0 and L["MACD_hist"] >= prev:
            items.append(("+", "MACD momentum positive and building"))
        elif L["MACD_hist"] < 0 and L["MACD_hist"] <= prev:
            items.append(("-", "MACD momentum negative and building"))

    if supports and supports[0]["distance"] > -0.03:
        items.append(("+", f"Close to support at {supports[0]['level']:,.2f} ({supports[0]['distance']:+.1%})"))
    if resistances and resistances[0]["distance"] < 0.03:
        items.append(("-", f"Close to resistance at {resistances[0]['level']:,.2f} ({resistances[0]['distance']:+.1%})"))
    if supports and resistances:
        room_up, room_down = resistances[0]["distance"], -supports[0]["distance"]
        if room_down > 0:
            ratio = room_up / room_down
            if ratio >= 2:
                items.append(("+", f"About {ratio:.1f}x more room to resistance than to support"))
            elif ratio <= 0.5:
                items.append(("-", "Much less room to resistance than to support"))

    if len(df) > 63 and len(idx_df) > 63:
        rs = (df["Close"].iloc[-1] / df["Close"].iloc[-64] - 1) - (idx_df["Close"].iloc[-1] / idx_df["Close"].iloc[-64] - 1)
        if rs > 0.05:
            items.append(("+", f"Beating the index by {rs:.1%} over 3 months"))
        elif rs < -0.05:
            items.append(("-", f"Lagging the index by {abs(rs):.1%} over 3 months"))

    if days_to_earnings is not None and 0 <= days_to_earnings <= 14:
        items.append(("!", f"Earnings in {days_to_earnings} days - expect a big move either way"))
    if pd.notna(L["VolAvg20"]) and L["VolAvg20"] > 0 and L["Volume"] / L["VolAvg20"] > 1.5:
        items.append(("!", "Unusually heavy volume today"))

    score = sum(1 if s == "+" else -1 if s == "-" else 0 for s, _ in items)
    label = "Favourable" if score >= 3 else "Unfavourable" if score <= -3 else "Mixed"
    return label, score, items
