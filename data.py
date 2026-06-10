"""Market data fetch + ALL quantitative metrics pre-computed in Python.
LLM agents only interpret these numbers; they never compute."""
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import ADXIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands


def hurst_exponent(prices: pd.Series, max_lag: int = 64) -> float:
    """Hurst via rescaled-range slope on log prices."""
    logp = np.log(prices.values)
    lags = range(2, max_lag)
    tau = [np.std(logp[lag:] - logp[:-lag]) for lag in lags]
    slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    return round(float(slope), 3)


def fractal_dimension(prices: pd.Series) -> float:
    """Katz fractal dimension of the recent price path (~1.0 smooth trend,
    ~1.5 random walk, higher = jagged/turbulent)."""
    y = prices.values
    n = len(y)
    L = np.sum(np.sqrt(1 + np.diff(y) ** 2))
    d = np.max(np.sqrt(np.arange(n) ** 2 + (y - y[0]) ** 2))
    return round(float(np.log10(n) / (np.log10(n) + np.log10(d / L))), 3)


def fetch_news(symbol: str, max_items: int = 8) -> list:
    """Recent headlines for the Social Change analyst."""
    try:
        items = yf.Ticker(symbol).news or []
        out = []
        for it in items[:max_items]:
            c = it.get("content", it)
            out.append({"title": c.get("title", ""),
                        "published": str(c.get("pubDate", c.get("providerPublishTime", "")))})
        return out
    except Exception:
        return []


def fetch_market_data(symbol: str) -> dict:
    df = yf.download(symbol, period="2y", interval="1d", auto_adjust=True,
                     progress=False)
    if df.empty:
        raise RuntimeError(f"no data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close, high, low = df["Close"], df["High"], df["Low"]
    last = float(close.iloc[-1])
    ret = close.pct_change()

    adx = ADXIndicator(high, low, close, window=14).adx().iloc[-1]
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    bb = BollingerBands(close, window=20, window_dev=2)

    def pct(days):
        return round(float(close.iloc[-1] / close.iloc[-days] - 1) * 100, 2)

    return {
        "symbol": symbol,
        "last_price": round(last, 2),
        "returns_1m": pct(21), "returns_3m": pct(63), "returns_12m": pct(252),
        "sma20": round(float(SMAIndicator(close, 20).sma_indicator().iloc[-1]), 2),
        "sma50": round(float(SMAIndicator(close, 50).sma_indicator().iloc[-1]), 2),
        "sma200": round(float(SMAIndicator(close, 200).sma_indicator().iloc[-1]), 2),
        "adx": round(float(adx), 1),
        "atr": round(float(atr.iloc[-1]), 2),
        "atr_avg20_ratio": round(float(atr.iloc[-1] / atr.iloc[-20:].mean()), 2),
        "bb_position": round(float((last - bb.bollinger_lband().iloc[-1])
                                   / (bb.bollinger_hband().iloc[-1]
                                      - bb.bollinger_lband().iloc[-1])), 2),
        "hurst": hurst_exponent(close.iloc[-252:]),
        "fractal_dim_60d": fractal_dimension(close.iloc[-60:]),
        "fractal_dim_prev60d": fractal_dimension(close.iloc[-120:-60]),
        "volume_ratio_5d_vs_60d": round(float(df["Volume"].iloc[-5:].mean()
                                              / df["Volume"].iloc[-60:].mean()), 2),
        "pct_from_52w_high": round(float(last / close.iloc[-252:].max() - 1) * 100, 2),
        "up_days_last_10": int((ret.iloc[-10:] > 0).sum()),
        "kurtosis": round(float(ret.iloc[-252:].kurtosis()), 2),
        "vol_21d_annualized": round(float(ret.iloc[-21:].std() * np.sqrt(252)) * 100, 1),
        "realized_vol_ratio": round(float(ret.iloc[-21:].std() / ret.iloc[-252:].std()), 2),
    }
