# AI Trader — Multi-Agent Paper Trading System

LLM multi-agent trading pipeline (Gemini) with deterministic Python orchestration, paper trading on Alpaca.

## Pipeline (one cycle)
1. `data.py` — fetch SPY (yfinance), pre-compute ALL metrics in Python (ADX, SMAs, Hurst, ATR, kurtosis). LLMs never do math.
2. Regime agent classifies the market (trending / ranging / high-vol / indeterminate).
3. Strategy pool runs **sequentially with early exit**: momentum → mean reversion → volatility. Each outputs direction + confidence + regime suitability with a mandatory 4-step reasoning trace.
4. Ensemble + position sizing + risk checks — pure Python (2% equity cap × participation, 5σ survival check).
5. **Audit+Judge agent** (temperature 0) reviews the full decision package for contradictions, bias, and unsupported claims → `PASS+UPHOLD` or `FAIL+REJECT`.
6. Order placed on Alpaca **paper** account only on PASS. Everything logged to `memory/trade_log.jsonl`.

## Setup
```
pip install -r requirements.txt
copy .env.example .env   # fill in your keys
python run_trade_cycle.py
```

`.env` keys: `GEMINI_API_KEY` (aistudio.google.com), `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (paper keys from alpaca.markets), `SYMBOL` (default SPY).

Schedule daily (Windows): see `Register-ScheduledTask` snippet in the design doc.

## Status
Paper trading only. Validation gate before any live money: 100+ trades, win rate > 55%, profit factor > 1.5, max drawdown < 15%.
