"""Memory layer — the system's accumulated knowledge about itself.
At this scale (hundreds of records) plain JSON aggregation beats a vector DB:
deterministic, inspectable, zero extra dependencies. FAISS is for v3 if the
log ever grows past ~10k records.

What it learns over time:
- strategy_stats: win rate per strategy per regime (the agents' prior)
- regime_history: recent regime states + transition counts
- rejection_patterns: what the Audit+Judge keeps flagging (its own memory)
- last_narrative: yesterday's social story, so the Sociologist reports the
  DELTA, not a re-summary
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "memory", "trade_log.jsonl")
OUTCOMES = os.path.join(HERE, "memory", "outcomes.json")
HORIZON_DAYS = 5  # a trade is judged win/loss against price N cycles later


def _load_log() -> list:
    if not os.path.exists(LOG):
        return []
    with open(LOG, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_outcomes() -> dict:
    if os.path.exists(OUTCOMES):
        with open(OUTCOMES, encoding="utf-8") as f:
            return json.load(f)
    return {}


def update_outcomes(current_price: float) -> None:
    """Mark past trades as win/loss once HORIZON_DAYS newer cycles exist.
    Entry price = market last_price at decision time (approximation until
    fill-price tracking is added)."""
    records = _load_log()
    outcomes = _load_outcomes()
    trades = [(i, r) for i, r in enumerate(records) if r.get("status") == "trade"]
    for i, r in trades:
        key = r["time"]
        if key in outcomes:
            continue
        newer_cycles = len(records) - 1 - i
        if newer_cycles < HORIZON_DAYS:
            continue
        entry = r["market"]["last_price"]
        side = r["order_intent"]["side"]
        ret = (current_price / entry - 1) * (1 if side == "buy" else -1)
        outcomes[key] = {"regime": r["regime"]["regime"], "side": side,
                         "entry": entry, "exit_ref": current_price,
                         "return_pct": round(ret * 100, 2),
                         "win": ret > 0,
                         "strategies_active": [s["strategy"] for s in r["signals"]
                                               if s["direction"] != "no_trade"]}
    with open(OUTCOMES, "w", encoding="utf-8") as f:
        json.dump(outcomes, f, indent=1)


def build_context() -> dict:
    """Knowledge package injected into every agent's input each cycle."""
    records = _load_log()
    outcomes = _load_outcomes()

    # strategy win rates per regime, from judged outcomes
    stats: dict = {}
    for o in outcomes.values():
        for strat in o["strategies_active"]:
            cell = stats.setdefault(strat, {}).setdefault(
                o["regime"], {"trades": 0, "wins": 0})
            cell["trades"] += 1
            cell["wins"] += int(o["win"])
    for strat in stats.values():
        for cell in strat.values():
            cell["win_rate"] = round(cell["wins"] / cell["trades"], 2)

    # regime history (last 10) + judge rejection patterns (last 5)
    regimes = [r["regime"]["regime"] for r in records if "regime" in r]
    rejections = [{"time": r["time"][:10],
                   "bias_flags": r["verdict"].get("bias_flags", []),
                   "rationale": r["verdict"].get("verdict_rationale", "")[:300]}
                  for r in records
                  if r.get("verdict", {}).get("verdict") == "FAIL+REJECT"][-5:]
    last_social = next((r["social"] for r in reversed(records) if r.get("social")),
                       None)

    return {
        "strategy_win_rates_by_regime": stats or "no judged trades yet",
        "recent_regimes_oldest_first": regimes[-10:],
        "judged_trades": len(outcomes),
        "recent_judge_rejections": rejections,
        "previous_narrative": (last_social or {}).get("dominant_narrative"),
    }
