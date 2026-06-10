"""Orchestrator — Python controls all flow; LLMs only analyze.
v1 pipeline (paper trading): regime -> 3 strategies (sequential, early exit)
-> ensemble + sizing + risk (pure Python) -> Audit+Judge gate -> order.
Stat-arb (needs pairs data) and Social Change (needs news feed) are v2."""
import memory
import prompts
from llm import AgentError, call_agent, MODEL_CHEAP, MODEL_JUDGE

DIRECTION_NUM = {"buy": 1.0, "sell": -1.0}
MAX_POSITION_PCT = 0.02          # hard cap: 2% of equity per trade
SIGNAL_FLOOR = 0.1               # |signal| below this => no trade

STRATEGIES = [
    ("momentum", prompts.MOMENTUM),
    ("mean_reversion", prompts.MEAN_REVERSION),
    ("volatility", prompts.VOLATILITY),
]

_STRAT_KEYS = ("strategy", "direction", "confidence", "regime_suitability", "reasoning")


def compute_ensemble(signals: list) -> dict:
    """Doc formula: no_trade agents abstain (excluded from both sums)."""
    active = [s for s in signals if s["direction"] in DIRECTION_NUM]
    if not active:
        return {"final_signal": 0.0, "participation": 0.0, "n_active": 0}
    weights = [s["confidence"] * s["regime_suitability"] for s in active]
    total_w = sum(weights) or 1e-9
    signal = sum(w * DIRECTION_NUM[s["direction"]]
                 for w, s in zip(weights, active)) / total_w
    return {"final_signal": round(signal, 3),
            "participation": round(total_w / len(STRATEGIES), 3),
            "n_active": len(active)}


def early_exit(signals: list) -> str | None:
    if len(signals) < 2:
        return None
    a, b = signals[0], signals[1]
    if (a["direction"] == b["direction"] == "no_trade"
            and a["confidence"] > 0.8 and b["confidence"] > 0.8):
        return "consensus_no_trade"
    if (a["direction"] == b["direction"] and a["direction"] in DIRECTION_NUM
            and a["confidence"] > 0.7 and b["confidence"] > 0.7):
        return "strong_agreement"
    return None


def risk_check(ensemble: dict, market: dict, equity: float) -> dict:
    """Deterministic risk layer (Artzner/Taleb rules in code, not LLM)."""
    pos_value = equity * MAX_POSITION_PCT * ensemble["participation"]
    # 5-sigma daily move survival check
    sigma_5_loss = pos_value * 5 * market["vol_21d_annualized"] / 100 / (252 ** 0.5)
    blocked = []
    if abs(ensemble["final_signal"]) <= SIGNAL_FLOOR:
        blocked.append(f"signal {ensemble['final_signal']} below floor {SIGNAL_FLOOR}")
    if pos_value < 10:  # Alpaca notional minimum is $1; $10 floor for sanity
        blocked.append(f"position value ${pos_value:.2f} too small")
    if market["kurtosis"] > 6 and ensemble["participation"] < 0.3:
        blocked.append("fat tails + weak participation")
    return {"notional": round(pos_value, 2),
            "sigma5_dollar_loss": round(sigma_5_loss, 2), "blocked": blocked}


def run_cycle(market: dict, equity: float = 100_000.0, news: list = None) -> dict:
    result = {"market": market, "status": "no_trade", "order_intent": None}

    # Memory: judge past trades against today's price, then load what the
    # system has learned so far (win rates per regime, rejection patterns)
    memory.update_outcomes(market["last_price"])
    history = memory.build_context()
    result["memory_context"] = history

    # Phase 0.0 — light: regime classification + social narrative (cheap)
    regime = call_agent(prompts.REGIME,
                        {"market": market,
                         "recent_regimes": history["recent_regimes_oldest_first"]},
                        MODEL_CHEAP, 0.2,
                        required_keys=("regime", "confidence", "reasoning"))
    social = call_agent(prompts.SOCIOLOGIST,
                        {"headlines": news or [],
                         "previous_narrative": history["previous_narrative"]},
                        MODEL_CHEAP, 0.2,
                        required_keys=("social_regime", "narrative_direction",
                                       "confidence", "reasoning"))
    result["regime"], result["social"] = regime, social

    # Phase 0.1 — heavy confirmation, only if the light regime is uncertain
    if regime["confidence"] < 0.7 or regime["regime"] == "indeterminate":
        physicist = call_agent(prompts.PHYSICIST, market, MODEL_JUDGE, 0.2,
                               required_keys=("regime_verdict",
                                              "suggested_regime", "confidence",
                                              "reasoning"))
        result["physicist"] = physicist
        if physicist["regime_verdict"] == "veto":
            regime = {"regime": physicist["suggested_regime"],
                      "confidence": min(physicist["confidence"], 0.7),
                      "reasoning": "physicist veto: " + physicist["reasoning"]}
        elif physicist["regime_verdict"] == "confirm":
            regime["confidence"] = min(regime["confidence"] + 0.15, 0.9)

        game = call_agent(prompts.GAME_THEORIST, market, MODEL_JUDGE, 0.2,
                          required_keys=("crowd_state", "contrarian_risk",
                                         "confidence", "reasoning"))
        result["game_theorist"] = game
        result["regime"] = regime

    # Phase 1 — strategy pool, sequential with early exit
    signals, exit_reason = [], None
    strat_input = {"market": market, "regime": regime, "social": social,
                   "crowd": result.get("game_theorist"),
                   "your_track_record": history["strategy_win_rates_by_regime"]}
    for name, prompt in STRATEGIES:
        out = call_agent(prompt, strat_input, MODEL_CHEAP, 0.2,
                         required_keys=_STRAT_KEYS)
        out["strategy"] = name
        signals.append(out)
        exit_reason = early_exit(signals)
        if exit_reason:
            break
    result["signals"] = signals
    result["early_exit"] = exit_reason

    # Phase 2 — ensemble + sizing + risk (pure Python)
    ensemble = compute_ensemble(signals)
    risk = risk_check(ensemble, market, equity)
    result["ensemble"], result["risk"] = ensemble, risk
    if risk["blocked"]:
        result["status"] = "no_trade"
        return result

    side = "buy" if ensemble["final_signal"] > 0 else "sell"
    intent = {"side": side, "notional": risk["notional"], "symbol": market["symbol"]}

    # Phase 3 — Audit+Judge gate (1 call, temp 0)
    package = {"market": market, "regime": regime, "social": social,
               "physicist": result.get("physicist"),
               "game_theorist": result.get("game_theorist"),
               "signals": signals, "ensemble": ensemble, "risk": risk,
               "order_intent": intent,
               "your_past_rejections": history["recent_judge_rejections"],
               "strategy_track_record": history["strategy_win_rates_by_regime"]}
    verdict = call_agent(prompts.AUDIT_JUDGE, package, MODEL_JUDGE, 0.0,
                         required_keys=("verdict", "verdict_rationale"))
    result["verdict"] = verdict
    if verdict["verdict"] == "PASS+UPHOLD":
        result["status"] = "trade"
        result["order_intent"] = intent
    else:
        result["status"] = "rejected"
    return result
