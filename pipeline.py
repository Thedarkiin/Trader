"""Orchestrator — Python controls all flow; LLMs only analyze.
v1 pipeline (paper trading): regime -> 3 strategies (sequential, early exit)
-> ensemble + sizing + risk (pure Python) -> Audit+Judge gate -> order.
Stat-arb (needs pairs data) and Social Change (needs news feed) are v2."""
import memory
import prompts
from llm import AgentError, call_agent, MODEL_CHEAP, MODEL_JUDGE

DIRECTION_NUM = {"buy": 1.0, "sell": -1.0}
MAX_POSITION_PCT = 0.04          # hard cap: 4% of equity notional per trade
SIGNAL_FLOOR = 0.1               # |signal| below this => no trade

STRATEGIES = [
    ("momentum", prompts.MOMENTUM),
    ("mean_reversion", prompts.MEAN_REVERSION),
    ("volatility", prompts.VOLATILITY),
]

_STRAT_KEYS = ("strategy", "direction", "confidence", "regime_suitability", "reasoning")

# The judge attests each checkpoint; Python derives the verdict (all must pass).
JUDGE_CHECKPOINTS = (
    "contradictions_addressed", "numbers_verified", "no_banned_language",
    "confidence_supported", "counterarguments_real", "not_repeat_offense",
    "track_record_respected",
)


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


def regime_hard_check(market: dict, regime: dict) -> str | None:
    """ADX/Hurst contradiction rule from the regime prompt, enforced in code:
    adx > 25 (trending) while hurst < 0.45 (mean-reverting) is structurally
    contradictory — the regime must be indeterminate, never a coin-flip pick."""
    if market["adx"] > 25 and market["hurst"] < 0.45 \
            and regime["regime"] != "indeterminate":
        note = (f"forced indeterminate: adx {market['adx']} > 25 contradicts "
                f"hurst {market['hurst']} < 0.45 (was {regime['regime']})")
        regime["regime"] = "indeterminate"
        regime["confidence"] = min(regime["confidence"], 0.6)
        return note
    return None


def regime_trend_promote(market: dict, regime: dict) -> str | None:
    """Deterministic counterpart to regime_hard_check: when price structure is
    an objectively clean trend (ADX >= 20 with a monotonic SMA stack) but the
    LLM hedged to ranging/indeterminate, promote the label in code. Python owns
    the flow; the regime agent at temp 0.2 proved unwilling to call a moderate
    trend on its own. high_volatility is left alone (it overrides trend labels),
    and the ADX/hurst contradiction case is left to regime_hard_check."""
    if regime["regime"] == "high_volatility":
        return None
    if market["adx"] > 25 and market["hurst"] < 0.45:
        return None                                  # genuine ADX/hurst conflict
    if market["adx"] < 20:
        return None
    up = market["sma20"] > market["sma50"] > market["sma200"]
    down = market["sma20"] < market["sma50"] < market["sma200"]
    if not (up or down):
        return None                                  # tangled stack -> not clean
    target = "trending_bull" if up else "trending_bear"
    if regime["regime"] == target:
        return None
    note = (f"promoted to {target}: adx {market['adx']} >= 20 with a "
            f"{'rising' if up else 'falling'} sma stack (was {regime['regime']})")
    regime["regime"] = target
    regime["confidence"] = max(regime.get("confidence", 0.65), 0.7)
    return note


def hard_gate(name: str, market: dict, regime: dict) -> str | None:
    """Deterministic strategy pre-filters — the pure arithmetic entry rules
    from the prompts, run in code BEFORE the LLM call. A gated strategy costs
    no tokens and cannot fumble its own threshold. Returns block reason."""
    r = regime["regime"]
    if name == "momentum" and r in ("ranging", "indeterminate"):
        return f"regime '{r}': momentum requires a trending regime"
    if name == "mean_reversion":
        bb = market["bb_position"]
        if 0.05 < bb < 0.95:
            return (f"bb_position {bb} inside bands "
                    f"(entry requires < 0.05 or > 0.95)")
        if r != "ranging":
            return f"regime '{r}': reversion entry requires ranging"
    if name == "volatility" and market["atr_avg20_ratio"] < 1.1:
        return (f"atr_avg20_ratio {market['atr_avg20_ratio']} < 1.1: "
                f"no breakout case")
    return None


def enforce_coherence(signals: list, regime: dict, physicist: dict | None,
                      game: dict | None) -> list[str]:
    """Deterministic cross-agent consistency layer (Python, not LLM prose).
    No agent may be more confident in a trade than the system is in its own
    read of the market. Mutates signals in place; returns adjustment notes."""
    notes = []
    cap = regime["confidence"]
    if physicist and physicist.get("regime_verdict") != "confirm":
        if physicist["confidence"] < cap:
            cap = physicist["confidence"]
            notes.append(f"regime confidence capped at {cap} "
                         f"(physicist verdict: {physicist['regime_verdict']})")
        regime["confidence"] = min(regime["confidence"], cap)
    if regime.get("fat_tails_flag") or (physicist or {}).get("phase_transition_risk"):
        if cap > 0.5:
            cap = 0.5
            notes.append("confidence capped at 0.5 (fat tails / phase transition risk)")
    for s in signals:
        if s["direction"] not in DIRECTION_NUM:
            continue
        if s["regime_suitability"] <= 0.3:
            notes.append(f"{s['strategy']}: active signal voided, "
                         f"regime_suitability {s['regime_suitability']} <= 0.3")
            s["direction"] = "no_trade"
            continue
        if s["confidence"] > cap:
            notes.append(f"{s['strategy']}: confidence {s['confidence']} "
                         f"clamped to regime cap {cap}")
            s["confidence"] = round(cap, 2)
        risk = (game or {}).get("contrarian_risk") or 0.0
        if risk >= 0.6:
            new = round(s["confidence"] * (1 - risk), 2)
            notes.append(f"{s['strategy']}: confidence {s['confidence']} -> {new} "
                         f"(contrarian_risk {risk})")
            s["confidence"] = new
    return notes


def early_exit(signals: list) -> str | None:
    # hard-gated abstains are deterministic and say nothing about the
    # remaining strategies — only real LLM opinions may form a consensus
    voiced = [s for s in signals if not s.get("hard_gated")]
    if len(voiced) < 2:
        return None
    a, b = voiced[0], voiced[1]
    if (a["direction"] == b["direction"] == "no_trade"
            and a["confidence"] > 0.8 and b["confidence"] > 0.8):
        return "consensus_no_trade"
    if (a["direction"] == b["direction"] and a["direction"] in DIRECTION_NUM
            and a["confidence"] > 0.7 and b["confidence"] > 0.7):
        return "strong_agreement"
    return None


def risk_check(ensemble: dict, market: dict, equity: float) -> dict:
    """Deterministic risk layer (Artzner/Taleb rules in code, not LLM)."""
    # A valid signal has already cleared the strategy gates, the regime-
    # suitability void, and (downstream) the judge — so floor the sizing
    # multiplier to 0.5 so a real trade isn't dust. The raw participation is
    # left untouched for the fat-tails risk rule below.
    size_part = max(ensemble["participation"], 0.5)
    pos_value = equity * MAX_POSITION_PCT * size_part
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
    memory.update_outcomes(market["symbol"], market["last_price"])
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
    result["regime_hard_check"] = [
        n for n in (regime_hard_check(market, regime),
                    regime_trend_promote(market, regime)) if n]

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
        # a physicist veto/confirm rewrote the regime — re-assert the
        # ADX/Hurst contradiction rule on the final version
        recheck = regime_hard_check(market, regime)
        if recheck:
            result["regime_hard_check"].append("post-physicist " + recheck)

    # Phase 1 — strategy pool, sequential with early exit
    signals, exit_reason = [], None
    strat_input = {"market": market, "regime": regime, "social": social,
                   "crowd": result.get("game_theorist"),
                   "your_track_record": history["strategy_win_rates_by_regime"]}
    for name, prompt in STRATEGIES:
        gate = hard_gate(name, market, regime)
        if gate:
            # deterministic abstain: full confidence, no LLM call spent
            signals.append({"strategy": name, "direction": "no_trade",
                            "confidence": 1.0, "regime_suitability": 0.0,
                            "target_price": None, "stop_loss": None,
                            "hard_gated": True,
                            "reasoning": "hard gate (code): " + gate})
        else:
            out = call_agent(prompt, strat_input, MODEL_CHEAP, 0.2,
                             required_keys=_STRAT_KEYS)
            out["strategy"] = name
            signals.append(out)
        exit_reason = early_exit(signals)
        if exit_reason:
            break
    result["signals"] = signals
    result["early_exit"] = exit_reason

    # Phase 1.5 — coherence gate: enforce cross-agent confidence caps in code
    result["coherence_adjustments"] = enforce_coherence(
        signals, regime, result.get("physicist"), result.get("game_theorist"))

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
    audit = call_agent(prompts.AUDIT_JUDGE, package, MODEL_JUDGE, 0.0,
                       required_keys=("checklist", "verdict_rationale"))
    # Verdict computed in code from the checklist: the judge attests each
    # checkpoint; it cannot hand-wave a PASS. Missing/malformed item = fail.
    failed = [name for name in JUDGE_CHECKPOINTS
              if not isinstance(audit["checklist"].get(name), dict)
              or audit["checklist"][name].get("pass") is not True]
    audit["verdict"] = "FAIL+REJECT" if failed else "PASS+UPHOLD"
    audit["failed_checkpoints"] = failed
    result["verdict"] = audit
    if failed:
        result["status"] = "rejected"
    else:
        result["status"] = "trade"
        result["order_intent"] = intent
    return result
