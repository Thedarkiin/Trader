"""Offline verification: stub the LLM, exercise gates / early-exit / coherence."""
import pipeline

CALLS = []

def fake_agent(responses):
    it = iter(responses)
    def call(prompt, payload, model, temp, required_keys=None):
        CALLS.append(prompt[:40])
        return next(it)
    return call

MARKET_RANGING = {"symbol": "TST", "last_price": 100.0, "adx": 15.0, "hurst": 0.40,
                  "atr": 2.0, "atr_avg20_ratio": 1.3, "bb_position": 0.5,
                  "kurtosis": 1.0, "vol_21d_annualized": 15.0}

# Case 1: ranging regime, bb inside bands, atr 1.3
#  -> momentum + mean_reversion hard-gated, volatility MUST still be called
CALLS.clear()
pipeline.call_agent = fake_agent([
    {"regime": "ranging", "confidence": 0.85, "fat_tails_flag": False, "reasoning": "r"},
    {"social_regime": "stable", "narrative_direction": "neutral",
     "confidence": 0.3, "reasoning": "s", "dominant_narrative": None},
    {"strategy": "volatility", "direction": "no_trade", "confidence": 0.2,
     "regime_suitability": 0.3, "reasoning": "v"},
])
r = pipeline.run_cycle(dict(MARKET_RANGING))
gated = [s["strategy"] for s in r["signals"] if s.get("hard_gated")]
assert gated == ["momentum", "mean_reversion"], gated
assert len(r["signals"]) == 3, "volatility was skipped by gated consensus!"
assert r["early_exit"] is None
assert r["status"] == "no_trade"
print("case 1 OK: gates fired, volatility still consulted, no false consensus")

# Case 2: ADX/Hurst contradiction -> forced indeterminate -> physicist escalation
CALLS.clear()
m2 = dict(MARKET_RANGING, adx=30.0, hurst=0.40, bb_position=0.5, atr_avg20_ratio=1.0)
pipeline.call_agent = fake_agent([
    {"regime": "trending_bull", "confidence": 0.9, "fat_tails_flag": False, "reasoning": "r"},
    {"social_regime": "stable", "narrative_direction": "neutral",
     "confidence": 0.3, "reasoning": "s", "dominant_narrative": None},
    # physicist VETO back to trending_bull (re-violates the contradiction)
    {"regime_verdict": "veto", "suggested_regime": "trending_bull",
     "confidence": 0.8, "reasoning": "p"},
    {"crowd_state": "no_signal", "contrarian_risk": 0.0, "confidence": 0.2,
     "reasoning": "g"},
    # momentum now allowed (trending)... but recheck must have re-forced indeterminate,
    # so momentum should actually be hard-gated. volatility gated by atr 1.0.
])
r2 = pipeline.run_cycle(m2)
assert len(r2["regime_hard_check"]) == 2, r2["regime_hard_check"]
assert r2["regime"]["regime"] == "indeterminate"
assert r2["regime"]["confidence"] <= 0.6
assert all(s.get("hard_gated") for s in r2["signals"]), r2["signals"]
print("case 2 OK: contradiction re-asserted after physicist veto, both notes kept")

# Case 3: trade path — trending regime, signals flow through coherence + ensemble
CALLS.clear()
m3 = dict(MARKET_RANGING, adx=30.0, hurst=0.60, bb_position=0.5, atr_avg20_ratio=1.6)
pipeline.call_agent = fake_agent([
    {"regime": "trending_bull", "confidence": 0.8, "fat_tails_flag": False, "reasoning": "r"},
    {"social_regime": "stable", "narrative_direction": "neutral",
     "confidence": 0.3, "reasoning": "s", "dominant_narrative": None},
    {"strategy": "momentum", "direction": "buy", "confidence": 0.95,
     "regime_suitability": 0.9, "reasoning": "m"},   # over-confident: must clamp to 0.8
    {"strategy": "volatility", "direction": "buy", "confidence": 0.7,
     "regime_suitability": 0.8, "reasoning": "v"},
    {"verdict": "PASS+UPHOLD", "verdict_rationale": "ok"},
])
r3 = pipeline.run_cycle(m3)
mom = next(s for s in r3["signals"] if s["strategy"] == "momentum")
assert mom["confidence"] == 0.8, mom["confidence"]
assert any("clamped" in n for n in r3["coherence_adjustments"])
assert r3["status"] == "trade" and r3["order_intent"]["side"] == "buy"
print("case 3 OK: clamp applied, trade flows to judge and passes")
print("ALL OK")
