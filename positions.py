"""Open-position tracking + exit rules.
Every entry stores its stop (2x ATR), target (3x ATR), and max holding time.
Each cycle starts by checking exits BEFORE any new decision — an experienced
trader manages existing risk before looking for new trades."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "memory", "positions.json")
MAX_HOLD_CYCLES = 10  # time stop: swing trades that go nowhere get closed


def _load() -> dict:
    if os.path.exists(FILE):
        with open(FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(d: dict) -> None:
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)


def record_entry(symbol: str, side: str, entry: float, atr: float) -> dict:
    pos = _load()
    sign = 1 if side == "buy" else -1
    pos[symbol] = {"side": side, "entry": entry,
                   "stop": round(entry - sign * 2 * atr, 2),
                   "target": round(entry + sign * 3 * atr, 2),
                   "cycles_held": 0}
    _save(pos)
    return pos[symbol]


def is_open(symbol: str) -> bool:
    return symbol in _load()


def check_exit(symbol: str, last_price: float) -> str | None:
    """Returns exit reason if the position should be closed, else None.
    Also increments holding age."""
    pos = _load()
    if symbol not in pos:
        return None
    p = pos[symbol]
    p["cycles_held"] += 1
    _save(pos)
    long = p["side"] == "buy"
    if (long and last_price <= p["stop"]) or (not long and last_price >= p["stop"]):
        return f"stop hit ({last_price} vs stop {p['stop']})"
    if (long and last_price >= p["target"]) or (not long and last_price <= p["target"]):
        return f"target hit ({last_price} vs target {p['target']})"
    if p["cycles_held"] >= MAX_HOLD_CYCLES:
        return f"time stop ({p['cycles_held']} cycles, went nowhere)"
    return None


def close(symbol: str) -> None:
    pos = _load()
    pos.pop(symbol, None)
    _save(pos)
