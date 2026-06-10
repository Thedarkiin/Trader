"""Entry point — one cycle per symbol. Schedule daily at 09:35 ET.
Order of operations (trader's discipline): manage existing positions FIRST
(stops/targets/time exits), only then look for new trades."""
import json
import os
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

import broker
import positions
from data import fetch_market_data, fetch_news
from pipeline import run_cycle

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
LOG = os.path.join(HERE, "memory", "trade_log.jsonl")


def log_record(record: dict) -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def process_symbol(symbol: str, equity: float) -> None:
    record = {"time": datetime.now(timezone.utc).isoformat(), "symbol": symbol}
    try:
        market = fetch_market_data(symbol)

        # 1. Exits first: does an open position need closing?
        exit_reason = positions.check_exit(symbol, market["last_price"])
        if exit_reason:
            record["exit"] = {"reason": exit_reason,
                              **broker.close_position(symbol)}
            positions.close(symbol)

        # 2. New entries: skip if still holding (no pyramiding)
        if positions.is_open(symbol):
            record["status"] = "holding"
        else:
            result = run_cycle(market, equity, news=fetch_news(symbol))
            record.update(result)
            if result["status"] == "trade":
                record["order"] = broker.place_order(result["order_intent"])
                record["position"] = positions.record_entry(
                    symbol, result["order_intent"]["side"],
                    market["last_price"], market["atr"])
    except Exception as e:
        record["status"] = "error"
        record["error"] = repr(e)
        record["trace"] = traceback.format_exc()
    log_record(record)
    print(f"[{record['time']}] {symbol}: {record.get('status')}"
          + (f" EXIT: {record['exit']}" if record.get("exit") else "")
          + (f" order={record.get('order')}" if record.get("order") else "")
          + (f" error={record.get('error')}" if record.get("error") else ""))


def main():
    symbols = [s.strip() for s in
               os.getenv("SYMBOLS", os.getenv("SYMBOL", "SPY")).split(",")]
    equity = broker.get_equity()
    for symbol in symbols:
        process_symbol(symbol, equity)


if __name__ == "__main__":
    main()
