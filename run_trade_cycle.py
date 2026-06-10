"""Entry point — one trade cycle. Schedule daily at 09:35 ET."""
import json
import os
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

import broker
from data import fetch_market_data
from pipeline import run_cycle

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
LOG = os.path.join(HERE, "memory", "trade_log.jsonl")


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    record = {"time": datetime.now(timezone.utc).isoformat()}
    try:
        symbol = os.getenv("SYMBOL", "SPY")
        result = run_cycle(fetch_market_data(symbol), broker.get_equity())
        record.update(result)
        if result["status"] == "trade":
            record["order"] = broker.place_order(result["order_intent"])
    except Exception as e:
        record["status"] = "error"
        record["error"] = repr(e)
        record["trace"] = traceback.format_exc()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print(f"[{record['time']}] status={record['status']}"
          + (f" order={record.get('order')}" if record.get("order") else "")
          + (f" error={record.get('error')}" if record.get("error") else ""))


if __name__ == "__main__":
    main()
