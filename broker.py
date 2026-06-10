"""Alpaca paper-trading broker. Falls back to log-only mode if keys missing."""
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET = os.getenv("ALPACA_SECRET_KEY", "")

LIVE = bool(API_KEY and SECRET)


def place_order(intent: dict) -> dict:
    if not LIVE:
        return {"status": "log_only", "note": "no Alpaca keys in .env — "
                "order logged, not sent", **intent}
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    client = TradingClient(API_KEY, SECRET, paper=True)
    req = MarketOrderRequest(
        symbol=intent["symbol"], notional=intent["notional"],
        side=OrderSide.BUY if intent["side"] == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY)
    order = client.submit_order(req)
    return {"status": "submitted", "order_id": str(order.id), **intent}


def close_position(symbol: str) -> dict:
    if not LIVE:
        return {"status": "log_only", "action": "close", "symbol": symbol}
    from alpaca.trading.client import TradingClient
    client = TradingClient(API_KEY, SECRET, paper=True)
    order = client.close_position(symbol)
    return {"status": "closed", "symbol": symbol, "order_id": str(order.id)}


def get_equity() -> float:
    if not LIVE:
        return 100_000.0
    from alpaca.trading.client import TradingClient
    return float(TradingClient(API_KEY, SECRET, paper=True).get_account().equity)
