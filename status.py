"""Quick P&L check: python status.py"""
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from alpaca.trading.client import TradingClient

c = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"),
                  paper=True)
a = c.get_account()
start = 100_000.0
eq = float(a.equity)
print(f"Equity: ${eq:,.2f}   Total P&L since start: ${eq - start:+,.2f} "
      f"({(eq / start - 1) * 100:+.2f}%)")
for p in c.get_all_positions():
    print(f"  {p.symbol}: {float(p.qty):.4f} shares, "
          f"value ${float(p.market_value):,.2f}, "
          f"unrealized {float(p.unrealized_plpc) * 100:+.2f}% "
          f"(${float(p.unrealized_pl):+,.2f})")
