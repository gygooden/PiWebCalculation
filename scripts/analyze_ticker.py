from __future__ import annotations
import argparse, json
from piwebcalc.risk.toolkit import analyze_ticker

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze VaR/CVaR and backtest for a ticker.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--cl", type=float, default=0.95)
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--returns", choices=["log","simple"], default="log")
    args = p.parse_args()

    report = analyze_ticker(args.ticker, args.start, args.end, args.cl, args.window, args.returns)
    print(json.dumps(report, indent=2))