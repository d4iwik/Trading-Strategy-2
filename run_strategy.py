"""
Runner / reporting harness for the systematic trend strategy.

Usage
-----
  python run_strategy.py --synthetic           # offline pipeline demo
  python run_strategy.py --csv prices.csv       # your own adjusted-close CSV
  python run_strategy.py                        # live download via yfinance

The CSV must be wide: a date index column plus one column per ticker of
*total-return-adjusted* close prices, covering every ticker in the universe
plus the benchmark.

Paper-trading hook
------------------
`latest_target_weights()` returns the weights the strategy WANTS to hold as of
the last available bar -- feed these to a broker API (e.g. Alpaca, IBKR) on a
scheduled job to go from backtest to paper/live with the same code path.
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from systematic_trend_strategy import (
    StrategyConfig, DataLoader, Backtester, Metrics, WalkForward,
    parameter_sensitivity,
)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)


# ----------------------------------------------------------------------------
def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "    n/a"
    return f"{v:8.2%}" if pct else f"{v:8.2f}"


def print_report(report: dict, cfg: StrategyConfig) -> None:
    pct_keys = {"CAGR", "Volatility", "MaxDrawdown", "AvgDrawdown",
                "Alpha_annual", "Benchmark_CAGR", "Benchmark_MaxDD",
                "Excess_CAGR_vs_Benchmark", "win_rate"}
    print("\n" + "=" * 64)
    print(" PERFORMANCE REPORT  (net of commission + slippage)")
    print("=" * 64)

    def line(label, key):
        print(f"  {label:<26}{_fmt(report.get(key), key in pct_keys)}")

    print(" Returns")
    line("CAGR", "CAGR")
    line("Volatility (ann.)", "Volatility")
    line("Alpha (ann., vs bench)", "Alpha_annual")
    line("Beta", "Beta")
    print(" Risk-adjusted")
    line("Sharpe", "Sharpe")
    line("Sortino", "Sortino")
    line("Calmar", "Calmar")
    line("Information Ratio", "InformationRatio")
    print(" Drawdown")
    line("Max Drawdown", "MaxDrawdown")
    line("Avg Drawdown", "AvgDrawdown")
    line("Max Recovery (days)", "max_recovery_days")
    print(" Trade quality")
    line("Profit Factor", "profit_factor")
    line("Win Rate", "win_rate")
    line("Avg Win ($)", "avg_win")
    line("Avg Loss ($)", "avg_loss")
    line("Expectancy ($/trade)", "expectancy")
    line("Trade Count", "trade_count")
    print(" Exposure / cost")
    line("Avg Gross Exposure", "AvgGrossExposure")
    line("Avg Annual Turnover", "AvgAnnualTurnover")
    line("Total Costs ($)", "TotalCosts")
    print(" Benchmark comparison")
    line("Benchmark CAGR", "Benchmark_CAGR")
    line("Benchmark Sharpe", "Benchmark_Sharpe")
    line("Benchmark Max DD", "Benchmark_MaxDD")
    line("Excess CAGR vs bench", "Excess_CAGR_vs_Benchmark")
    print("=" * 64)


def validation_gate(report: dict, wf: pd.DataFrame) -> None:
    """Apply the user's explicit rejection rules and print PASS/FAIL."""
    print("\n VALIDATION GATE (reject if any FAIL)")
    print(" " + "-" * 50)
    checks = [
        ("Sharpe >= 1.0", report["Sharpe"] >= 1.0),
        ("Profit Factor >= 1.2", report["profit_factor"] >= 1.2),
        ("Max Drawdown <= 20%", abs(report["MaxDrawdown"]) <= 0.20),
        ("Trade count >= 100 (significance)", report["trade_count"] >= 100),
    ]
    if wf is not None and not wf.empty:
        oos_ok = (wf["OOS_Sharpe"] > 0).mean() >= 0.6
        checks.append(("OOS Sharpe>0 in >=60% folds", oos_ok))
    for label, ok in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}]  {label}")
    print(" " + "-" * 50)


def latest_target_weights(prices: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """Paper-trading hook: weights as of the last bar (post-lag, executable)."""
    res = Backtester(cfg).run(prices)
    return res.weights.iloc[-1]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--start", type=str, default="2007-01-01")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = StrategyConfig()
    prices = DataLoader(cfg).load(csv_path=args.csv, start=args.start,
                                  synthetic=args.synthetic, seed=args.seed)
    print(f"Loaded prices: {prices.shape[0]} rows, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")

    res = Backtester(cfg).run(prices)
    report = Metrics(cfg.risk_free_rate).full_report(res)

    wf = WalkForward(cfg).run(prices)

    print_report(report, cfg)

    print("\n WALK-FORWARD (out-of-sample folds)")
    if not wf.empty:
        print(wf.to_string(index=False,
                           formatters={c: (lambda x: f"{x:6.2%}")
                                       for c in ["OOS_CAGR", "OOS_MaxDD", "Bench_CAGR"]}))
        print(f"\n   Mean OOS Sharpe: {wf['OOS_Sharpe'].mean():.2f} | "
              f"% folds Sharpe>0: {(wf['OOS_Sharpe']>0).mean():.0%} | "
              f"Worst OOS MaxDD: {wf['OOS_MaxDD'].min():.2%}")
    else:
        print("   (insufficient history for walk-forward folds)")

    validation_gate(report, wf)

    print("\n PARAMETER SENSITIVITY (Sharpe should be stable, not spiky)")
    sens = parameter_sensitivity(
        prices, cfg, "target_portfolio_vol", [0.06, 0.08, 0.10, 0.12, 0.15])
    print(sens.to_string(index=False,
                         formatters={"Sharpe": lambda x: f"{x:.2f}",
                                     "CAGR": lambda x: f"{x:.2%}",
                                     "MaxDD": lambda x: f"{x:.2%}"}))
    sens2 = parameter_sensitivity(
        prices, cfg, "vol_lookback", [42, 63, 84, 105])
    print(sens2.to_string(index=False,
                          formatters={"Sharpe": lambda x: f"{x:.2f}",
                                      "CAGR": lambda x: f"{x:.2%}",
                                      "MaxDD": lambda x: f"{x:.2%}"}))


if __name__ == "__main__":
    main()
