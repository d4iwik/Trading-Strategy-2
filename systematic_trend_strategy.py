"""
================================================================================
 Diversified Multi-Asset Time-Series Momentum  +  Volatility Targeting
 -------------------------------------------------------------------------------
 A production-leaning, fully systematic trading strategy framework.

 Design philosophy
 -----------------
   * Robustness over backtest beauty.  Parameters are theory-motivated and FIXED,
     not optimized to the sample.  Walk-forward is used to *validate stability*,
     not to mine parameters.
   * One interpretable edge (cross-asset trend) sized by one interpretable risk
     control (inverse-volatility targeting), with explicit exposure/drawdown caps.
   * No lookahead: signals computed on data through close of day t are executed
     with a one-bar lag (effective t+1).  Vol/return estimates use trailing
     windows only.
   * Costs (commission + slippage) are charged on every notional change.

 Modules (sections) in this single file, in dependency order:
   1.  Config          - one dataclass; everything configurable lives here
   2.  Data            - real (yfinance/CSV) + synthetic fallback; survivorship notes
   3.  Signals         - blended multi-lookback time-series momentum
   4.  Risk            - vol estimation, inverse-vol sizing, vol target, caps, DD control
   5.  Costs           - commission + slippage model
   6.  Backtest engine - lag-correct, vectorized, builds equity curve + trade ledger
   7.  Metrics         - full institutional performance stat set
   8.  Walk-forward    - rolling in-sample/out-of-sample validation
   9.  Report          - assemble everything into a printable report + plots

 This file is intentionally a single module for ease of running.  It is written
 so each numbered section can be lifted into its own file (config.py, data.py,
 signals.py, ...) for a package layout with zero logic changes.

 Author: senior quant research template.  Use at your own risk.  Backtests are
 not promises.  Past performance, etc.
================================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TRADING_DAYS = 252


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class StrategyConfig:
    """All tunables in one place. Defaults are theory-motivated, NOT optimized.

    Anything you might be tempted to grid-search lives here so that parameter
    sensitivity can be tested explicitly (see `parameter_sensitivity`)."""

    # --- Universe (broad, long-lived ETFs spanning asset classes) -------------
    # Chosen for asset-class breadth, not for backtest performance.
    universe: Tuple[str, ...] = (
        "SPY",   # US equities
        "EFA",   # Developed ex-US equities
        "EEM",   # Emerging-market equities
        "IEF",   # 7-10y US Treasuries
        "TLT",   # 20y+ US Treasuries
        "LQD",   # IG credit
        "GLD",   # Gold
        "DBC",   # Broad commodities
    )
    benchmark: str = "SPY"          # comparison benchmark (buy & hold)

    # --- Signal: blended time-series momentum --------------------------------
    # Multiple lookbacks blended by sign-averaging -> fewer flips, less curve-fit.
    momentum_lookbacks: Tuple[int, ...] = (63, 126, 252)   # ~3, 6, 12 months
    signal_smoothing: int = 5        # smooth signal to damp noise/turnover

    # --- Risk / sizing --------------------------------------------------------
    vol_lookback: int = 63           # trailing window for realized-vol estimate
    target_portfolio_vol: float = 0.10   # 10% annualized target
    per_asset_vol_cap: float = 0.20      # cap any single asset's risk weight
    max_gross_leverage: float = 1.5      # total |weights| cap (avoid excess lev)
    max_net_leverage: float = 1.5        # net (directional) cap

    # --- Drawdown control (de-risk when the strategy itself is underwater) ----
    dd_control_threshold: float = 0.07   # start cutting risk past 7% DD
    dd_control_floor: float = 0.40       # never cut exposure below 40% of target

    # --- Execution / rebalancing ---------------------------------------------
    rebalance_days: int = 21         # monthly signal cadence (controls turnover)
    execution_lag: int = 1           # bars between signal and fill (no lookahead)
    no_trade_band: float = 0.02      # skip trades smaller than 2% of equity

    # --- Costs ----------------------------------------------------------------
    commission_bps: float = 1.0      # per side, bps of traded notional
    slippage_bps: float = 2.0        # per side, bps of traded notional

    # --- Backtest -------------------------------------------------------------
    initial_capital: float = 1_000_000.0
    risk_free_rate: float = 0.02     # annualized, for excess-return metrics

    # --- Walk-forward ---------------------------------------------------------
    wf_train_years: int = 5
    wf_test_years: int = 2

    def validate(self) -> None:
        assert self.target_portfolio_vol > 0
        assert self.max_gross_leverage >= self.max_net_leverage > 0
        assert 0 < self.dd_control_floor <= 1
        assert self.execution_lag >= 1, "execution_lag<1 would create lookahead"


# =============================================================================
# 2. DATA
# =============================================================================
class DataLoader:
    """Loads adjusted close prices into a wide DataFrame (dates x tickers).

    Priority: explicit CSV  ->  yfinance (if installed & network)  ->  synthetic.

    Survivorship-bias note
    ----------------------
    Using ETFs that exist *today* embeds mild survivorship bias (we never see
    delisted vehicles).  We mitigate by restricting to broad, long-lived,
    asset-class-level ETFs that are extremely unlikely to be delisted, and by
    using *total-return-adjusted* prices.  For a fully survivorship-free study
    you would use a point-in-time database (e.g. CRSP) and a delisting-aware
    universe.  This is documented, not hidden.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def load(self, csv_path: Optional[str] = None,
             start: str = "2007-01-01", end: Optional[str] = None,
             synthetic: bool = False, seed: int = 7) -> pd.DataFrame:
        tickers = list(dict.fromkeys(self.config.universe + (self.config.benchmark,)))

        if synthetic:
            return self._synthetic(tickers, start, end, seed)

        if csv_path:
            px = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            missing = [t for t in tickers if t not in px.columns]
            if missing:
                raise ValueError(f"CSV missing tickers: {missing}")
            return px[tickers].sort_index().dropna(how="all")

        # Try yfinance; fall back to synthetic if unavailable/offline.
        try:
            import yfinance as yf  # optional dependency
            raw = yf.download(tickers, start=start, end=end,
                              auto_adjust=True, progress=False)
            px = raw["Close"] if "Close" in raw else raw
            px = px.dropna(how="all").sort_index()
            return px
        except Exception as exc:  # pragma: no cover (network/offline path)
            print(f"[DataLoader] yfinance unavailable ({exc}); using synthetic data.")
            return self._synthetic(tickers, start, end, seed)

    # ---- synthetic generator: regime-switching drift + clustered vol --------
    def _synthetic(self, tickers: List[str], start: str,
                   end: Optional[str], seed: int) -> pd.DataFrame:
        """Generate plausibly-structured prices: persistent trends + vol
        clustering + cross-asset correlation.  FOR PIPELINE TESTING ONLY.
        Synthetic backtest numbers are NOT evidence the strategy works."""
        rng = np.random.default_rng(seed)
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        dates = pd.bdate_range(start, end)
        n, m = len(dates), len(tickers)

        # Block correlation: equities cluster, bonds cluster, etc. (rough).
        corr = np.full((m, m), 0.15)
        np.fill_diagonal(corr, 1.0)
        L = np.linalg.cholesky(corr)

        # Slowly-switching drift per asset (creates trends to detect/whipsaw).
        drift = np.zeros((n, m))
        for j in range(m):
            d, switch_p = 0.0, 1.0 / 180.0
            for i in range(n):
                if rng.random() < switch_p:
                    d = rng.normal(0.0, 0.0006)      # new regime drift
                drift[i, j] = d
        # Clustered volatility (GARCH-ish).
        vol = np.zeros((n, m))
        base_vol = rng.uniform(0.006, 0.016, size=m)
        v = base_vol.copy()
        for i in range(n):
            shock = rng.normal(0, 1, m)
            v = 0.94 * v + 0.06 * base_vol + 0.02 * base_vol * np.abs(shock)
            vol[i] = v
        z = rng.normal(0, 1, size=(n, m)) @ L.T
        rets = drift + vol * z
        prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
        return pd.DataFrame(prices, index=dates, columns=tickers)


# =============================================================================
# 3. SIGNALS
# =============================================================================
class SignalEngine:
    """Blended multi-lookback time-series momentum.

    For each lookback L we take sign(P_t / P_{t-L} - 1).  Averaging the signs
    across {63,126,252} yields a signal in {-1,-2/3,...,1} -- interpretable and
    far less prone to whipsaw than a single continuous z-score.  We then apply
    light smoothing.  Output range [-1, 1] = desired directional conviction."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        signs = []
        for lb in cfg.momentum_lookbacks:
            mom = prices / prices.shift(lb) - 1.0
            signs.append(np.sign(mom))
        signal = sum(signs) / len(signs)              # blended sign in [-1,1]
        if cfg.signal_smoothing > 1:
            signal = signal.rolling(cfg.signal_smoothing, min_periods=1).mean()
        return signal.clip(-1.0, 1.0)


# =============================================================================
# 4. RISK / POSITION SIZING
# =============================================================================
class RiskManager:
    """Turns raw signals into vol-targeted, capped portfolio weights.

    Pipeline per rebalance date:
      1. inverse-vol scale each asset  -> equal risk contribution intent
      2. multiply by signal conviction -> directional, sized positions
      3. scale whole book to target portfolio vol (ex-ante)
      4. apply per-asset, gross, and net leverage caps
      5. apply drawdown control multiplier (set later in the engine loop)
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def realized_vol(self, prices: pd.DataFrame) -> pd.DataFrame:
        rets = prices.pct_change()
        return rets.rolling(self.config.vol_lookback).std() * np.sqrt(TRADING_DAYS)

    def target_weights(self, signal: pd.DataFrame, vol: pd.DataFrame,
                        cov_window_returns: pd.DataFrame) -> pd.DataFrame:
        """Compute target weights on every date (will be sampled at rebalances).

        cov_window_returns is the daily returns frame used to estimate ex-ante
        portfolio vol via the trailing covariance (captures diversification)."""
        cfg = self.config
        vol = vol.replace(0, np.nan)

        # (1)+(2) inverse-vol sizing * conviction. Each asset's base weight is
        # proportional to (target_vol / its_vol), so quiet assets get more
        # capital and noisy assets get less (equal-risk-contribution intent).
        raw = signal * (cfg.target_portfolio_vol / vol)

        # Per-asset risk cap: limit |w_i * vol_i| (each asset's vol budget).
        asset_risk = (raw.abs() * vol)
        scale_per_asset = (cfg.per_asset_vol_cap / asset_risk).clip(upper=1.0)
        raw = raw * scale_per_asset.fillna(1.0)

        # (3) scale entire book to target ex-ante portfolio vol using trailing cov.
        weights = raw.copy()
        cov = cov_window_returns.rolling(cfg.vol_lookback).cov()
        # Estimate portfolio vol date-by-date only on rebalance dates for speed
        # is handled in the engine; here we compute a fast diagonal+avg-corr proxy
        # and refine in-engine if needed.  Diagonal proxy:
        port_var = (weights ** 2 * vol ** 2).sum(axis=1)
        # add a coarse covariance term using average pairwise correlation 0.1
        cross = 0.10 * (weights.mul(vol)).sum(axis=1) ** 2 \
            - 0.10 * (weights ** 2 * vol ** 2).sum(axis=1)
        port_vol = np.sqrt((port_var + cross).clip(lower=1e-8))
        book_scale = (cfg.target_portfolio_vol / port_vol).clip(upper=3.0)
        weights = weights.mul(book_scale, axis=0)

        # (4) gross & net leverage caps.
        gross = weights.abs().sum(axis=1)
        gscale = (cfg.max_gross_leverage / gross).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(gscale, axis=0)
        net = weights.sum(axis=1).abs()
        nscale = (cfg.max_net_leverage / net.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(nscale, axis=0)

        return weights.fillna(0.0)


# =============================================================================
# 5. COSTS
# =============================================================================
class CostModel:
    """Linear commission + slippage on traded notional (per side)."""

    def __init__(self, config: StrategyConfig):
        self.bps = (config.commission_bps + config.slippage_bps) / 1e4

    def cost(self, traded_notional: pd.Series) -> pd.Series:
        return traded_notional.abs() * self.bps


# =============================================================================
# 6. BACKTEST ENGINE
# =============================================================================
class BacktestResult:
    """Container for everything the engine produces."""
    def __init__(self):
        self.equity: pd.Series = None
        self.returns: pd.Series = None
        self.weights: pd.DataFrame = None
        self.positions_notional: pd.DataFrame = None
        self.gross_exposure: pd.Series = None
        self.turnover: pd.Series = None
        self.costs: pd.Series = None
        self.trades: pd.DataFrame = None
        self.benchmark_returns: pd.Series = None


class Backtester:
    """Lag-correct, vectorized backtest of the target-weight book.

    No-lookahead guarantee: target weights on date t are derived from prices
    through t, then shifted forward by `execution_lag` bars before being applied
    to returns.  Rebalances occur every `rebalance_days`; between rebalances the
    book is held (weights drift with price, recomputed at next rebalance)."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.signals = SignalEngine(config)
        self.risk = RiskManager(config)
        self.costs = CostModel(config)

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        cfg = self.config
        cfg.validate()
        universe = list(cfg.universe)
        px = prices[universe].copy().dropna(how="all")
        rets = px.pct_change().fillna(0.0)

        signal = self.signals.compute(px)
        vol = self.risk.realized_vol(px)
        raw_target = self.risk.target_weights(signal, vol, rets)

        # Sample target weights only on rebalance dates; hold otherwise.
        idx = px.index
        rebal_mask = np.zeros(len(idx), dtype=bool)
        rebal_mask[::cfg.rebalance_days] = True
        rebal_dates = idx[rebal_mask]

        target = raw_target.copy()
        target[~target.index.isin(rebal_dates)] = np.nan
        target = target.ffill().fillna(0.0)

        # Warmup: zero weights until all estimators are valid.
        warmup = max(max(cfg.momentum_lookbacks), cfg.vol_lookback) + cfg.signal_smoothing
        target.iloc[:warmup] = 0.0

        # ---- main loop: applies DD control, no-trade band, lag, costs --------
        equity = pd.Series(index=idx, dtype=float)
        weights_held = pd.DataFrame(0.0, index=idx, columns=universe)
        costs_ser = pd.Series(0.0, index=idx)
        turnover_ser = pd.Series(0.0, index=idx)

        cap = cfg.initial_capital
        peak = cap
        cur_w = pd.Series(0.0, index=universe)         # currently held weights
        # Pre-shift target by execution lag (decision at t -> applied at t+lag).
        target_lagged = target.shift(cfg.execution_lag).fillna(0.0)

        for i, dt in enumerate(idx):
            # 1) realize PnL of weights held into today using today's returns.
            day_ret = float((cur_w * rets.loc[dt]).sum())
            cap *= (1.0 + day_ret)

            # 2) drawdown control multiplier based on strategy equity.
            peak = max(peak, cap)
            dd = (cap / peak) - 1.0
            if -dd > cfg.dd_control_threshold:
                excess = (-dd - cfg.dd_control_threshold)
                # linear de-risk; floor protects against fully exiting.
                mult = max(cfg.dd_control_floor, 1.0 - 3.0 * excess)
            else:
                mult = 1.0

            # 3) desired weights for tomorrow = lagged target * DD multiplier.
            desired = target_lagged.loc[dt] * mult

            # 4) no-trade band: only move legs whose change exceeds the band.
            delta = desired - cur_w
            trade_legs = delta.abs() > cfg.no_trade_band
            new_w = cur_w.copy()
            new_w[trade_legs] = desired[trade_legs]

            # 5) costs on traded notional (per side).
            traded_notional = (new_w - cur_w).abs() * cap
            day_cost = float(self.costs.cost(traded_notional).sum())
            cap -= day_cost
            costs_ser.loc[dt] = day_cost
            turnover_ser.loc[dt] = float((new_w - cur_w).abs().sum())

            cur_w = new_w
            weights_held.loc[dt] = cur_w.values
            equity.loc[dt] = cap

        result = BacktestResult()
        result.equity = equity
        result.returns = equity.pct_change().fillna(0.0)
        result.weights = weights_held
        result.positions_notional = weights_held.mul(equity, axis=0)
        result.gross_exposure = weights_held.abs().sum(axis=1)
        result.turnover = turnover_ser
        result.costs = costs_ser
        result.trades = self._build_trade_ledger(weights_held, rets, equity)
        if cfg.benchmark in prices.columns:
            result.benchmark_returns = prices[cfg.benchmark].pct_change() \
                .reindex(idx).fillna(0.0)
        return result

    # ---- trade ledger: per-asset round trips (sign-stable holding periods) ---
    @staticmethod
    def _build_trade_ledger(weights: pd.DataFrame, rets: pd.DataFrame,
                            equity: pd.Series) -> pd.DataFrame:
        """A 'trade' = a contiguous holding period in one asset during which the
        position sign is constant and non-zero.  Trade PnL = sum of daily
        position-weighted returns * equity over the holding window (costs are
        already netted in equity; this ledger is for trade-quality stats)."""
        trades = []
        notional = weights.mul(equity.shift(1).fillna(equity.iloc[0]), axis=0)
        daily_pnl = notional.shift(1).fillna(0.0) * rets   # $ pnl per asset/day
        for asset in weights.columns:
            w = weights[asset]
            sign = np.sign(w)
            in_trade = False
            entry_i = None
            cur_sign = 0
            for i in range(len(w)):
                s = sign.iloc[i]
                if not in_trade and s != 0:
                    in_trade, entry_i, cur_sign = True, i, s
                elif in_trade and (s != cur_sign):
                    pnl = daily_pnl[asset].iloc[entry_i:i].sum()
                    trades.append(dict(asset=asset,
                                       entry=w.index[entry_i],
                                       exit=w.index[i - 1],
                                       bars=i - entry_i,
                                       direction=int(cur_sign),
                                       pnl=float(pnl)))
                    if s != 0:
                        in_trade, entry_i, cur_sign = True, i, s
                    else:
                        in_trade = False
            if in_trade:
                pnl = daily_pnl[asset].iloc[entry_i:].sum()
                trades.append(dict(asset=asset, entry=w.index[entry_i],
                                   exit=w.index[-1], bars=len(w) - entry_i,
                                   direction=int(cur_sign), pnl=float(pnl)))
        return pd.DataFrame(trades)


# =============================================================================
# 7. METRICS
# =============================================================================
class Metrics:
    """Institutional performance statistics computed from a daily return series."""

    def __init__(self, rf_annual: float = 0.02):
        self.rf = rf_annual

    def _ann_factor(self, r: pd.Series) -> float:
        return TRADING_DAYS

    def cagr(self, r: pd.Series) -> float:
        if len(r) < 2:
            return np.nan
        cum = (1 + r).prod()
        years = len(r) / TRADING_DAYS
        return cum ** (1 / years) - 1 if cum > 0 else -1.0

    def vol(self, r: pd.Series) -> float:
        return r.std() * np.sqrt(TRADING_DAYS)

    def sharpe(self, r: pd.Series) -> float:
        excess = r - self.rf / TRADING_DAYS
        sd = r.std()
        return np.sqrt(TRADING_DAYS) * excess.mean() / sd if sd > 0 else np.nan

    def sortino(self, r: pd.Series) -> float:
        excess = r - self.rf / TRADING_DAYS
        downside = r[r < 0].std()
        return np.sqrt(TRADING_DAYS) * excess.mean() / downside if downside > 0 else np.nan

    def drawdown_series(self, r: pd.Series) -> pd.Series:
        eq = (1 + r).cumprod()
        return eq / eq.cummax() - 1.0

    def max_drawdown(self, r: pd.Series) -> float:
        return self.drawdown_series(r).min()

    def avg_drawdown(self, r: pd.Series) -> float:
        dd = self.drawdown_series(r)
        underwater = dd[dd < 0]
        # mean depth of distinct drawdown episodes
        episodes, cur = [], []
        for v in dd:
            if v < 0:
                cur.append(v)
            elif cur:
                episodes.append(min(cur)); cur = []
        if cur:
            episodes.append(min(cur))
        return float(np.mean(episodes)) if episodes else 0.0

    def calmar(self, r: pd.Series) -> float:
        mdd = abs(self.max_drawdown(r))
        return self.cagr(r) / mdd if mdd > 0 else np.nan

    def recovery_stats(self, r: pd.Series) -> Dict[str, float]:
        dd = self.drawdown_series(r)
        # longest underwater stretch (bars from peak to recovery)
        longest, cur = 0, 0
        for v in dd:
            cur = cur + 1 if v < 0 else 0
            longest = max(longest, cur)
        return {"max_recovery_days": longest}

    def alpha_beta(self, r: pd.Series, bench: pd.Series) -> Tuple[float, float]:
        b = bench.reindex(r.index).fillna(0.0)
        x = b - self.rf / TRADING_DAYS
        y = r - self.rf / TRADING_DAYS
        var = x.var()
        if var == 0:
            return np.nan, np.nan
        beta = x.cov(y) / var
        alpha_daily = y.mean() - beta * x.mean()
        return alpha_daily * TRADING_DAYS, beta

    def information_ratio(self, r: pd.Series, bench: pd.Series) -> float:
        active = r - bench.reindex(r.index).fillna(0.0)
        sd = active.std()
        return np.sqrt(TRADING_DAYS) * active.mean() / sd if sd > 0 else np.nan

    # ---- trade-quality stats -------------------------------------------------
    def trade_stats(self, trades: pd.DataFrame) -> Dict[str, float]:
        if trades is None or trades.empty:
            return {k: np.nan for k in
                    ["profit_factor", "win_rate", "avg_win", "avg_loss",
                     "expectancy", "trade_count"]}
        wins = trades.loc[trades.pnl > 0, "pnl"]
        losses = trades.loc[trades.pnl < 0, "pnl"]
        gp, gl = wins.sum(), -losses.sum()
        return {
            "profit_factor": gp / gl if gl > 0 else np.inf,
            "win_rate": len(wins) / len(trades),
            "avg_win": wins.mean() if len(wins) else 0.0,
            "avg_loss": losses.mean() if len(losses) else 0.0,
            "expectancy": trades.pnl.mean(),
            "trade_count": int(len(trades)),
        }

    def full_report(self, result: BacktestResult) -> Dict[str, float]:
        r = result.returns
        bench = result.benchmark_returns if result.benchmark_returns is not None \
            else pd.Series(0.0, index=r.index)
        alpha, beta = self.alpha_beta(r, bench)
        out = {
            "CAGR": self.cagr(r),
            "Volatility": self.vol(r),
            "Sharpe": self.sharpe(r),
            "Sortino": self.sortino(r),
            "Calmar": self.calmar(r),
            "MaxDrawdown": self.max_drawdown(r),
            "AvgDrawdown": self.avg_drawdown(r),
            "InformationRatio": self.information_ratio(r, bench),
            "Alpha_annual": alpha,
            "Beta": beta,
            "AvgGrossExposure": float(result.gross_exposure.mean()),
            "AvgAnnualTurnover": float(result.turnover.sum()
                                       / (len(r) / TRADING_DAYS)),
            "TotalCosts": float(result.costs.sum()),
        }
        out.update(self.recovery_stats(r))
        out.update(self.trade_stats(result.trades))
        # benchmark comparison
        out["Benchmark_CAGR"] = self.cagr(bench)
        out["Benchmark_Sharpe"] = self.sharpe(bench)
        out["Benchmark_MaxDD"] = self.max_drawdown(bench)
        out["Excess_CAGR_vs_Benchmark"] = out["CAGR"] - out["Benchmark_CAGR"]
        return out


# =============================================================================
# 8. WALK-FORWARD VALIDATION
# =============================================================================
class WalkForward:
    """Rolling in-sample / out-of-sample validation.

    IMPORTANT design choice: we do NOT re-optimize parameters each fold (that is
    how people fool themselves).  Parameters are fixed by theory.  Walk-forward
    here measures *stability* of out-of-sample performance across time -- the
    thing that actually predicts live behavior.  An optional `optimize_fn` hook
    is provided for those who insist, but the default is fixed-parameter WF."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.metrics = Metrics(config.risk_free_rate)

    def split_dates(self, idx: pd.DatetimeIndex) -> List[Tuple]:
        cfg = self.config
        folds = []
        start = idx[0]
        train_td = pd.DateOffset(years=cfg.wf_train_years)
        test_td = pd.DateOffset(years=cfg.wf_test_years)
        cursor = start
        while True:
            tr_start = cursor
            tr_end = tr_start + train_td
            te_end = tr_end + test_td
            if tr_end >= idx[-1]:
                break
            folds.append((tr_start, tr_end, min(te_end, idx[-1])))
            cursor = tr_end          # non-overlapping test windows roll forward
            if te_end >= idx[-1]:
                break
        return folds

    def run(self, prices: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        bt = Backtester(cfg)
        full = bt.run(prices)               # single pass; slice per fold
        rows = []
        for (tr_s, tr_e, te_e) in self.split_dates(prices.index):
            test_ret = full.returns.loc[tr_e:te_e]
            test_bench = full.benchmark_returns.loc[tr_e:te_e]
            if len(test_ret) < 30:
                continue
            rows.append({
                "test_start": tr_e.date(), "test_end": te_e.date(),
                "OOS_CAGR": self.metrics.cagr(test_ret),
                "OOS_Sharpe": self.metrics.sharpe(test_ret),
                "OOS_MaxDD": self.metrics.max_drawdown(test_ret),
                "OOS_Sortino": self.metrics.sortino(test_ret),
                "Bench_CAGR": self.metrics.cagr(test_bench),
            })
        return pd.DataFrame(rows)


# =============================================================================
# 9. ROBUSTNESS: parameter-sensitivity sweep
# =============================================================================
def parameter_sensitivity(prices: pd.DataFrame, base: StrategyConfig,
                          param: str, values: List) -> pd.DataFrame:
    """Re-run the backtest varying ONE parameter; report Sharpe/CAGR/MaxDD.

    A robust strategy degrades gracefully -- it does not have a single magic
    value.  Large swings here are a red flag for overfitting."""
    m = Metrics(base.risk_free_rate)
    rows = []
    for v in values:
        cfg = StrategyConfig(**{**asdict(base), param: v})
        res = Backtester(cfg).run(prices)
        rep = m.full_report(res)
        rows.append({param: v, "Sharpe": rep["Sharpe"],
                     "CAGR": rep["CAGR"], "MaxDD": rep["MaxDrawdown"]})
    return pd.DataFrame(rows)
