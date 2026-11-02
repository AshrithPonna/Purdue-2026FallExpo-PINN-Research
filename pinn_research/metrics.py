"""Financial and numerical metrics, kept separate by strategy family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .backtest import BacktestResult
from .model_types import FloatArray
from .stat_arb import PairCalibration, StatArbResult


@dataclass(frozen=True)
class DrawdownMetrics:
    maximum_drawdown: float
    maximum_drawdown_fraction: float
    maximum_drawdown_duration_steps: int


@dataclass(frozen=True)
class StatisticalArbitrageMetrics:
    net_pnl: float
    maximum_drawdown: float
    maximum_drawdown_fraction: float
    maximum_drawdown_duration_steps: int
    inventory_variance_by_asset: FloatArray
    mean_gross_exposure: float
    maximum_gross_exposure: float
    total_turnover: float
    total_transaction_costs: float
    transaction_cost_fraction_of_turnover: float
    mean_absolute_spread_zscore: float
    active_pair_fraction: float


@dataclass(frozen=True)
class MarketMakingMetrics:
    net_pnl: float
    maximum_drawdown: float
    maximum_drawdown_fraction: float
    maximum_drawdown_duration_steps: int
    inventory_variance_by_asset: FloatArray
    spread_capture_after_fees: float
    submitted_orders: int
    trades: int
    quote_opportunity_to_trade_ratio: float | None


def drawdown_metrics(equity: FloatArray) -> DrawdownMetrics:
    """Measure peak-to-trough drawdown and longest unrecovered duration."""
    values = np.asarray(equity, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("equity must be a finite vector with at least two observations")
    peaks = np.maximum.accumulate(values)
    drawdowns = peaks - values
    fractions = np.divide(
        drawdowns,
        np.abs(peaks),
        out=np.zeros_like(drawdowns),
        where=peaks != 0,
    )
    current_duration = 0
    longest_duration = 0
    for drawdown in drawdowns:
        if drawdown > 0:
            current_duration += 1
            longest_duration = max(longest_duration, current_duration)
        else:
            current_duration = 0
    return DrawdownMetrics(
        maximum_drawdown=float(np.max(drawdowns)),
        maximum_drawdown_fraction=float(np.max(fractions)),
        maximum_drawdown_duration_steps=longest_duration,
    )


def statistical_arbitrage_metrics(
    result: StatArbResult,
) -> StatisticalArbitrageMetrics:
    drawdown = drawdown_metrics(
        np.concatenate(([result.initial_capital], result.equity))
    )
    variance = np.var(result.holdings, axis=0, ddof=0)
    variance.setflags(write=False)
    turnover = float(np.sum(result.turnover))
    costs = float(np.sum(result.transaction_costs))
    return StatisticalArbitrageMetrics(
        net_pnl=result.terminal_pnl,
        maximum_drawdown=drawdown.maximum_drawdown,
        maximum_drawdown_fraction=drawdown.maximum_drawdown_fraction,
        maximum_drawdown_duration_steps=drawdown.maximum_drawdown_duration_steps,
        inventory_variance_by_asset=variance,
        mean_gross_exposure=float(np.mean(result.gross_exposure)),
        maximum_gross_exposure=float(np.max(result.gross_exposure)),
        total_turnover=turnover,
        total_transaction_costs=costs,
        transaction_cost_fraction_of_turnover=costs / turnover if turnover else 0.0,
        mean_absolute_spread_zscore=float(np.mean(np.abs(result.zscores))),
        active_pair_fraction=float(np.mean(result.pair_positions != 0.0)),
    )


def market_making_metrics(
    result: BacktestResult,
    mid_prices: FloatArray,
    fee_per_unit: FloatArray,
) -> MarketMakingMetrics:
    mids = np.asarray(mid_prices, dtype=np.float64)
    fees = np.asarray(fee_per_unit, dtype=np.float64)
    steps = len(result.timestamps) - 1
    assets = len(result.symbols)
    if mids.shape != (steps + 1, assets):
        raise ValueError("mid_prices dimensions must match backtest timestamps and assets")
    if fees.shape != (assets,) or not np.all(np.isfinite(fees)) or np.any(fees < 0):
        raise ValueError("fee_per_unit must contain finite nonnegative fees per asset")
    bid_capture = result.bid_fills * (mids[:-1] - result.bid_quotes)
    ask_capture = result.ask_fills * (result.ask_quotes - mids[:-1])
    total_fills = result.bid_fills + result.ask_fills
    net_capture = float(
        np.sum(bid_capture + ask_capture)
        - np.sum(total_fills * fees[np.newaxis, :])
    )
    orders = int(np.sum(result.bid_orders) + np.sum(result.ask_orders))
    trades = int(np.sum(total_fills))
    drawdown = drawdown_metrics(result.equity)
    inventory_variance = np.var(result.inventory, axis=0, ddof=0)
    inventory_variance.setflags(write=False)
    return MarketMakingMetrics(
        net_pnl=float(result.equity[-1] - result.equity[0]),
        maximum_drawdown=drawdown.maximum_drawdown,
        maximum_drawdown_fraction=drawdown.maximum_drawdown_fraction,
        maximum_drawdown_duration_steps=drawdown.maximum_drawdown_duration_steps,
        inventory_variance_by_asset=inventory_variance,
        spread_capture_after_fees=net_capture,
        submitted_orders=orders,
        trades=trades,
        quote_opportunity_to_trade_ratio=orders / trades if trades else None,
    )


def pair_selection_frequencies(
    selections: Sequence[Sequence[PairCalibration]],
) -> dict[tuple[str, str], float]:
    """Return fraction of runs selecting each canonical asset pair."""
    if not selections:
        raise ValueError("at least one pair-selection run is required")
    counts: dict[tuple[str, str], int] = {}
    for selected_pairs in selections:
        run_pairs = {
            tuple(sorted((pair.asset_y, pair.asset_x)))
            for pair in selected_pairs
        }
        for pair in run_pairs:
            counts[pair] = counts.get(pair, 0) + 1
    return {
        pair: count / len(selections)
        for pair, count in sorted(counts.items())
    }
