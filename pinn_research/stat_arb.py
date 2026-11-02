"""Training-calibrated cointegrated-pairs selection and backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.stattools import coint

from .market_data import MarketData
from .model_types import FloatArray


@dataclass(frozen=True)
class PairCalibration:
    asset_y: str
    asset_x: str
    intercept: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    cointegration_pvalue: float
    fdr_qvalue: float
    half_life: float


@dataclass(frozen=True)
class StatArbConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    cointegration_fdr: float = 0.05
    min_half_life: float = 1.0
    max_half_life: float = 100.0
    max_pairs: int = 1
    gross_leverage: float = 1.0
    transaction_cost_bps: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if not np.isfinite(self.exit_z) or not 0 <= self.exit_z < self.entry_z:
            raise ValueError("exit_z must be finite, nonnegative, and below entry_z")
        if (
            not np.isfinite(self.cointegration_fdr)
            or not 0 < self.cointegration_fdr <= 1
        ):
            raise ValueError("cointegration_fdr must be in (0, 1]")
        if (
            not np.isfinite(self.min_half_life)
            or not np.isfinite(self.max_half_life)
            or self.min_half_life <= 0
            or self.max_half_life < self.min_half_life
        ):
            raise ValueError("half-life bounds must be finite and positive")
        if not isinstance(self.max_pairs, int) or self.max_pairs < 1:
            raise ValueError("max_pairs must be a positive integer")
        if not np.isfinite(self.gross_leverage) or self.gross_leverage <= 0:
            raise ValueError("gross_leverage must be finite and positive")
        if (
            not np.isfinite(self.transaction_cost_bps)
            or self.transaction_cost_bps < 0
        ):
            raise ValueError("transaction_cost_bps must be finite and nonnegative")


@dataclass(frozen=True)
class StatArbModel:
    symbols: tuple[str, ...]
    pairs: tuple[PairCalibration, ...]
    config: StatArbConfig


@dataclass(frozen=True)
class StatArbResult:
    timestamps: tuple[str, ...]
    symbols: tuple[str, ...]
    selected_pairs: tuple[PairCalibration, ...]
    initial_capital: float
    equity: FloatArray
    cash: FloatArray
    holdings: FloatArray
    gross_exposure: FloatArray
    zscores: FloatArray
    pair_positions: FloatArray
    turnover: FloatArray
    transaction_costs: FloatArray

    @property
    def terminal_pnl(self) -> float:
        return float(self.equity[-1] - self.initial_capital)


def _half_life(spread: FloatArray) -> float:
    lagged = spread[:-1]
    changes = np.diff(spread)
    if np.std(lagged) <= np.finfo(np.float64).eps:
        return float("inf")
    slope = float(linregress(lagged, changes).slope)
    persistence = 1.0 + slope
    if not 0.0 < persistence < 1.0:
        return float("inf")
    return float(-np.log(2.0) / np.log(persistence))


def _benjamini_hochberg(pvalues: FloatArray) -> FloatArray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    count = pvalues.size
    order = np.argsort(pvalues)
    ranked = pvalues[order] * count / np.arange(1, count + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted


def fit_cointegrated_pairs(
    training_data: MarketData,
    config: StatArbConfig = StatArbConfig(),
) -> StatArbModel:
    """Select non-overlapping Engle-Granger pairs using training data only."""
    if training_data.steps < 100:
        raise ValueError("at least 101 training observations are required")
    log_prices = np.log(training_data.mid_prices)
    candidates: list[PairCalibration] = []
    for y_index, x_index in combinations(range(len(training_data.symbols)), 2):
        y = log_prices[:, y_index]
        x = log_prices[:, x_index]
        if np.std(y) <= np.finfo(np.float64).eps or np.std(x) <= np.finfo(np.float64).eps:
            continue
        design = np.column_stack((np.ones_like(x), x))
        intercept, hedge_ratio = np.linalg.lstsq(design, y, rcond=None)[0]
        spread = y - intercept - hedge_ratio * x
        spread_std = float(np.std(spread, ddof=1))
        if not np.isfinite(spread_std) or spread_std <= np.finfo(np.float64).eps:
            continue
        statistic, pvalue, _ = coint(y, x, trend="c", autolag="aic")
        half_life = _half_life(spread)
        if not np.isfinite(statistic) or not np.isfinite(pvalue):
            continue
        candidates.append(
            PairCalibration(
                asset_y=training_data.symbols[y_index],
                asset_x=training_data.symbols[x_index],
                intercept=float(intercept),
                hedge_ratio=float(hedge_ratio),
                spread_mean=float(np.mean(spread)),
                spread_std=spread_std,
                cointegration_pvalue=float(pvalue),
                fdr_qvalue=1.0,
                half_life=half_life,
            )
        )
    if not candidates:
        raise ValueError("no non-degenerate candidate asset pairs found")

    qvalues = _benjamini_hochberg(
        np.asarray([pair.cointegration_pvalue for pair in candidates], dtype=np.float64)
    )
    calibrated = [
        PairCalibration(
            asset_y=pair.asset_y,
            asset_x=pair.asset_x,
            intercept=pair.intercept,
            hedge_ratio=pair.hedge_ratio,
            spread_mean=pair.spread_mean,
            spread_std=pair.spread_std,
            cointegration_pvalue=pair.cointegration_pvalue,
            fdr_qvalue=float(qvalue),
            half_life=pair.half_life,
        )
        for pair, qvalue in zip(candidates, qvalues)
    ]
    eligible = [
        pair for pair in calibrated
        if pair.fdr_qvalue <= config.cointegration_fdr
        and config.min_half_life <= pair.half_life <= config.max_half_life
    ]
    eligible.sort(key=lambda pair: (pair.fdr_qvalue, pair.half_life, pair.asset_y, pair.asset_x))

    selected: list[PairCalibration] = []
    used_assets: set[str] = set()
    for pair in eligible:
        if pair.asset_x in used_assets or pair.asset_y in used_assets:
            continue
        selected.append(pair)
        used_assets.update((pair.asset_x, pair.asset_y))
        if len(selected) == config.max_pairs:
            break
    if not selected:
        raise ValueError(
            "no pairs passed the cointegration FDR and half-life filters; "
            "do not fall back to an uncalibrated strategy"
        )
    return StatArbModel(training_data.symbols, tuple(selected), config)


def backtest_stat_arb(
    model: StatArbModel,
    test_data: MarketData,
    *,
    initial_capital: float,
) -> StatArbResult:
    """Trade calibrated pair spreads on held-out data with turnover costs."""
    if test_data.symbols != model.symbols:
        raise ValueError("test symbols must match training symbols in order")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    symbol_indices = {symbol: index for index, symbol in enumerate(test_data.symbols)}
    pair_count = len(model.pairs)
    time_count, asset_count = test_data.mid_prices.shape
    equity = np.empty(time_count, dtype=np.float64)
    cash_history = np.empty(time_count, dtype=np.float64)
    holdings_history = np.zeros((time_count, asset_count), dtype=np.float64)
    gross_exposure = np.zeros(time_count, dtype=np.float64)
    zscores = np.empty((time_count, pair_count), dtype=np.float64)
    pair_positions = np.zeros((time_count, pair_count), dtype=np.float64)
    turnover = np.zeros(time_count, dtype=np.float64)
    transaction_costs = np.zeros(time_count, dtype=np.float64)
    holdings = np.zeros(asset_count, dtype=np.float64)
    cash = float(initial_capital)
    active_directions = np.zeros(pair_count, dtype=np.float64)
    capital_per_pair = initial_capital * model.config.gross_leverage / pair_count
    fee_rate = model.config.transaction_cost_bps / 10_000.0

    pair_holdings = np.zeros((asset_count, pair_count), dtype=np.float64)
    for time_index, prices in enumerate(test_data.mid_prices):
        for pair_index, pair in enumerate(model.pairs):
            y_index = symbol_indices[pair.asset_y]
            x_index = symbol_indices[pair.asset_x]
            spread = (
                np.log(prices[y_index])
                - pair.intercept
                - pair.hedge_ratio * np.log(prices[x_index])
            )
            zscore = (spread - pair.spread_mean) / pair.spread_std
            zscores[time_index, pair_index] = zscore
            direction = active_directions[pair_index]
            previous_direction = direction
            if direction == 0.0:
                if zscore >= model.config.entry_z:
                    direction = -1.0
                elif zscore <= -model.config.entry_z:
                    direction = 1.0
            elif direction > 0.0 and zscore >= -model.config.exit_z:
                direction = 0.0
            elif direction < 0.0 and zscore <= model.config.exit_z:
                direction = 0.0
            active_directions[pair_index] = direction
            pair_positions[time_index, pair_index] = direction
            if direction != previous_direction:
                if direction == 0.0:
                    pair_holdings[:, pair_index] = 0.0
                else:
                    y_notional = capital_per_pair / (1.0 + abs(pair.hedge_ratio))
                    pair_holdings[y_index, pair_index] = (
                        direction * y_notional / prices[y_index]
                    )
                    pair_holdings[x_index, pair_index] = (
                        -direction * pair.hedge_ratio * y_notional / prices[x_index]
                    )
        for pair_index in range(pair_count):
            pair_gross = float(
                np.sum(np.abs(pair_holdings[:, pair_index] * prices))
            )
            if pair_gross > capital_per_pair:
                pair_holdings[:, pair_index] *= capital_per_pair / pair_gross
        target_holdings = pair_holdings.sum(axis=1)

        trade_delta = target_holdings - holdings
        step_turnover = float(np.sum(np.abs(trade_delta * prices)))
        step_cost = fee_rate * step_turnover
        cash -= float(trade_delta @ prices) + step_cost
        holdings = target_holdings
        equity[time_index] = cash + float(holdings @ prices)
        gross_exposure[time_index] = float(np.sum(np.abs(holdings * prices)))
        cash_history[time_index] = cash
        holdings_history[time_index] = holdings
        turnover[time_index] = step_turnover
        transaction_costs[time_index] = step_cost

    if np.any(holdings != 0.0):
        final_prices = test_data.mid_prices[-1]
        final_turnover = float(np.sum(np.abs(holdings * final_prices)))
        final_cost = fee_rate * final_turnover
        cash += float(holdings @ final_prices) - final_cost
        holdings = np.zeros_like(holdings)
        equity[-1] = cash
        cash_history[-1] = cash
        holdings_history[-1] = holdings
        gross_exposure[-1] = 0.0
        turnover[-1] += final_turnover
        transaction_costs[-1] += final_cost

    for array in (
        equity, cash_history, holdings_history, gross_exposure, zscores,
        pair_positions, turnover, transaction_costs,
    ):
        array.setflags(write=False)
    return StatArbResult(
        timestamps=test_data.timestamps,
        symbols=test_data.symbols,
        selected_pairs=model.pairs,
        initial_capital=float(initial_capital),
        equity=equity,
        cash=cash_history,
        holdings=holdings_history,
        gross_exposure=gross_exposure,
        zscores=zscores,
        pair_positions=pair_positions,
        turnover=turnover,
        transaction_costs=transaction_costs,
    )
