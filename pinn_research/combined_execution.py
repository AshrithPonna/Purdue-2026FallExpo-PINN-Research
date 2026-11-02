"""Avellaneda-Stoikov execution of pair-HJB target inventories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .baselines import avellaneda_stoikov_offsets
from .market_data import MarketData
from .model_types import FloatArray, IntArray
from .stat_arb import StatArbModel, StatArbResult


class PairTradingPolicy(Protocol):
    def trading_rate(
        self,
        time_to_horizon: float,
        zscore: float,
        inventory: float,
    ) -> float:
        """Return the desired change in normalized pair inventory per time unit."""


@dataclass(frozen=True)
class ASPairExecutionResult:
    """Portfolio accounting plus quote-level and AKF-regime diagnostics."""

    portfolio: StatArbResult
    bid_quotes: FloatArray
    ask_quotes: FloatArray
    bid_fills: IntArray
    ask_fills: IntArray
    target_inventory: FloatArray
    regime_multiplier: FloatArray
    spread_capture_after_fees: float
    order_count: int
    fill_count: int


def execute_pair_targets_with_as(
    model: StatArbModel,
    filtered_test_data: MarketData,
    policies: tuple[PairTradingPolicy, ...],
    process_variances: FloatArray,
    *,
    training_process_variances: FloatArray,
    initial_capital: float,
    maximum_inventory: float,
    dt: float,
    risk_aversion: float,
    arrival_rate: float,
    distance_sensitivity: float,
    maximum_inventory_slots: int,
    regime_threshold: float,
    regime_multiplier_cap: float = 3.0,
    seed: int = 42,
) -> ASPairExecutionResult:
    """Use AS quotes to execute causal HJB target positions in each pair."""
    if filtered_test_data.symbols != model.symbols:
        raise ValueError("test symbols must match training symbols in order")
    if len(policies) != len(model.pairs):
        raise ValueError("one HJB-PINN policy is required per selected pair")
    if (
        not np.isfinite(initial_capital)
        or initial_capital <= 0
        or not np.isfinite(maximum_inventory)
        or maximum_inventory <= 0
        or not np.isfinite(dt)
        or dt <= 0
        or not np.isfinite(risk_aversion)
        or risk_aversion <= 0
        or not np.isfinite(arrival_rate)
        or arrival_rate <= 0
        or not np.isfinite(distance_sensitivity)
        or distance_sensitivity <= 0
        or not isinstance(maximum_inventory_slots, int)
        or maximum_inventory_slots < 1
        or not np.isfinite(regime_threshold)
        or regime_threshold <= 1
        or not np.isfinite(regime_multiplier_cap)
        or regime_multiplier_cap < regime_threshold
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("execution parameters must be finite and valid")

    variance = np.asarray(process_variances, dtype=np.float64)
    training_variance = np.asarray(training_process_variances, dtype=np.float64)
    expected_shape = filtered_test_data.mid_prices.shape
    if variance.shape != expected_shape or not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("test process variances must be finite positive values matching prices")
    if (
        training_variance.ndim != 2
        or training_variance.shape[1] != len(model.symbols)
        or training_variance.shape[0] < 2
        or not np.all(np.isfinite(training_variance))
        or np.any(training_variance <= 0)
    ):
        raise ValueError("training process variances must be finite positive values")
    symbols = model.symbols
    symbol_indices = {symbol: index for index, symbol in enumerate(symbols)}
    prices = filtered_test_data.mid_prices
    steps, asset_count = filtered_test_data.steps, len(symbols)
    pair_count = len(model.pairs)
    pair_budget = initial_capital * model.config.gross_leverage / pair_count
    fee_rate = model.config.transaction_cost_bps / 10_000.0
    max_slots = maximum_inventory_slots
    lot_sizes = np.zeros(asset_count, dtype=np.float64)
    pair_asset_indices: list[tuple[int, int]] = []
    for pair in model.pairs:
        y_index = symbol_indices[pair.asset_y]
        x_index = symbol_indices[pair.asset_x]
        beta = abs(pair.hedge_ratio)
        if beta <= np.finfo(np.float64).eps:
            raise ValueError(f"pair {pair.asset_y}/{pair.asset_x} has a zero hedge ratio")
        y_lot_notional = (
            pair_budget / (1.0 + beta) / (max_slots * maximum_inventory)
        )
        x_lot_notional = (
            pair_budget * beta / (1.0 + beta) / (max_slots * maximum_inventory)
        )
        lot_sizes[y_index] = y_lot_notional / prices[0, y_index]
        lot_sizes[x_index] = x_lot_notional / prices[0, x_index]
        pair_asset_indices.append((y_index, x_index))
    active_assets = lot_sizes > 0.0
    asset_pair_index = np.full(asset_count, -1, dtype=np.int64)
    for pair_index, (y_index, x_index) in enumerate(pair_asset_indices):
        asset_pair_index[y_index] = pair_index
        asset_pair_index[x_index] = pair_index

    training_baseline = np.median(training_variance, axis=0)
    if np.any(training_baseline <= 0):
        raise ValueError("training process variance baseline must be positive")

    cash = float(initial_capital)
    inventory = np.zeros(asset_count, dtype=np.int64)
    cash_history = np.empty(steps + 1, dtype=np.float64)
    holdings_history = np.zeros((steps + 1, asset_count), dtype=np.float64)
    equity_history = np.empty(steps + 1, dtype=np.float64)
    gross_history = np.zeros(steps + 1, dtype=np.float64)
    zscores = np.empty((steps + 1, pair_count), dtype=np.float64)
    pair_positions = np.zeros((steps + 1, pair_count), dtype=np.float64)
    turnover = np.zeros(steps + 1, dtype=np.float64)
    costs = np.zeros(steps + 1, dtype=np.float64)
    bid_quotes = np.full((steps, asset_count), np.nan, dtype=np.float64)
    ask_quotes = np.full((steps, asset_count), np.nan, dtype=np.float64)
    bid_orders = np.zeros((steps, asset_count), dtype=np.int64)
    ask_orders = np.zeros((steps, asset_count), dtype=np.int64)
    bid_fills = np.zeros((steps, asset_count), dtype=np.int64)
    ask_fills = np.zeros((steps, asset_count), dtype=np.int64)
    target_history = np.zeros((steps, asset_count), dtype=np.float64)
    regime_history = np.ones((steps + 1, asset_count), dtype=np.float64)
    rng = np.random.default_rng(seed)
    spread_capture = 0.0

    cash_history[0] = cash
    equity_history[0] = cash
    for pair_index, pair in enumerate(model.pairs):
        y_index, x_index = pair_asset_indices[pair_index]
        spread = (
            np.log(prices[0, y_index])
            - pair.intercept
            - pair.hedge_ratio * np.log(prices[0, x_index])
        )
        zscores[0, pair_index] = (spread - pair.spread_mean) / pair.spread_std

    for step in range(steps):
        mids = prices[step]
        multipliers = np.clip(
            np.sqrt(np.maximum(variance[step] / training_baseline, 1.0)),
            1.0,
            regime_multiplier_cap,
        )
        regime_history[step] = multipliers
        desired = np.zeros(asset_count, dtype=np.float64)
        for pair_index, (pair, policy) in enumerate(zip(model.pairs, policies)):
            y_index, x_index = pair_asset_indices[pair_index]
            spread = (
                np.log(mids[y_index])
                - pair.intercept
                - pair.hedge_ratio * np.log(mids[x_index])
            )
            zscore = (spread - pair.spread_mean) / pair.spread_std
            zscores[step, pair_index] = zscore
            actual_pair_inventory = (
                inventory[y_index] / max_slots * maximum_inventory
            )
            time_to_horizon = (steps - step) * dt
            rate = policy.trading_rate(
                time_to_horizon,
                zscore,
                actual_pair_inventory,
            )
            target = float(
                np.clip(
                    actual_pair_inventory + rate * dt,
                    -maximum_inventory,
                    maximum_inventory,
                )
            )
            pair_multiplier = max(multipliers[y_index], multipliers[x_index])
            if pair_multiplier >= regime_threshold:
                target /= pair_multiplier
            desired[y_index] = target / maximum_inventory * max_slots
            desired[x_index] = (
                -np.sign(pair.hedge_ratio)
                * target
                / maximum_inventory
                * max_slots
            )
        target_history[step] = desired

        time_fraction = (steps - step) * dt / (steps * dt)
        volatility = mids * np.sqrt(variance[step])
        asset_risk_aversion = np.full(asset_count, risk_aversion)
        asset_distance_sensitivity = np.full(asset_count, distance_sensitivity)
        inventory_error = (
            inventory.astype(np.float64) - desired
        ) * lot_sizes
        skew = (
            inventory_error
            * asset_risk_aversion
            * volatility**2
            * time_fraction
            * multipliers
        )
        bid_offset, ask_offset = avellaneda_stoikov_offsets(
            volatility,
            asset_risk_aversion,
            asset_distance_sensitivity,
            skew,
            time_fraction,
        )
        bids = np.maximum(0.0, mids - bid_offset)
        asks = mids + ask_offset
        quotes_enabled = active_assets & (lot_sizes > 0)
        bid_enabled = quotes_enabled & (inventory < max_slots)
        ask_enabled = quotes_enabled & (inventory > -max_slots)
        bid_quotes[step] = np.where(quotes_enabled, bids, np.nan)
        ask_quotes[step] = np.where(quotes_enabled, asks, np.nan)
        bid_orders[step] = bid_enabled
        ask_orders[step] = ask_enabled
        bid_probability = -np.expm1(
            -arrival_rate
            * np.exp(-asset_distance_sensitivity * bid_offset)
            * dt
        )
        ask_probability = -np.expm1(
            -arrival_rate
            * np.exp(-asset_distance_sensitivity * ask_offset)
            * dt
        )
        proposed_bids = (rng.random(asset_count) < bid_probability) & bid_enabled
        proposed_asks = (rng.random(asset_count) < ask_probability) & ask_enabled

        gross_by_pair = np.zeros(pair_count, dtype=np.float64)
        for pair_index, (y_index, x_index) in enumerate(pair_asset_indices):
            gross_by_pair[pair_index] = (
                abs(inventory[y_index] * lot_sizes[y_index] * mids[y_index])
                + abs(inventory[x_index] * lot_sizes[x_index] * mids[x_index])
            )
        step_turnover = 0.0
        step_cost = 0.0
        for asset_index in range(asset_count):
            pair_index = int(asset_pair_index[asset_index])
            if pair_index < 0:
                continue
            if proposed_bids[asset_index]:
                proposed_inventory = inventory[asset_index] + 1
                y_index, x_index = pair_asset_indices[pair_index]
                projected_y = proposed_inventory if asset_index == y_index else inventory[y_index]
                projected_x = proposed_inventory if asset_index == x_index else inventory[x_index]
                projected_gross = (
                    abs(projected_y * lot_sizes[y_index] * mids[y_index])
                    + abs(projected_x * lot_sizes[x_index] * mids[x_index])
                )
                if (
                    projected_gross <= pair_budget * (1.0 + 1e-12)
                    or projected_gross < gross_by_pair[pair_index]
                ):
                    notional = bids[asset_index] * lot_sizes[asset_index]
                    fee = fee_rate * notional
                    if cash >= notional + fee:
                        cash -= notional + fee
                        inventory[asset_index] = proposed_inventory
                        bid_fills[step, asset_index] = 1
                        gross_by_pair[pair_index] = projected_gross
                        fill_turnover = lot_sizes[asset_index] * mids[asset_index]
                        step_turnover += fill_turnover
                        step_cost += fee
                        spread_capture += (
                            (mids[asset_index] - bids[asset_index])
                            * lot_sizes[asset_index]
                            - fee
                        )
            if proposed_asks[asset_index] and inventory[asset_index] > -max_slots:
                proposed_inventory = inventory[asset_index] - 1
                y_index, x_index = pair_asset_indices[pair_index]
                projected_y = proposed_inventory if asset_index == y_index else inventory[y_index]
                projected_x = proposed_inventory if asset_index == x_index else inventory[x_index]
                projected_gross = (
                    abs(projected_y * lot_sizes[y_index] * mids[y_index])
                    + abs(projected_x * lot_sizes[x_index] * mids[x_index])
                )
                if (
                    projected_gross <= pair_budget * (1.0 + 1e-12)
                    or projected_gross < gross_by_pair[pair_index]
                ):
                    notional = asks[asset_index] * lot_sizes[asset_index]
                    fee = fee_rate * notional
                    cash += notional - fee
                    inventory[asset_index] = proposed_inventory
                    ask_fills[step, asset_index] = 1
                    gross_by_pair[pair_index] = projected_gross
                    fill_turnover = lot_sizes[asset_index] * mids[asset_index]
                    step_turnover += fill_turnover
                    step_cost += fee
                    spread_capture += (
                        (asks[asset_index] - mids[asset_index])
                        * lot_sizes[asset_index]
                        - fee
                    )

        holdings = inventory.astype(np.float64) * lot_sizes
        turnover[step + 1] = step_turnover
        costs[step + 1] = step_cost
        holdings_history[step + 1] = holdings
        cash_history[step + 1] = cash
        equity_history[step + 1] = cash + float(holdings @ prices[step + 1])
        gross_history[step + 1] = float(
            np.sum(np.abs(holdings * prices[step + 1]))
        )
        for pair_index, (pair, (y_index, x_index)) in enumerate(
            zip(model.pairs, pair_asset_indices)
        ):
            spread = (
                np.log(prices[step + 1, y_index])
                - pair.intercept
                - pair.hedge_ratio * np.log(prices[step + 1, x_index])
            )
            zscores[step + 1, pair_index] = (
                spread - pair.spread_mean
            ) / pair.spread_std
            pair_positions[step + 1, pair_index] = inventory[y_index] / max_slots

    final_prices = prices[-1]
    liquidation_turnover = float(
        np.sum(np.abs(inventory.astype(np.float64) * lot_sizes * final_prices))
    )
    liquidation_cost = fee_rate * liquidation_turnover
    cash += float((inventory.astype(np.float64) * lot_sizes) @ final_prices)
    cash -= liquidation_cost
    inventory[:] = 0
    turnover[-1] += liquidation_turnover
    costs[-1] += liquidation_cost
    cash_history[-1] = cash
    equity_history[-1] = cash
    holdings_history[-1] = 0.0
    gross_history[-1] = 0.0
    pair_positions[-1] = 0.0
    regime_history[-1] = np.clip(
        np.sqrt(np.maximum(variance[-1] / training_baseline, 1.0)),
        1.0,
        regime_multiplier_cap,
    )

    portfolio = StatArbResult(
        timestamps=filtered_test_data.timestamps,
        symbols=symbols,
        selected_pairs=model.pairs,
        initial_capital=float(initial_capital),
        equity=equity_history,
        cash=cash_history,
        holdings=holdings_history,
        gross_exposure=gross_history,
        zscores=zscores,
        pair_positions=pair_positions,
        turnover=turnover,
        transaction_costs=costs,
    )
    for array in (
        equity_history,
        cash_history,
        holdings_history,
        gross_history,
        zscores,
        pair_positions,
        turnover,
        costs,
    ):
        array.setflags(write=False)
    arrays = (
        bid_quotes,
        ask_quotes,
        bid_fills,
        ask_fills,
        target_history,
        regime_history,
    )
    for array in arrays:
        array.setflags(write=False)
    return ASPairExecutionResult(
        portfolio=portfolio,
        bid_quotes=bid_quotes,
        ask_quotes=ask_quotes,
        bid_fills=bid_fills,
        ask_fills=ask_fills,
        target_inventory=target_history,
        regime_multiplier=regime_history,
        spread_capture_after_fees=float(spread_capture),
        order_count=int(np.sum(bid_orders) + np.sum(ask_orders)),
        fill_count=int(np.sum(bid_fills) + np.sum(ask_fills)),
    )
