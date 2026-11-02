"""Discrete execution simulation and portfolio bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines import ASParameters, QuotingPolicy
from market_data import MarketData
from model_types import FloatArray, IntArray


@dataclass(frozen=True)
class BacktestResult:
    timestamps: tuple[str, ...]
    symbols: tuple[str, ...]
    cash: FloatArray
    inventory: IntArray
    equity: FloatArray
    bid_quotes: FloatArray
    ask_quotes: FloatArray
    bid_orders: IntArray
    ask_orders: IntArray
    bid_fills: IntArray
    ask_fills: IntArray


def run_backtest(
    market: MarketData,
    policy: QuotingPolicy,
    parameters: ASParameters,
    initial_cash: float,
    dt: float,
    seed: int,
) -> BacktestResult:
    """Backtest unit orders with one Bernoulli fill opportunity per side/step."""
    if market.symbols != parameters.symbols:
        raise ValueError("market symbols must match parameter symbols in order")
    if not np.isfinite(initial_cash) or initial_cash < 0:
        raise ValueError("initial_cash must be finite and nonnegative")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    steps, assets = market.steps, len(market.symbols)
    cash_history = np.empty(steps + 1, dtype=np.float64)
    inventory_history = np.zeros((steps + 1, assets), dtype=np.int64)
    equity_history = np.empty(steps + 1, dtype=np.float64)
    bid_quotes = np.full((steps, assets), np.nan, dtype=np.float64)
    ask_quotes = np.full((steps, assets), np.nan, dtype=np.float64)
    bid_orders = np.zeros((steps, assets), dtype=np.int64)
    ask_orders = np.zeros((steps, assets), dtype=np.int64)
    bid_fills = np.zeros((steps, assets), dtype=np.int64)
    ask_fills = np.zeros((steps, assets), dtype=np.int64)
    cash = float(initial_cash)
    inventory = np.zeros(assets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    cash_history[0] = cash
    equity_history[0] = cash + inventory @ market.mid_prices[0]

    for step in range(steps):
        mids = market.mid_prices[step]
        tau = (steps - step) * dt
        quotes = policy.quote(mids, inventory, tau)
        bid_quotes[step] = quotes.bid
        ask_quotes[step] = quotes.ask
        bid_orders[step] = quotes.bid_enabled
        ask_orders[step] = quotes.ask_enabled
        bid_prob = -np.expm1(
            -parameters.arrival_rate
            * np.exp(-parameters.distance_sensitivity * (mids - quotes.bid))
            * dt
        )
        ask_prob = -np.expm1(
            -parameters.arrival_rate
            * np.exp(-parameters.distance_sensitivity * (quotes.ask - mids))
            * dt
        )
        bid_arrivals = rng.random(assets) < bid_prob
        ask_arrivals = rng.random(assets) < ask_prob

        for asset in range(assets):
            if bid_arrivals[asset] and quotes.bid_enabled[asset]:
                cost = quotes.bid[asset] + parameters.fee_per_unit[asset]
                if inventory[asset] < parameters.max_inventory[asset] and cash >= cost:
                    cash -= cost
                    inventory[asset] += 1
                    bid_fills[step, asset] = 1
            if ask_arrivals[asset] and quotes.ask_enabled[asset]:
                if inventory[asset] > parameters.min_inventory[asset]:
                    cash += quotes.ask[asset] - parameters.fee_per_unit[asset]
                    inventory[asset] -= 1
                    ask_fills[step, asset] = 1

        cash_history[step + 1] = cash
        inventory_history[step + 1] = inventory
        equity_history[step + 1] = cash + inventory @ market.mid_prices[step + 1]

    return BacktestResult(
        timestamps=market.timestamps,
        symbols=market.symbols,
        cash=cash_history,
        inventory=inventory_history,
        equity=equity_history,
        bid_quotes=bid_quotes,
        ask_quotes=ask_quotes,
        bid_orders=bid_orders,
        ask_orders=ask_orders,
        bid_fills=bid_fills,
        ask_fills=ask_fills,
    )
