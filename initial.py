"""Multi-asset market data, GBM simulation, and AS market-making baseline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class MarketData:
    """Aligned positive mid-prices indexed by timestamp and asset."""

    timestamps: tuple[str, ...]
    symbols: tuple[str, ...]
    mid_prices: FloatArray

    def __post_init__(self) -> None:
        prices = np.asarray(self.mid_prices, dtype=np.float64)
        if prices.ndim != 2:
            raise ValueError("mid_prices must have shape (timestamps, assets)")
        if prices.shape != (len(self.timestamps), len(self.symbols)):
            raise ValueError("price dimensions must match timestamps and symbols")
        if len(self.timestamps) < 2:
            raise ValueError("at least two timestamps are required")
        if (
            not self.symbols
            or any(not isinstance(symbol, str) or not symbol for symbol in self.symbols)
            or len(set(self.symbols)) != len(self.symbols)
        ):
            raise ValueError("symbols must be non-empty and unique")
        if any(a >= b for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("timestamps must be strictly increasing")
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("mid-prices must be finite and positive")
        prices.setflags(write=False)
        object.__setattr__(self, "mid_prices", prices)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        timestamp_column: str = "timestamp",
        symbol_column: str = "symbol",
        price_column: str = "mid_price",
    ) -> MarketData:
        """Load long-form CSV rows and require a complete timestamp/asset grid."""
        rows_by_time: dict[str, dict[str, float]] = {}
        symbols: set[str] = set()
        with Path(path).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required = {timestamp_column, symbol_column, price_column}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"CSV must contain columns: {sorted(required)}")
            for row in reader:
                timestamp = row[timestamp_column]
                symbol = row[symbol_column]
                if not timestamp or not symbol:
                    raise ValueError("timestamps and symbols must not be empty")
                try:
                    price = float(row[price_column])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid price for {symbol} at {timestamp}") from error
                prices_at_time = rows_by_time.setdefault(timestamp, {})
                if symbol in prices_at_time:
                    raise ValueError(f"duplicate row for {symbol} at {timestamp}")
                prices_at_time[symbol] = price
                symbols.add(symbol)

        ordered_timestamps = tuple(sorted(rows_by_time))
        ordered_symbols = tuple(sorted(symbols))
        if not ordered_timestamps or not ordered_symbols:
            raise ValueError("CSV contains no market data")
        matrix = np.empty((len(ordered_timestamps), len(ordered_symbols)), dtype=np.float64)
        for time_index, timestamp in enumerate(ordered_timestamps):
            prices_at_time = rows_by_time[timestamp]
            if set(prices_at_time) != set(ordered_symbols):
                raise ValueError(f"missing asset price at timestamp {timestamp}")
            for symbol_index, symbol in enumerate(ordered_symbols):
                matrix[time_index, symbol_index] = prices_at_time[symbol]
        return cls(ordered_timestamps, ordered_symbols, matrix)

    @property
    def steps(self) -> int:
        return len(self.timestamps) - 1


def simulate_gbm(
    symbols: tuple[str, ...],
    initial_prices: FloatArray,
    log_drift: FloatArray,
    covariance: FloatArray,
    steps: int,
    dt: float,
    seed: int,
) -> MarketData:
    """Simulate correlated log-price diffusion with an explicit random seed."""
    initial = np.asarray(initial_prices, dtype=np.float64)
    drift = np.asarray(log_drift, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    asset_count = len(symbols)
    if asset_count == 0 or len(set(symbols)) != asset_count:
        raise ValueError("symbols must be non-empty and unique")
    if initial.shape != (asset_count,) or drift.shape != (asset_count,):
        raise ValueError("initial_prices and log_drift must have one value per symbol")
    if cov.shape != (asset_count, asset_count):
        raise ValueError("covariance must have shape (assets, assets)")
    if not np.all(np.isfinite(initial)) or np.any(initial <= 0):
        raise ValueError("initial prices must be finite and positive")
    if not np.all(np.isfinite(drift)) or not np.all(np.isfinite(cov)):
        raise ValueError("drift and covariance must be finite")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    try:
        cholesky = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((steps, asset_count)) @ cholesky.T
    log_returns = drift * dt + shocks * np.sqrt(dt)
    log_prices = np.vstack((np.log(initial), np.log(initial) + np.cumsum(log_returns, axis=0)))
    prices = np.exp(log_prices)
    timestamp_width = len(str(steps))
    timestamps = tuple(f"{index:0{timestamp_width}d}" for index in range(steps + 1))
    return MarketData(timestamps, symbols, prices)


@dataclass(frozen=True)
class ASParameters:
    """Per-asset parameters for the Avellaneda-Stoikov baseline."""

    symbols: tuple[str, ...]
    risk_aversion: float
    volatility: FloatArray
    arrival_rate: FloatArray
    distance_sensitivity: FloatArray
    min_inventory: IntArray
    max_inventory: IntArray
    fee_per_unit: FloatArray

    def __post_init__(self) -> None:
        volatility = np.asarray(self.volatility, dtype=np.float64)
        arrival_rate = np.asarray(self.arrival_rate, dtype=np.float64)
        sensitivity = np.asarray(self.distance_sensitivity, dtype=np.float64)
        minimum_values = np.asarray(self.min_inventory, dtype=np.float64)
        maximum_values = np.asarray(self.max_inventory, dtype=np.float64)
        if (
            not np.all(np.isfinite(minimum_values))
            or not np.all(np.isfinite(maximum_values))
            or np.any(minimum_values != np.floor(minimum_values))
            or np.any(maximum_values != np.floor(maximum_values))
        ):
            raise ValueError("inventory limits must be finite integers")
        minimum = minimum_values.astype(np.int64)
        maximum = maximum_values.astype(np.int64)
        fee = np.asarray(self.fee_per_unit, dtype=np.float64)
        arrays = (volatility, arrival_rate, sensitivity, minimum, maximum, fee)
        if (
            not self.symbols
            or any(not isinstance(symbol, str) or not symbol for symbol in self.symbols)
            or len(set(self.symbols)) != len(self.symbols)
        ):
            raise ValueError("parameter symbols must be non-empty and unique")
        if volatility.ndim != 1 or any(array.shape != volatility.shape for array in arrays):
            raise ValueError("all parameter arrays must be one-dimensional and equally sized")
        if volatility.size != len(self.symbols):
            raise ValueError("parameter arrays must have one value per symbol")
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be finite and positive")
        if not np.all(np.isfinite(volatility)) or np.any(volatility < 0):
            raise ValueError("volatility must be finite and nonnegative")
        if not np.all(np.isfinite(arrival_rate)) or np.any(arrival_rate <= 0):
            raise ValueError("arrival rates must be finite and positive")
        if not np.all(np.isfinite(sensitivity)) or np.any(sensitivity <= 0):
            raise ValueError("distance sensitivities must be finite and positive")
        if np.any(minimum > maximum):
            raise ValueError("minimum inventory cannot exceed maximum inventory")
        if not np.all(np.isfinite(fee)) or np.any(fee < 0):
            raise ValueError("fees must be finite and nonnegative")
        for name, array in zip(
            ("volatility", "arrival_rate", "distance_sensitivity",
             "min_inventory", "max_inventory", "fee_per_unit"),
            arrays,
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def asset_count(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True)
class Quotes:
    bid: FloatArray
    ask: FloatArray
    bid_enabled: NDArray[np.bool_]
    ask_enabled: NDArray[np.bool_]


class QuotingPolicy(Protocol):
    def quote(self, mid_prices: FloatArray, inventory: IntArray, time_to_horizon: float) -> Quotes:
        """Return one bid and ask quote per asset."""


@dataclass(frozen=True)
class AvellanedaStoikov:
    """Inventory-aware AS policy with non-crossing quote and inventory limits."""

    parameters: ASParameters

    def quote(
        self, mid_prices: FloatArray, inventory: IntArray, time_to_horizon: float
    ) -> Quotes:
        mids = np.asarray(mid_prices, dtype=np.float64)
        positions = np.asarray(inventory, dtype=np.int64)
        params = self.parameters
        if mids.shape != (params.asset_count,) or positions.shape != mids.shape:
            raise ValueError("prices and inventory must have one value per asset")
        if not np.all(np.isfinite(mids)) or np.any(mids <= 0):
            raise ValueError("mid-prices must be finite and positive")
        if not np.isfinite(time_to_horizon) or time_to_horizon < 0:
            raise ValueError("time_to_horizon must be finite and nonnegative")
        if np.any(positions < params.min_inventory) or np.any(positions > params.max_inventory):
            raise ValueError("inventory is outside configured bounds")

        gamma = params.risk_aversion
        variance = params.volatility**2
        inventory_skew = positions * gamma * variance * time_to_horizon
        half_spread = (
            0.5 * gamma * variance * time_to_horizon
            + np.log1p(gamma / params.distance_sensitivity) / gamma
        )
        bid_offset = np.maximum(0.0, half_spread + inventory_skew)
        ask_offset = np.maximum(0.0, half_spread - inventory_skew)
        bid = np.maximum(0.0, mids - bid_offset)
        ask = mids + ask_offset
        bid_enabled = positions < params.max_inventory
        ask_enabled = positions > params.min_inventory
        return Quotes(bid, ask, bid_enabled, ask_enabled)


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