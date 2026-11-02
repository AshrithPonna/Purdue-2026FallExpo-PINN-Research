"""Market data validation and reproducible price-path simulators."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model_types import FloatArray


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
    if (
        asset_count == 0
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
        or len(set(symbols)) != asset_count
    ):
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


def simulate_log_price_observations(
    market: MarketData, measurement_variance: float, seed: int
) -> FloatArray:
    """Add seeded Gaussian measurement noise to true log mid-prices."""
    if not np.isfinite(measurement_variance) or measurement_variance <= 0:
        raise ValueError("measurement_variance must be finite and positive")
    rng = np.random.default_rng(seed)
    observations = np.log(market.mid_prices) + rng.normal(
        0.0,
        np.sqrt(measurement_variance),
        size=market.mid_prices.shape,
    )
    observations.setflags(write=False)
    return observations
