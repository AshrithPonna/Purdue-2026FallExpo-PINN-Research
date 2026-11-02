"""Classical multi-asset market-making policies and their parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from model_types import FloatArray, IntArray


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
