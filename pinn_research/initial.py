"""Compatibility imports for the original single-module prototype.

New code should import from ``market_data``, ``baselines``, ``filters``, or
``backtest`` directly.
"""

from .backtest import BacktestResult, run_backtest
from .baselines import ASParameters, AvellanedaStoikov, QuotingPolicy, Quotes
from .filters import AdaptiveKalmanFilter, FilterResult
from .market_data import (
    MarketData,
    simulate_gbm,
    simulate_log_price_observations,
)
from .model_types import FloatArray, IntArray

__all__ = [
    "AdaptiveKalmanFilter",
    "ASParameters",
    "AvellanedaStoikov",
    "BacktestResult",
    "FilterResult",
    "FloatArray",
    "IntArray",
    "MarketData",
    "QuotingPolicy",
    "Quotes",
    "run_backtest",
    "simulate_gbm",
    "simulate_log_price_observations",
]
