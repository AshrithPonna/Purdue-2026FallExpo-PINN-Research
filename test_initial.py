"""Compatibility tests for the original single-module imports."""

import unittest

from initial import (
    AdaptiveKalmanFilter,
    ASParameters,
    AvellanedaStoikov,
    BacktestResult,
    FilterResult,
    MarketData,
    Quotes,
    run_backtest,
    simulate_gbm,
    simulate_log_price_observations,
)


class CompatibilityTests(unittest.TestCase):
    def test_original_imports_remain_available(self) -> None:
        self.assertTrue(all((
            AdaptiveKalmanFilter,
            ASParameters,
            AvellanedaStoikov,
            BacktestResult,
            FilterResult,
            MarketData,
            Quotes,
            run_backtest,
            simulate_gbm,
            simulate_log_price_observations,
        )))


if __name__ == "__main__":
    unittest.main()
