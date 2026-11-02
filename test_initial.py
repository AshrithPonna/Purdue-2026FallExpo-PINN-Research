import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from initial import (
    AdaptiveKalmanFilter,
    ASParameters,
    AvellanedaStoikov,
    MarketData,
    run_backtest,
    simulate_gbm,
    simulate_log_price_observations,
)


def make_parameters(
    *,
    arrival_rate: float = 1_000_000.0,
    volatility: float = 0.0,
    min_inventory: int = 0,
    max_inventory: int = 1,
) -> ASParameters:
    return ASParameters(
        symbols=("A",),
        risk_aversion=0.1,
        volatility=np.array([volatility]),
        arrival_rate=np.array([arrival_rate]),
        distance_sensitivity=np.array([0.01]),
        min_inventory=np.array([min_inventory]),
        max_inventory=np.array([max_inventory]),
        fee_per_unit=np.array([0.0]),
    )


class MarketDataTests(unittest.TestCase):
    def test_rejects_nonpositive_prices(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            MarketData(("0", "1"), ("A",), np.array([[10.0], [0.0]]))

    def test_gbm_is_positive_reproducible_and_chronological(self) -> None:
        arguments = {
            "symbols": ("A", "B"),
            "initial_prices": np.array([100.0, 50.0]),
            "log_drift": np.array([0.0, 0.0]),
            "covariance": np.array([[0.04, 0.01], [0.01, 0.09]]),
            "steps": 12,
            "dt": 1 / 252,
            "seed": 42,
        }
        first = simulate_gbm(**arguments)
        second = simulate_gbm(**arguments)

        self.assertEqual(first.mid_prices.shape, (13, 2))
        self.assertTrue(np.all(first.mid_prices > 0))
        np.testing.assert_array_equal(first.mid_prices, second.mid_prices)
        self.assertEqual(first.timestamps, tuple(sorted(first.timestamps)))

    def test_csv_interface_aligns_long_form_rows(self) -> None:
        rows = [
            ("2026-01-02T00:01:00Z", "B", "51"),
            ("2026-01-01T00:00:00Z", "A", "100"),
            ("2026-01-02T00:01:00Z", "A", "101"),
            ("2026-01-01T00:00:00Z", "B", "50"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("timestamp", "symbol", "mid_price"))
                writer.writerows(rows)
            market = MarketData.from_csv(path)

        self.assertEqual(market.symbols, ("A", "B"))
        np.testing.assert_array_equal(market.mid_prices, [[100, 50], [101, 51]])


class AvellanedaStoikovTests(unittest.TestCase):
    def test_positive_inventory_skews_quotes_to_reduce_inventory(self) -> None:
        policy = AvellanedaStoikov(make_parameters(volatility=0.2, max_inventory=3))
        neutral = policy.quote(np.array([100.0]), np.array([0]), 1.0)
        long = policy.quote(np.array([100.0]), np.array([2]), 1.0)

        self.assertLess(long.bid[0], neutral.bid[0])
        self.assertLess(long.ask[0], neutral.ask[0])

    def test_inventory_bounds_disable_risk_increasing_side(self) -> None:
        policy = AvellanedaStoikov(make_parameters(min_inventory=-1, max_inventory=1))
        at_maximum = policy.quote(np.array([100.0]), np.array([1]), 1.0)
        at_minimum = policy.quote(np.array([100.0]), np.array([-1]), 1.0)

        self.assertFalse(at_maximum.bid_enabled[0])
        self.assertTrue(at_maximum.ask_enabled[0])
        self.assertTrue(at_minimum.bid_enabled[0])
        self.assertFalse(at_minimum.ask_enabled[0])

    def test_rejects_fractional_inventory_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite integers"):
            ASParameters(
                symbols=("A",),
                risk_aversion=0.1,
                volatility=np.array([0.0]),
                arrival_rate=np.array([1.0]),
                distance_sensitivity=np.array([1.0]),
                min_inventory=np.array([0.5]),
                max_inventory=np.array([1.0]),
                fee_per_unit=np.array([0.0]),
            )


class AdaptiveKalmanFilterTests(unittest.TestCase):
    def test_filter_reduces_seeded_measurement_error_vs_raw_observations(self) -> None:
        rng = np.random.default_rng(42)
        true_process_variance = 2e-5
        measurement_variance = 0.0025
        truth = np.concatenate(
            (np.zeros((1, 1)), np.cumsum(
                rng.normal(0.0, np.sqrt(true_process_variance), size=(500, 1)),
                axis=0,
            ))
        )
        market = MarketData(
            tuple(f"{index:04d}" for index in range(len(truth))),
            ("A",),
            np.exp(truth),
        )
        observations = simulate_log_price_observations(
            market, measurement_variance, seed=7
        )
        estimator = AdaptiveKalmanFilter(
            initial_process_variance=1e-4,
            measurement_variance=measurement_variance,
            initial_state_variance=0.01,
            adaptation_rate=0.05,
        )

        result = estimator.filter(observations)
        raw_rmse = np.sqrt(np.mean((observations[1:] - np.log(market.mid_prices[1:])) ** 2))
        filtered_rmse = np.sqrt(
            np.mean(
                (result.estimated_log_prices[1:] - np.log(market.mid_prices[1:])) ** 2
            )
        )

        self.assertLess(filtered_rmse, raw_rmse)
        self.assertTrue(np.all(result.process_variances > 0))
        self.assertTrue(np.all(np.isfinite(result.estimated_prices)))

    def test_filter_handles_assets_independently_and_causally(self) -> None:
        observations = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        estimator = AdaptiveKalmanFilter(
            initial_process_variance=0.1,
            measurement_variance=0.2,
            initial_state_variance=1.0,
            adaptation_rate=0.1,
        )

        full_result = estimator.filter(observations)
        prefix_result = estimator.filter(observations[:2])

        self.assertEqual(full_result.estimated_log_prices.shape, (3, 2))
        self.assertTrue(np.all(full_result.estimated_prices > 0))
        np.testing.assert_array_equal(
            full_result.estimated_log_prices[:2],
            prefix_result.estimated_log_prices,
        )

    def test_filter_rejects_nonfinite_observations(self) -> None:
        estimator = AdaptiveKalmanFilter(
            initial_process_variance=0.1,
            measurement_variance=0.2,
            initial_state_variance=1.0,
            adaptation_rate=0.1,
        )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            estimator.filter(np.array([[0.0], [np.nan]]))


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.market = MarketData(
            ("0", "1", "2"),
            ("A",),
            np.array([[100.0], [100.0], [100.0]]),
        )

    def test_unit_fills_obey_inventory_limits_and_are_reproducible(self) -> None:
        parameters = make_parameters()
        policy = AvellanedaStoikov(parameters)
        arguments = {
            "market": self.market,
            "policy": policy,
            "parameters": parameters,
            "initial_cash": 101.0,
            "dt": 1.0,
            "seed": 7,
        }
        first = run_backtest(**arguments)
        second = run_backtest(**arguments)

        self.assertEqual(first.inventory[:, 0].tolist(), [0, 1, 0])
        self.assertTrue(np.all(first.inventory >= parameters.min_inventory))
        self.assertTrue(np.all(first.inventory <= parameters.max_inventory))
        self.assertTrue(np.all(first.cash >= 0))
        np.testing.assert_array_equal(first.equity, second.equity)

    def test_backtest_does_not_buy_without_sufficient_cash(self) -> None:
        parameters = make_parameters()
        result = run_backtest(
            self.market,
            AvellanedaStoikov(parameters),
            parameters,
            initial_cash=1.0,
            dt=1.0,
            seed=7,
        )

        self.assertEqual(int(result.bid_fills.sum()), 0)
        self.assertEqual(int(result.inventory.sum()), 0)
        self.assertTrue(np.all(result.cash >= 0))

    def test_backtest_rejects_mismatched_asset_order(self) -> None:
        parameters = ASParameters(
            symbols=("B",),
            risk_aversion=0.1,
            volatility=np.array([0.0]),
            arrival_rate=np.array([1.0]),
            distance_sensitivity=np.array([1.0]),
            min_inventory=np.array([0]),
            max_inventory=np.array([1]),
            fee_per_unit=np.array([0.0]),
        )
        with self.assertRaisesRegex(ValueError, "must match parameter symbols"):
            run_backtest(
                self.market,
                AvellanedaStoikov(parameters),
                parameters,
                initial_cash=100.0,
                dt=1.0,
                seed=1,
            )

    def test_backtest_supports_multiple_assets(self) -> None:
        market = MarketData(
            ("0", "1", "2"),
            ("A", "B"),
            np.array([[100.0, 50.0], [100.0, 50.0], [100.0, 50.0]]),
        )
        parameters = ASParameters(
            symbols=("A", "B"),
            risk_aversion=0.1,
            volatility=np.array([0.0, 0.0]),
            arrival_rate=np.array([1.0, 1.0]),
            distance_sensitivity=np.array([0.01, 0.02]),
            min_inventory=np.array([0, 0]),
            max_inventory=np.array([1, 1]),
            fee_per_unit=np.array([0.0, 0.0]),
        )
        result = run_backtest(
            market,
            AvellanedaStoikov(parameters),
            parameters,
            initial_cash=200.0,
            dt=1.0,
            seed=1,
        )

        self.assertEqual(result.inventory.shape, (3, 2))
        self.assertEqual(result.equity.shape, (3,))


if __name__ == "__main__":
    unittest.main()
