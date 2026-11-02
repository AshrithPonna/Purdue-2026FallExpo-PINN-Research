import unittest

import numpy as np

from pinn_research.filters import AdaptiveKalmanFilter
from pinn_research.market_data import MarketData, simulate_log_price_observations


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
        raw_rmse = np.sqrt(
            np.mean((observations[1:] - np.log(market.mid_prices[1:])) ** 2)
        )
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


if __name__ == "__main__":
    unittest.main()
