import unittest

import numpy as np

from run_integrated_stat_arb import _synthetic_observations
from run_monte_carlo import _continued_observations


class MonteCarloPathTests(unittest.TestCase):
    def test_paths_retain_training_prefix_and_vary_by_seed(self) -> None:
        training_market = _synthetic_observations(
            seed=23,
            count=100,
            regime_change=100,
        )
        training = training_market.mid_prices[:80]
        first = _continued_observations(
            training,
            pair_intercept=0.0,
            hedge_ratio=0.7,
            spread_mean=0.0,
            seed=101,
            test_steps=20,
            regime_change=10,
        )
        repeated = _continued_observations(
            training,
            pair_intercept=0.0,
            hedge_ratio=0.7,
            spread_mean=0.0,
            seed=101,
            test_steps=20,
            regime_change=10,
        )
        second = _continued_observations(
            training,
            pair_intercept=0.0,
            hedge_ratio=0.7,
            spread_mean=0.0,
            seed=102,
            test_steps=20,
            regime_change=10,
        )

        self.assertEqual(first.shape, (100, 3))
        np.testing.assert_array_equal(first[:79], training[:79])
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first[79:], second[79:]))
        self.assertTrue(np.all(first > 0.0))

    def test_rejects_invalid_regime_change(self) -> None:
        training = np.ones((10, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "horizon and regime-change"):
            _continued_observations(
                training,
                pair_intercept=0.0,
                hedge_ratio=1.0,
                spread_mean=0.0,
                seed=1,
                test_steps=10,
                regime_change=10,
            )


if __name__ == "__main__":
    unittest.main()
