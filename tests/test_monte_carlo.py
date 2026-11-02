import unittest

import numpy as np

from pinn_research.run_integrated_stat_arb import _synthetic_observations
from pinn_research.run_monte_carlo import _continued_observations, _continued_paths


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

    def test_stress_scenarios_are_reproducible_and_distinct(self) -> None:
        training = np.exp(
            np.column_stack(
                (
                    np.linspace(0.0, 0.1, 100),
                    np.linspace(0.0, 0.2, 100),
                    np.linspace(0.0, -0.1, 100),
                )
            )
        )
        outputs = {}
        for scenario in (
            "volatility_shift",
            "heavy_tail",
            "cointegration_break",
        ):
            outputs[scenario] = _continued_paths(
                training,
                pair_intercept=0.0,
                hedge_ratio=0.7,
                spread_mean=0.0,
                seed=2026,
                test_steps=100,
                regime_change=40,
                stress_scenario=scenario,
            )
            repeated = _continued_paths(
                training,
                pair_intercept=0.0,
                hedge_ratio=0.7,
                spread_mean=0.0,
                seed=2026,
                test_steps=100,
                regime_change=40,
                stress_scenario=scenario,
            )
            np.testing.assert_array_equal(outputs[scenario][0], repeated[0])
            np.testing.assert_array_equal(outputs[scenario][1], repeated[1])

        self.assertFalse(
            np.array_equal(
                outputs["volatility_shift"][1],
                outputs["heavy_tail"][1],
            )
        )
        self.assertFalse(
            np.array_equal(
                outputs["volatility_shift"][1],
                outputs["cointegration_break"][1],
            )
        )

    def test_rejects_unknown_stress_scenario(self) -> None:
        with self.assertRaisesRegex(ValueError, "stress scenario"):
            _continued_paths(
                np.ones((10, 3)),
                pair_intercept=0.0,
                hedge_ratio=1.0,
                spread_mean=0.0,
                seed=1,
                test_steps=10,
                regime_change=5,
                stress_scenario="optimistic",
            )


if __name__ == "__main__":
    unittest.main()
