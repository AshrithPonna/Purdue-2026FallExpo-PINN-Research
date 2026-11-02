import unittest

import numpy as np

from pinn_research.baselines import ASParameters
from pinn_research.evaluation import (
    ScenarioConfig,
    evaluate_scenarios,
    paired_terminal_pnl_differences,
    simulate_scenario,
)
from pinn_research.hjb import InventoryHJBParameters
from pinn_research.pinn import PINNTrainingConfig, train_inventory_pinn


def make_model() -> tuple[InventoryHJBParameters, ASParameters]:
    hjb_parameters = InventoryHJBParameters(
        risk_aversion=0.1,
        volatility=0.2,
        arrival_rate=2.0,
        distance_sensitivity=1.5,
        min_inventory=-2,
        max_inventory=2,
        fee_per_unit=0.001,
        horizon=1.0,
        max_quote_distance=2.0,
    )
    as_parameters = ASParameters(
        symbols=("ASSET",),
        risk_aversion=0.1,
        volatility=np.array([0.2]),
        arrival_rate=np.array([2.0]),
        distance_sensitivity=np.array([1.5]),
        min_inventory=np.array([-2]),
        max_inventory=np.array([2]),
        fee_per_unit=np.array([0.001]),
    )
    return hjb_parameters, as_parameters


class ScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ScenarioConfig(
            steps=20,
            dt=0.05,
            initial_price=100.0,
            volatility=0.2,
            jump_intensity=2.0,
            jump_scale=0.02,
            jump_shape=3.0,
            jump_log_cap=0.2,
            regime_volatility_multiplier=2.0,
        )

    def test_seeded_scenarios_are_reproducible_and_positive(self) -> None:
        for scenario in ("ordinary", "heavy_tail", "regime_change"):
            first = simulate_scenario(scenario, self.config, seed=13)
            second = simulate_scenario(scenario, self.config, seed=13)
            np.testing.assert_array_equal(first.mid_prices, second.mid_prices)
            self.assertTrue(np.all(first.mid_prices > 0))

    def test_heavy_tail_requires_jump_parameters(self) -> None:
        config = ScenarioConfig(steps=20, dt=0.05, initial_price=100.0, volatility=0.2)
        with self.assertRaisesRegex(ValueError, "requires positive jump"):
            simulate_scenario("heavy_tail", config, seed=1)

    def test_regime_change_has_declared_volatility_shift(self) -> None:
        ordinary = simulate_scenario("ordinary", self.config, seed=13)
        regime = simulate_scenario("regime_change", self.config, seed=13)

        self.assertFalse(np.array_equal(ordinary.mid_prices, regime.mid_prices))


class EvaluationTests(unittest.TestCase):
    def test_runs_paired_policies_on_all_scenarios(self) -> None:
        hjb_parameters, as_parameters = make_model()
        pinn = train_inventory_pinn(
            hjb_parameters,
            PINNTrainingConfig(
                iterations=10,
                collocation_times=4,
                hidden_width=8,
                hidden_layers=1,
                seed=1,
            ),
        )
        config = ScenarioConfig(
            steps=4,
            dt=0.25,
            initial_price=100.0,
            volatility=0.2,
            jump_intensity=1.0,
            jump_scale=0.02,
        )
        runs = evaluate_scenarios(
            pinn,
            hjb_parameters,
            as_parameters,
            config,
            seeds=(3, 5),
            initial_cash=1_000.0,
        )

        self.assertEqual(len(runs), 12)
        differences = paired_terminal_pnl_differences(runs)
        self.assertEqual(set(differences), {"ordinary", "heavy_tail", "regime_change"})
        self.assertEqual(differences["ordinary"].shape, (2,))

    def test_rejects_mismatched_policy_parameters(self) -> None:
        hjb_parameters, as_parameters = make_model()
        pinn = train_inventory_pinn(
            hjb_parameters,
            PINNTrainingConfig(
                iterations=1,
                collocation_times=2,
                hidden_width=4,
                hidden_layers=1,
            ),
        )
        config = ScenarioConfig(
            steps=4,
            dt=0.25,
            initial_price=100.0,
            volatility=0.3,
            jump_intensity=1.0,
            jump_scale=0.02,
        )
        with self.assertRaisesRegex(ValueError, "match ordinary scenario volatility"):
            evaluate_scenarios(
                pinn,
                hjb_parameters,
                as_parameters,
                config,
                seeds=(3,),
                initial_cash=1_000.0,
            )


if __name__ == "__main__":
    unittest.main()
