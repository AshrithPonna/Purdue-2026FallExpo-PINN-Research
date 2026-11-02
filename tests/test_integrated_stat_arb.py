import unittest

import numpy as np

from pinn_research.filters import AdaptiveKalmanFilter
from pinn_research.integrated_stat_arb import (
    IntegratedStatArbConfig,
    run_integrated_stat_arb,
)
from pinn_research.market_data import MarketData
from pinn_research.metrics import statistical_arbitrage_metrics
from pinn_research.pair_pinn import PairPINNTrainingConfig
from pinn_research.stat_arb import StatArbConfig


def make_observed_pair_market(seed: int, observations: int = 800) -> MarketData:
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.0, 0.01, observations))
    spread = np.zeros(observations)
    for index in range(1, observations):
        innovation_scale = 0.01 if index < 600 else 0.03
        spread[index] = (
            0.8 * spread[index - 1]
            + rng.normal(0.0, innovation_scale)
        )
    unrelated = np.cumsum(rng.normal(0.0, 0.01, observations))
    log_prices = np.column_stack(
        (0.7 * common + spread, common, unrelated)
    )
    log_prices += rng.normal(0.0, 0.001, log_prices.shape)
    return MarketData(
        tuple(f"{index:05d}" for index in range(observations)),
        ("Y", "X", "Z"),
        np.exp(log_prices),
    )


class IntegratedStatArbTests(unittest.TestCase):
    def test_akf_and_pair_hjb_pinn_run_on_held_out_prices(self) -> None:
        market = make_observed_pair_market(42)
        run = run_integrated_stat_arb(
            market,
            training_observations=600,
            initial_capital=100_000.0,
            strategy_config=StatArbConfig(
                cointegration_fdr=0.05,
                min_half_life=0.5,
                max_half_life=100.0,
                max_pairs=1,
                gross_leverage=1.0,
                transaction_cost_bps=1.0,
            ),
            integration_config=IntegratedStatArbConfig(
                filter=AdaptiveKalmanFilter(
                    initial_process_variance=1e-4,
                    measurement_variance=1e-6,
                    initial_state_variance=1e-4,
                    adaptation_rate=0.05,
                ),
                pinn=PairPINNTrainingConfig(
                    iterations=20,
                    collocation_points=32,
                    data_batch_size=128,
                    hidden_width=12,
                    hidden_layers=2,
                    seed=17,
                ),
                hjb_time_steps=15,
                hjb_z_points=21,
                hjb_inventory_points=9,
            ),
        )

        result = run.result
        self.assertEqual(len(result.selected_pairs), 1)
        self.assertEqual(result.equity.shape, (201,))
        self.assertTrue(np.all(np.isfinite(result.equity)))
        self.assertTrue(np.all(np.isfinite(result.zscores)))
        self.assertTrue(np.all(np.abs(result.pair_positions) <= 1.0))
        self.assertTrue(np.all(result.holdings[-1] == 0.0))
        self.assertLessEqual(np.max(result.gross_exposure), 100_000.0 + 1e-7)
        self.assertGreater(result.turnover.sum(), 0.0)
        self.assertGreater(result.transaction_costs.sum(), 0.0)
        self.assertGreater(run.execution.order_count, 0)
        self.assertGreater(run.execution.fill_count, 0)
        self.assertGreater(run.execution.spread_capture_after_fees, 0.0)
        self.assertTrue(np.all(run.execution.target_inventory <= 10.0))
        self.assertTrue(np.all(run.execution.target_inventory >= -10.0))
        quoted = np.isfinite(run.execution.bid_quotes) & np.isfinite(
            run.execution.ask_quotes
        )
        self.assertTrue(np.all(
            run.execution.ask_quotes[quoted] >= run.execution.bid_quotes[quoted]
        ))
        self.assertTrue(np.all(run.execution.regime_multiplier >= 1.0))
        self.assertTrue(np.any(run.execution.regime_multiplier >= 1.5))
        self.assertTrue(
            np.isfinite(statistical_arbitrage_metrics(result).maximum_drawdown)
        )
        self.assertTrue(np.all(np.isfinite(run.filter_result.estimated_prices)))
        self.assertTrue(np.isfinite(run.pinn_comparisons[0].value_rmse))
        self.assertAlmostEqual(
            run.pair_pinns[0].parameters.execution_fee_rate,
            0.0001,
        )

    def test_requires_training_and_held_out_observations(self) -> None:
        with self.assertRaisesRegex(ValueError, "training_observations"):
            run_integrated_stat_arb(
                make_observed_pair_market(11, observations=120),
                training_observations=120,
                initial_capital=100_000.0,
            )


if __name__ == "__main__":
    unittest.main()
