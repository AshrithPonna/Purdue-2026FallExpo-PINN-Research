import unittest

import numpy as np

from pinn_research.pair_hjb import PairHJBParameters, _operator, solve_pair_hjb
from pinn_research.pair_pinn import PairPINNTrainingConfig, train_pair_pinn


class PairHJBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = PairHJBParameters(
            mean_reversion_speed=0.2,
            spread_volatility=0.3,
            spread_scale=0.01,
            risk_penalty=0.02,
            trading_penalty=0.1,
            maximum_inventory=1.0,
            maximum_trading_rate=0.5,
            z_limit=4.0,
            horizon=5.0,
        )

    def test_reference_solution_is_finite_and_inventory_bounded(self) -> None:
        result = solve_pair_hjb(
            self.parameters,
            time_steps=11,
            z_points=17,
            inventory_points=9,
        )

        self.assertEqual(result.value.shape, (11, 17, 9))
        self.assertTrue(np.all(np.isfinite(result.value)))
        self.assertTrue(np.all(np.isfinite(result.residual)))
        self.assertLessEqual(np.max(np.abs(result.trading_rate)), 0.5)
        self.assertTrue(np.all(result.trading_rate[:, :, 0] >= 0.0))
        self.assertTrue(np.all(result.trading_rate[:, :, -1] <= 0.0))

    def test_execution_fee_creates_a_no_trade_band(self) -> None:
        z_grid = np.linspace(-4.0, 4.0, 9)
        inventory_grid = np.linspace(-1.0, 1.0, 9)
        value = np.broadcast_to(
            0.04 * inventory_grid[np.newaxis, :],
            (z_grid.size, inventory_grid.size),
        )
        _, no_fee_rate = _operator(
            value,
            z_grid,
            inventory_grid,
            self.parameters,
        )
        fee_parameters = PairHJBParameters(
            **{
                **self.parameters.__dict__,
                "execution_fee_rate": 0.01,
            }
        )
        _, fee_aware_rate = _operator(
            value,
            z_grid,
            inventory_grid,
            fee_parameters,
        )

        self.assertAlmostEqual(no_fee_rate[4, 4], 0.4)
        self.assertAlmostEqual(fee_aware_rate[4, 4], 0.3)

    def test_pair_pinn_trains_and_produces_bounded_rates(self) -> None:
        reference = solve_pair_hjb(
            self.parameters,
            time_steps=11,
            z_points=17,
            inventory_points=9,
        )
        pinn = train_pair_pinn(
            self.parameters,
            PairPINNTrainingConfig(
                iterations=12,
                collocation_points=16,
                data_batch_size=64,
                hidden_width=8,
                hidden_layers=1,
                seed=5,
            ),
            reference=reference,
        )

        comparison = pinn.compare(reference)
        self.assertTrue(np.isfinite(comparison.value_rmse))
        self.assertTrue(np.isfinite(comparison.mean_absolute_residual))
        self.assertLessEqual(
            abs(pinn.trading_rate(2.5, 1.0, 1.0)),
            self.parameters.maximum_trading_rate,
        )


if __name__ == "__main__":
    unittest.main()
