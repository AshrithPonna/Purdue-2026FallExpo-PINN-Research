import unittest

import numpy as np

from hjb import InventoryHJBParameters, solve_inventory_hjb
from pinn import PINNTrainingConfig, train_inventory_pinn


class InventoryPINNTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = InventoryHJBParameters(
            risk_aversion=0.1,
            volatility=0.2,
            arrival_rate=1.0,
            distance_sensitivity=1.5,
            min_inventory=-2,
            max_inventory=2,
            fee_per_unit=0.001,
            horizon=1.0,
            max_quote_distance=2.0,
        )

    def test_training_approximates_numerical_reference(self) -> None:
        reference = solve_inventory_hjb(self.parameters, time_steps=51)
        pinn = train_inventory_pinn(
            self.parameters,
            PINNTrainingConfig(
                iterations=500,
                collocation_times=32,
                hidden_width=32,
                hidden_layers=2,
                seed=7,
            ),
        )

        comparison = pinn.compare(reference)
        self.assertLess(comparison.value_factor_rmse, 0.06)
        self.assertLess(comparison.value_factor_max_error, 0.12)
        self.assertLess(comparison.terminal_rmse, 0.02)
        self.assertTrue(np.isfinite(comparison.mean_absolute_residual))
        self.assertTrue(np.isfinite(comparison.max_absolute_residual))

    def test_prediction_validates_time_domain(self) -> None:
        pinn = train_inventory_pinn(
            self.parameters,
            PINNTrainingConfig(
                iterations=2,
                collocation_times=2,
                hidden_width=4,
                hidden_layers=1,
            ),
        )
        with self.assertRaisesRegex(ValueError, "within the HJB horizon"):
            pinn.predict_value_factor(np.array([-0.01]))


if __name__ == "__main__":
    unittest.main()
