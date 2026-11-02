import unittest

import numpy as np

from pinn_research.hjb import InventoryHJBParameters, solve_inventory_hjb


class InventoryHJBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = InventoryHJBParameters(
            risk_aversion=0.1,
            volatility=0.2,
            arrival_rate=1.0,
            distance_sensitivity=1.5,
            min_inventory=-5,
            max_inventory=5,
            fee_per_unit=0.001,
            horizon=1.0,
            max_quote_distance=2.0,
        )

    def test_terminal_condition_and_value_shape(self) -> None:
        result = solve_inventory_hjb(self.parameters, time_steps=101)

        self.assertEqual(result.value_factor.shape, (101, 11))
        np.testing.assert_array_equal(result.inventory, np.arange(-5, 6))
        np.testing.assert_allclose(result.times[-1], self.parameters.horizon)
        np.testing.assert_allclose(result.value_factor[-1], np.ones(11), atol=1e-12)
        self.assertTrue(np.all(np.isfinite(result.value(100.0, 50.0))))

    def test_inventory_boundaries_disable_outward_fills(self) -> None:
        result = solve_inventory_hjb(self.parameters, time_steps=31)

        self.assertTrue(np.all(result.bid_offsets[:, 0] == 0.0))
        self.assertTrue(np.all(result.ask_offsets[:, -1] == 0.0))
        self.assertTrue(np.all(result.ask_offsets[:, 0] > 0.0))
        self.assertTrue(np.all(result.bid_offsets[:, -1] > 0.0))

    def test_hjb_residual_is_small_on_interior_time_grid(self) -> None:
        result = solve_inventory_hjb(self.parameters, time_steps=201)

        self.assertTrue(np.all(np.isfinite(result.residual)))
        self.assertLess(float(np.max(np.abs(result.residual[2:-2]))), 1e-6)

    def test_quote_offsets_obey_feasible_bounds(self) -> None:
        result = solve_inventory_hjb(self.parameters, time_steps=31)

        self.assertTrue(np.all(result.bid_offsets >= 0.0))
        self.assertTrue(np.all(result.ask_offsets >= 0.0))
        self.assertTrue(
            np.all(result.bid_offsets <= self.parameters.max_quote_distance)
        )
        self.assertTrue(
            np.all(result.ask_offsets <= self.parameters.max_quote_distance)
        )

    def test_rejects_invalid_inventory_or_time_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum inventory"):
            InventoryHJBParameters(
                risk_aversion=0.1,
                volatility=0.2,
                arrival_rate=1.0,
                distance_sensitivity=1.5,
                min_inventory=2,
                max_inventory=-2,
                fee_per_unit=0.0,
                horizon=1.0,
                max_quote_distance=2.0,
            )
        with self.assertRaisesRegex(ValueError, "at least 3"):
            solve_inventory_hjb(self.parameters, time_steps=2)


if __name__ == "__main__":
    unittest.main()
