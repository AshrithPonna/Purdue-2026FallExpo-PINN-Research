import unittest

import numpy as np

from pinn_research.baselines import ASParameters, AvellanedaStoikov


def make_parameters(
    *,
    volatility: float = 0.0,
    min_inventory: int = 0,
    max_inventory: int = 1,
) -> ASParameters:
    return ASParameters(
        symbols=("A",),
        risk_aversion=0.1,
        volatility=np.array([volatility]),
        arrival_rate=np.array([1.0]),
        distance_sensitivity=np.array([0.01]),
        min_inventory=np.array([min_inventory]),
        max_inventory=np.array([max_inventory]),
        fee_per_unit=np.array([0.0]),
    )


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


if __name__ == "__main__":
    unittest.main()
