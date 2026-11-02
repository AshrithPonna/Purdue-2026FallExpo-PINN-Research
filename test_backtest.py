import unittest

import numpy as np

from backtest import run_backtest
from baselines import ASParameters, AvellanedaStoikov
from market_data import MarketData


def make_parameters(
    symbols: tuple[str, ...] = ("A",),
    *,
    arrival_rate: float = 1_000_000.0,
    max_inventory: int = 1,
) -> ASParameters:
    count = len(symbols)
    return ASParameters(
        symbols=symbols,
        risk_aversion=0.1,
        volatility=np.zeros(count),
        arrival_rate=np.full(count, arrival_rate),
        distance_sensitivity=np.full(count, 0.01),
        min_inventory=np.zeros(count, dtype=np.int64),
        max_inventory=np.full(count, max_inventory, dtype=np.int64),
        fee_per_unit=np.zeros(count),
    )


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
        parameters = make_parameters(("B",), arrival_rate=1.0)
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
        symbols = ("A", "B")
        market = MarketData(
            ("0", "1", "2"),
            symbols,
            np.array([[100.0, 50.0], [100.0, 50.0], [100.0, 50.0]]),
        )
        parameters = make_parameters(symbols, arrival_rate=1.0)
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
