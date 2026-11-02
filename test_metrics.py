import unittest

import numpy as np

from backtest import run_backtest
from baselines import ASParameters, AvellanedaStoikov
from market_data import MarketData
from metrics import (
    drawdown_metrics,
    market_making_metrics,
    pair_selection_frequencies,
    statistical_arbitrage_metrics,
)
from stat_arb import PairCalibration, StatArbConfig, StatArbResult


class DrawdownMetricTests(unittest.TestCase):
    def test_drawdown_depth_fraction_and_duration(self) -> None:
        metrics = drawdown_metrics(np.array([100.0, 110.0, 90.0, 95.0, 111.0, 105.0]))

        self.assertAlmostEqual(metrics.maximum_drawdown, 20.0)
        self.assertAlmostEqual(metrics.maximum_drawdown_fraction, 20.0 / 110.0)
        self.assertEqual(metrics.maximum_drawdown_duration_steps, 2)


class StrategyMetricTests(unittest.TestCase):
    def test_market_making_metrics_keep_spread_capture_distinct(self) -> None:
        market = MarketData(
            ("0", "1", "2"),
            ("A",),
            np.array([[100.0], [101.0], [100.0]]),
        )
        parameters = ASParameters(
            symbols=("A",),
            risk_aversion=0.1,
            volatility=np.array([0.0]),
            arrival_rate=np.array([1_000_000.0]),
            distance_sensitivity=np.array([0.01]),
            min_inventory=np.array([0]),
            max_inventory=np.array([1]),
            fee_per_unit=np.array([0.0]),
        )
        backtest = run_backtest(
            market,
            AvellanedaStoikov(parameters),
            parameters,
            initial_cash=200.0,
            dt=1.0,
            seed=7,
        )

        metrics = market_making_metrics(
            backtest,
            market.mid_prices,
            parameters.fee_per_unit,
        )
        self.assertEqual(metrics.trades, int(backtest.bid_fills.sum() + backtest.ask_fills.sum()))
        self.assertGreater(metrics.submitted_orders, 0)
        self.assertIsNotNone(metrics.quote_opportunity_to_trade_ratio)
        self.assertTrue(np.all(metrics.inventory_variance_by_asset >= 0))

    def test_stat_arb_metrics_include_initial_capital_peak(self) -> None:
        pair = PairCalibration(
            asset_y="Y",
            asset_x="X",
            intercept=0.0,
            hedge_ratio=1.0,
            spread_mean=0.0,
            spread_std=1.0,
            cointegration_pvalue=0.01,
            fdr_qvalue=0.01,
            half_life=5.0,
        )
        result = StatArbResult(
            timestamps=("0", "1", "2"),
            symbols=("Y", "X"),
            selected_pairs=(pair,),
            initial_capital=100.0,
            equity=np.array([99.0, 90.0, 101.0]),
            cash=np.array([99.0, 90.0, 101.0]),
            holdings=np.zeros((3, 2)),
            gross_exposure=np.zeros(3),
            zscores=np.zeros((3, 1)),
            pair_positions=np.zeros((3, 1)),
            turnover=np.array([1.0, 2.0, 0.0]),
            transaction_costs=np.array([0.01, 0.02, 0.0]),
        )

        metrics = statistical_arbitrage_metrics(result)

        self.assertAlmostEqual(metrics.net_pnl, 1.0)
        self.assertAlmostEqual(metrics.maximum_drawdown, 10.0)
        self.assertEqual(metrics.maximum_drawdown_duration_steps, 2)
        self.assertAlmostEqual(metrics.total_turnover, 3.0)
        self.assertAlmostEqual(metrics.total_transaction_costs, 0.03)
        self.assertAlmostEqual(metrics.transaction_cost_fraction_of_turnover, 0.01)

    def test_pair_selection_frequency_uses_canonical_pair_names(self) -> None:
        first = PairCalibration("Y", "X", 0.0, 1.0, 0.0, 1.0, 0.01, 0.02, 2.0)
        reversed_pair = PairCalibration("X", "Y", 0.0, 1.0, 0.0, 1.0, 0.01, 0.02, 2.0)
        other = PairCalibration("B", "A", 0.0, 1.0, 0.0, 1.0, 0.01, 0.02, 2.0)

        frequencies = pair_selection_frequencies(
            ((first,), (reversed_pair, other))
        )

        self.assertEqual(frequencies[("X", "Y")], 1.0)
        self.assertEqual(frequencies[("A", "B")], 0.5)


if __name__ == "__main__":
    unittest.main()
