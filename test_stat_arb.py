import unittest

import numpy as np

from market_data import MarketData
from stat_arb import StatArbConfig, backtest_stat_arb, fit_cointegrated_pairs


def make_cointegrated_prices(seed: int, observations: int = 1_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.0, 0.01, observations))
    spread = np.zeros(observations)
    for index in range(1, observations):
        spread[index] = 0.8 * spread[index - 1] + rng.normal(0.0, 0.01)
    independent = np.cumsum(rng.normal(0.0, 0.01, observations))
    log_prices = np.column_stack((0.7 * common + spread, common, independent))
    return np.exp(log_prices)


def make_market(prices: np.ndarray, start: int = 0) -> MarketData:
    return MarketData(
        tuple(f"{index:06d}" for index in range(start, start + len(prices))),
        ("Y", "X", "Z"),
        prices,
    )


class PairSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = make_cointegrated_prices(seed=42)
        self.training = make_market(self.prices[:700])
        self.config = StatArbConfig(
            cointegration_fdr=0.05,
            min_half_life=0.5,
            max_half_life=100.0,
            max_pairs=1,
        )

    def test_selects_cointegrated_pair_on_training_data(self) -> None:
        model = fit_cointegrated_pairs(self.training, self.config)

        self.assertEqual(len(model.pairs), 1)
        pair = model.pairs[0]
        self.assertEqual((pair.asset_y, pair.asset_x), ("Y", "X"))
        self.assertLess(pair.fdr_qvalue, self.config.cointegration_fdr)
        self.assertGreater(pair.hedge_ratio, 0.5)
        self.assertLess(pair.hedge_ratio, 0.9)
        self.assertGreaterEqual(pair.half_life, self.config.min_half_life)
        self.assertLessEqual(pair.half_life, self.config.max_half_life)

    def test_fits_only_pairs_that_pass_cointegration_filters(self) -> None:
        with self.assertRaisesRegex(ValueError, "no pairs passed"):
            fit_cointegrated_pairs(
                self.training,
                StatArbConfig(cointegration_fdr=1e-12, min_half_life=0.5, max_half_life=100),
            )

    def test_requires_sufficient_training_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 101"):
            fit_cointegrated_pairs(make_market(self.prices[:100]), self.config)

    def test_selects_multiple_non_overlapping_pairs_from_asset_universe(self) -> None:
        rng = np.random.default_rng(22)
        observations = 1_000
        x1 = np.cumsum(rng.normal(0.0, 0.01, observations))
        x2 = np.cumsum(rng.normal(0.0, 0.01, observations))
        spread1 = np.zeros(observations)
        spread2 = np.zeros(observations)
        for index in range(1, observations):
            spread1[index] = 0.75 * spread1[index - 1] + rng.normal(0.0, 0.01)
            spread2[index] = 0.8 * spread2[index - 1] + rng.normal(0.0, 0.01)
        unrelated = np.cumsum(rng.normal(0.0, 0.01, observations))
        prices = np.exp(
            np.column_stack(
                (0.7 * x1 + spread1, x1, -0.4 * x2 + spread2, x2, unrelated)
            )
        )
        training = MarketData(
            tuple(f"{index:06d}" for index in range(700)),
            ("Y1", "X1", "Y2", "X2", "Z"),
            prices[:700],
        )

        model = fit_cointegrated_pairs(
            training,
            StatArbConfig(
                cointegration_fdr=0.05,
                min_half_life=0.5,
                max_half_life=100,
                max_pairs=2,
            ),
        )

        self.assertEqual(len(model.pairs), 2)
        selected_assets = [
            asset
            for pair in model.pairs
            for asset in (pair.asset_y, pair.asset_x)
        ]
        self.assertEqual(len(selected_assets), len(set(selected_assets)))


class PairBacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = make_cointegrated_prices(seed=42)
        self.training = make_market(self.prices[:700])
        self.test_data = make_market(self.prices[700:], start=700)
        self.config = StatArbConfig(
            entry_z=1.0,
            exit_z=0.25,
            cointegration_fdr=0.05,
            min_half_life=0.5,
            max_half_life=100.0,
            max_pairs=1,
            gross_leverage=1.0,
            transaction_cost_bps=1.0,
        )
        self.model = fit_cointegrated_pairs(self.training, self.config)

    def test_backtest_trades_selected_pair_on_held_out_data(self) -> None:
        result = backtest_stat_arb(
            self.model,
            self.test_data,
            initial_capital=100_000.0,
        )

        self.assertEqual(result.equity.shape, (self.test_data.mid_prices.shape[0],))
        self.assertTrue(np.all(np.isfinite(result.equity)))
        self.assertTrue(np.all(result.transaction_costs >= 0))
        self.assertGreater(result.turnover.sum(), 0.0)
        self.assertGreater(result.transaction_costs.sum(), 0.0)
        self.assertTrue(np.all(np.isin(result.pair_positions, (-1.0, 0.0, 1.0))))
        self.assertTrue(np.all(result.holdings[-1] == 0.0))
        self.assertAlmostEqual(
            result.terminal_pnl,
            result.equity[-1] - result.initial_capital,
        )
        self.assertLessEqual(
            np.max(np.sum(np.abs(result.holdings * self.test_data.mid_prices), axis=1)),
            100_000.0 + 1e-8,
        )

    def test_test_data_does_not_change_fitted_pair_calibration(self) -> None:
        altered_test = make_market(
            np.exp(np.column_stack((
                np.log(self.test_data.mid_prices[:, 0]) + 0.2,
                np.log(self.test_data.mid_prices[:, 1]),
                np.log(self.test_data.mid_prices[:, 2]),
            ))),
            start=700,
        )
        original_pairs = self.model.pairs
        original_result = backtest_stat_arb(
            self.model,
            self.test_data,
            initial_capital=100_000,
        )
        altered_result = backtest_stat_arb(
            self.model,
            altered_test,
            initial_capital=100_000,
        )
        no_cost_model = fit_cointegrated_pairs(
            self.training,
            StatArbConfig(
                entry_z=1.0,
                exit_z=0.25,
                cointegration_fdr=0.05,
                min_half_life=0.5,
                max_half_life=100.0,
                transaction_cost_bps=0.0,
            ),
        )
        charged = backtest_stat_arb(self.model, self.test_data, initial_capital=100_000)
        uncharged = backtest_stat_arb(no_cost_model, self.test_data, initial_capital=100_000)

        self.assertEqual(original_pairs, self.model.pairs)
        self.assertFalse(np.array_equal(original_result.zscores, altered_result.zscores))
        self.assertGreaterEqual(uncharged.terminal_pnl, charged.terminal_pnl)

    def test_rejects_mismatched_asset_order(self) -> None:
        wrong_order = MarketData(
            self.test_data.timestamps,
            ("X", "Y", "Z"),
            self.test_data.mid_prices[:, [1, 0, 2]],
        )
        with self.assertRaisesRegex(ValueError, "must match training symbols"):
            backtest_stat_arb(self.model, wrong_order, initial_capital=100_000)


if __name__ == "__main__":
    unittest.main()
