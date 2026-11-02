import unittest

from pinn_research.run_stress_ablation import _latex_table


class StressAblationLatexTests(unittest.TestCase):
    def test_table_includes_abstract_metrics_and_method_caveats(self) -> None:
        rows = [
            {
                "scenario": "cointegration_break",
                "variant": "threshold_akf_variance_derisk",
                "mean_net_pnl": 13_000.0,
                "lower_tail_5pct_mean_pnl": 5_000.0,
                "mean_normalized_pair_inventory_variance": 0.01,
                "mean_spread_capture_efficiency": 0.004,
                "mean_quote_opportunities_per_fill_proxy": 3.1,
                "mean_max_drawdown_fraction": 0.018,
                "p95_max_drawdown_fraction": 0.045,
                "mean_max_drawdown_duration_steps": 127.4,
                "hjb_mean_absolute_pde_residual": "",
            },
            {
                "scenario": "cointegration_break",
                "variant": "hjb_fee_aware_akf_variance_derisk",
                "mean_net_pnl": 10_400.0,
                "lower_tail_5pct_mean_pnl": -3_600.0,
                "mean_normalized_pair_inventory_variance": 0.045,
                "mean_spread_capture_efficiency": 0.0028,
                "mean_quote_opportunities_per_fill_proxy": 3.04,
                "mean_max_drawdown_fraction": 0.037,
                "p95_max_drawdown_fraction": 0.091,
                "mean_max_drawdown_duration_steps": 143.0,
                "hjb_mean_absolute_pde_residual": 0.00627,
            },
        ]

        table = _latex_table(rows)

        self.assertIn(r"\begin{tabular}{llrrrrrrrrr}", table)
        self.assertIn(r"Mean P\&L", table)
        self.assertIn("Cointegration break", table)
        self.assertIn("---", table)
        self.assertIn("0.00627", table)
        self.assertIn("not an exchange order-to-trade ratio", table)


if __name__ == "__main__":
    unittest.main()
