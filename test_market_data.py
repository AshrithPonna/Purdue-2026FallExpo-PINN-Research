import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from market_data import MarketData, simulate_gbm


class MarketDataTests(unittest.TestCase):
    def test_rejects_nonpositive_prices(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            MarketData(("0", "1"), ("A",), np.array([[10.0], [0.0]]))

    def test_gbm_is_positive_reproducible_and_chronological(self) -> None:
        arguments = {
            "symbols": ("A", "B"),
            "initial_prices": np.array([100.0, 50.0]),
            "log_drift": np.array([0.0, 0.0]),
            "covariance": np.array([[0.04, 0.01], [0.01, 0.09]]),
            "steps": 12,
            "dt": 1 / 252,
            "seed": 42,
        }
        first = simulate_gbm(**arguments)
        second = simulate_gbm(**arguments)

        self.assertEqual(first.mid_prices.shape, (13, 2))
        self.assertTrue(np.all(first.mid_prices > 0))
        np.testing.assert_array_equal(first.mid_prices, second.mid_prices)
        self.assertEqual(first.timestamps, tuple(sorted(first.timestamps)))

    def test_csv_interface_aligns_long_form_rows(self) -> None:
        rows = [
            ("2026-01-02T00:01:00Z", "B", "51"),
            ("2026-01-01T00:00:00Z", "A", "100"),
            ("2026-01-02T00:01:00Z", "A", "101"),
            ("2026-01-01T00:00:00Z", "B", "50"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("timestamp", "symbol", "mid_price"))
                writer.writerows(rows)
            market = MarketData.from_csv(path)

        self.assertEqual(market.symbols, ("A", "B"))
        np.testing.assert_array_equal(market.mid_prices, [[100, 50], [101, 51]])


if __name__ == "__main__":
    unittest.main()
