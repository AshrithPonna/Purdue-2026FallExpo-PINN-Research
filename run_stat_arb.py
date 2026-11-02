"""Reproduce the synthetic multi-pair statistical-arbitrage experiment."""

from __future__ import annotations

import numpy as np

from metrics import pair_selection_frequencies, statistical_arbitrage_metrics
from market_data import MarketData
from stat_arb import StatArbConfig, backtest_stat_arb, fit_cointegrated_pairs


def _synthetic_prices(seed: int, observations: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x1 = np.cumsum(rng.normal(0.0, 0.01, observations))
    x2 = np.cumsum(rng.normal(0.0, 0.008, observations))
    spread1 = np.zeros(observations)
    spread2 = np.zeros(observations)
    for index in range(1, observations):
        spread1[index] = 0.8 * spread1[index - 1] + rng.normal(0.0, 0.008)
        spread2[index] = 0.75 * spread2[index - 1] + rng.normal(0.0, 0.01)
    unrelated = np.cumsum(rng.normal(0.0, 0.01, observations))
    log_prices = np.column_stack(
        (0.7 * x1 + spread1, x1, -0.5 * x2 + spread2, x2, unrelated)
    )
    return np.exp(log_prices)


def main() -> None:
    training_steps = 750
    test_steps = 500
    symbols = ("Y1", "X1", "Y2", "X2", "NOISE")
    config = StatArbConfig(
        entry_z=2.0,
        exit_z=0.5,
        cointegration_fdr=0.05,
        min_half_life=1.0,
        max_half_life=100.0,
        max_pairs=2,
        gross_leverage=1.0,
        transaction_cost_bps=1.0,
    )
    pnl_values: list[float] = []
    metric_runs = []
    pair_selections = []
    for seed in (11, 23, 37, 51, 79):
        prices = _synthetic_prices(seed, training_steps + test_steps)
        training = MarketData(
            tuple(f"{index:06d}" for index in range(training_steps)),
            symbols,
            prices[:training_steps],
        )
        test = MarketData(
            tuple(
                f"{index:06d}"
                for index in range(training_steps, training_steps + test_steps)
            ),
            symbols,
            prices[training_steps:],
        )
        model = fit_cointegrated_pairs(training, config)
        result = backtest_stat_arb(model, test, initial_capital=100_000.0)
        pnl_values.append(result.terminal_pnl)
        metric_runs.append(statistical_arbitrage_metrics(result))
        pair_selections.append(model.pairs)
        selected = ", ".join(
            f"{pair.asset_y}/{pair.asset_x} "
            f"(q={pair.fdr_qvalue:.3g}, half-life={pair.half_life:.2f})"
            for pair in model.pairs
        )
        print(
            f"seed={seed} pairs=[{selected}] "
            f"test_PnL={result.terminal_pnl:.2f} "
            f"costs={result.transaction_costs.sum():.2f}"
        )
    pnl = np.asarray(pnl_values, dtype=np.float64)
    print(
        f"synthetic test P&L: mean={pnl.mean():.2f}, "
        f"sample_sd={pnl.std(ddof=1):.2f}, runs={pnl.size}"
    )
    fields = (
        ("max_drawdown", "maximum_drawdown"),
        ("max_drawdown_fraction", "maximum_drawdown_fraction"),
        ("drawdown_duration_steps", "maximum_drawdown_duration_steps"),
        ("mean_gross_exposure", "mean_gross_exposure"),
        ("max_gross_exposure", "maximum_gross_exposure"),
        ("turnover", "total_turnover"),
        ("transaction_costs", "total_transaction_costs"),
        ("cost_fraction_of_turnover", "transaction_cost_fraction_of_turnover"),
        ("mean_absolute_spread_zscore", "mean_absolute_spread_zscore"),
        ("active_pair_fraction", "active_pair_fraction"),
        ("inventory_variance_mean", "inventory_variance_by_asset"),
    )
    for label, field in fields:
        values = np.asarray(
            [
                np.mean(getattr(metrics, field))
                if field == "inventory_variance_by_asset"
                else getattr(metrics, field)
                for metrics in metric_runs
            ],
            dtype=np.float64,
        )
        print(
            f"{label}: mean={values.mean():.4f}, "
            f"sample_sd={values.std(ddof=1):.4f}"
        )
    frequencies = pair_selection_frequencies(pair_selections)
    print("pair selection frequency:")
    for pair, frequency in frequencies.items():
        print(f"  {pair[0]}/{pair[1]}: {frequency:.0%}")


if __name__ == "__main__":
    main()
