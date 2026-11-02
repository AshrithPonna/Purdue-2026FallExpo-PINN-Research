"""Run a causal AKF + pair-level HJB-PINN synthetic experiment."""

from __future__ import annotations

import numpy as np

from filters import AdaptiveKalmanFilter
from integrated_stat_arb import (
    IntegratedStatArbConfig,
    run_integrated_stat_arb,
)
from market_data import MarketData
from metrics import statistical_arbitrage_metrics
from pair_pinn import PairPINNTrainingConfig
from stat_arb import StatArbConfig


def _synthetic_observations(seed: int, count: int) -> MarketData:
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.0, 0.01, count))
    spread = np.zeros(count)
    for index in range(1, count):
        spread[index] = 0.8 * spread[index - 1] + rng.normal(0.0, 0.008)
    unrelated = np.cumsum(rng.normal(0.0, 0.01, count))
    log_prices = np.column_stack(
        (0.7 * common + spread, common, unrelated)
    )
    observed_log_prices = log_prices + rng.normal(
        0.0, 0.003, size=log_prices.shape
    )
    return MarketData(
        tuple(f"{index:05d}" for index in range(count)),
        ("Y", "X", "NOISE"),
        np.exp(observed_log_prices),
    )


def main() -> None:
    run = run_integrated_stat_arb(
        _synthetic_observations(seed=23, count=1_250),
        training_observations=750,
        initial_capital=100_000.0,
        strategy_config=StatArbConfig(
            entry_z=2.0,
            exit_z=0.5,
            cointegration_fdr=0.05,
            min_half_life=1.0,
            max_half_life=100.0,
            max_pairs=1,
            gross_leverage=1.0,
            transaction_cost_bps=1.0,
        ),
        integration_config=IntegratedStatArbConfig(
            filter=AdaptiveKalmanFilter(
                initial_process_variance=1e-4,
                measurement_variance=9e-6,
                initial_state_variance=1e-3,
                adaptation_rate=0.05,
            ),
            pinn=PairPINNTrainingConfig(
                iterations=500,
                collocation_points=256,
                data_batch_size=512,
                hidden_width=32,
                hidden_layers=2,
                seed=19,
            ),
            hjb_time_steps=41,
            hjb_z_points=41,
            hjb_inventory_points=11,
        ),
    )
    metrics = statistical_arbitrage_metrics(run.result)
    print("Selected training pairs:")
    for pair in run.model.pairs:
        print(
            f"  {pair.asset_y}/{pair.asset_x}: "
            f"q={pair.fdr_qvalue:.3g}, half_life={pair.half_life:.2f}"
        )
    print(
        f"HJB-PINN comparison: value_rmse={run.pinn_comparisons[0].value_rmse:.6f}, "
        f"max_error={run.pinn_comparisons[0].value_max_error:.6f}, "
        f"mean_abs_residual={run.pinn_comparisons[0].mean_absolute_residual:.6f}"
    )
    print(
        f"AKF + pair HJB-PINN held-out synthetic result: "
        f"net_pnl={metrics.net_pnl:.2f}, "
        f"max_drawdown={metrics.maximum_drawdown:.2f}, "
        f"max_drawdown_fraction={metrics.maximum_drawdown_fraction:.4%}, "
        f"turnover={metrics.total_turnover:.2f}, "
        f"costs={metrics.total_transaction_costs:.2f}, "
        f"max_gross_exposure={metrics.maximum_gross_exposure:.2f}"
    )
    print(
        "Synthetic pipeline check only; it is not evidence of live-market "
        "profitability or execution quality."
    )


if __name__ == "__main__":
    main()
