"""Run a causal AKF + pair-level HJB-PINN synthetic experiment."""

from __future__ import annotations

import numpy as np

from .combined_execution import execute_pair_targets_with_as
from .integrated_stat_arb import (
    run_integrated_stat_arb,
)
from .market_data import MarketData
from .metrics import statistical_arbitrage_metrics
from .research_config import (
    INITIAL_CAPITAL,
    TEST_STEPS,
    TRAINING_OBSERVATIONS,
    TRAINING_SEED,
    default_research_configs,
)


class _ThresholdPairPolicy:
    def __init__(
        self,
        entry_z: float,
        exit_z: float,
        maximum_rate: float,
        maximum_inventory: float,
    ) -> None:
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.maximum_rate = maximum_rate
        self.maximum_inventory = maximum_inventory

    def trading_rate(
        self,
        time_to_horizon: float,
        zscore: float,
        inventory: float,
    ) -> float:
        del time_to_horizon
        target = inventory
        if inventory == 0.0:
            if zscore >= self.entry_z:
                target = -self.maximum_inventory
            elif zscore <= -self.entry_z:
                target = self.maximum_inventory
        elif inventory > 0.0 and zscore >= -self.exit_z:
            target = 0.0
        elif inventory < 0.0 and zscore <= self.exit_z:
            target = 0.0
        return float(
            np.clip(
                target - inventory,
                -self.maximum_rate,
                self.maximum_rate,
            )
        )


def _synthetic_observations(
    seed: int,
    count: int,
    regime_change: int,
) -> MarketData:
    rng = np.random.default_rng(seed)
    common_increments = rng.normal(0.0, 0.01, count)
    common_increments[regime_change:] *= 2.0
    common = np.cumsum(common_increments)
    spread = np.zeros(count)
    for index in range(1, count):
        innovation_scale = 0.008 if index < regime_change else 0.02
        spread[index] = (
            0.8 * spread[index - 1]
            + rng.normal(0.0, innovation_scale)
        )
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
    observed_market = _synthetic_observations(
        seed=TRAINING_SEED,
        count=TRAINING_OBSERVATIONS + TEST_STEPS,
        regime_change=TRAINING_OBSERVATIONS,
    )
    strategy_config, integration_config = default_research_configs()
    run = run_integrated_stat_arb(
        observed_market,
        training_observations=TRAINING_OBSERVATIONS,
        initial_capital=INITIAL_CAPITAL,
        strategy_config=strategy_config,
        integration_config=integration_config,
    )
    training_observations = TRAINING_OBSERVATIONS
    filtered_test_data = MarketData(
        observed_market.timestamps[training_observations - 1 :],
        observed_market.symbols,
        run.filter_result.estimated_prices[training_observations - 1 :],
    )
    threshold_policy = _ThresholdPairPolicy(
        strategy_config.entry_z,
        strategy_config.exit_z,
        integration_config.maximum_trading_rate,
        integration_config.maximum_inventory,
    )
    threshold_execution = execute_pair_targets_with_as(
        run.model,
        filtered_test_data,
        tuple(threshold_policy for _ in run.model.pairs),
        run.filter_result.process_variances[training_observations - 1 :],
        training_process_variances=run.filter_result.process_variances[
            :training_observations
        ],
        initial_capital=100_000.0,
        maximum_inventory=integration_config.maximum_inventory,
        dt=integration_config.execution_dt,
        risk_aversion=integration_config.as_risk_aversion,
        arrival_rate=integration_config.as_arrival_rate,
        distance_sensitivity=integration_config.as_distance_sensitivity,
        maximum_inventory_slots=integration_config.maximum_inventory_slots,
        regime_threshold=integration_config.volatility_regime_threshold,
        regime_multiplier_cap=integration_config.volatility_regime_cap,
        seed=integration_config.execution_seed,
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
        f"AS execution: spread_capture_after_fees="
        f"{run.execution.spread_capture_after_fees:.2f}, "
        f"quote_opportunities={run.execution.order_count}, "
        f"fills={run.execution.fill_count}, "
        f"high_volatility_asset_time_fraction="
        f"{np.mean(run.execution.regime_multiplier >= 1.5):.2%}"
    )
    threshold_metrics = statistical_arbitrage_metrics(
        threshold_execution.portfolio
    )
    print(
        f"Matched AS threshold baseline: net_pnl={threshold_metrics.net_pnl:.2f}, "
        f"max_drawdown={threshold_metrics.maximum_drawdown:.2f}, "
        f"turnover={threshold_metrics.total_turnover:.2f}, "
        f"costs={threshold_metrics.total_transaction_costs:.2f}, "
        f"pnl_difference="
        f"{metrics.net_pnl - threshold_metrics.net_pnl:.2f}"
    )
    print(
        "Synthetic pipeline check only; it is not evidence of live-market "
        "profitability or execution quality."
    )


if __name__ == "__main__":
    main()
