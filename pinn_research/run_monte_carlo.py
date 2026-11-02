"""Paired Monte Carlo evaluation of HJB-PINN versus threshold pair control."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t

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
from .run_integrated_stat_arb import (
    _ThresholdPairPolicy,
    _synthetic_observations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "monte_carlo_results_as_units.csv"


def _continued_paths(
    training_observations: np.ndarray,
    *,
    pair_intercept: float,
    hedge_ratio: float,
    spread_mean: float,
    seed: int,
    test_steps: int,
    regime_change: int,
    stress_scenario: str = "volatility_shift",
) -> tuple[np.ndarray, np.ndarray]:
    """Append one noisy OU-pair path and return observations plus latent truth."""
    if training_observations.ndim != 2 or training_observations.shape[1] != 3:
        raise ValueError("the Monte Carlo generator expects three training assets")
    if (
        not isinstance(test_steps, int)
        or test_steps < 2
        or not isinstance(regime_change, int)
        or not 1 <= regime_change < test_steps
        or stress_scenario
        not in ("volatility_shift", "heavy_tail", "cointegration_break")
    ):
        raise ValueError(
            "test horizon and regime-change step are invalid for stress scenario"
        )
    rng = np.random.default_rng(seed)
    training_logs = np.log(training_observations)
    initial_y, initial_x, initial_noise = training_logs[-1]
    spread_state = initial_y - pair_intercept - hedge_ratio * initial_x
    common = np.empty(test_steps + 1, dtype=np.float64)
    spread = np.empty(test_steps + 1, dtype=np.float64)
    unrelated = np.empty(test_steps + 1, dtype=np.float64)
    common[0] = initial_x
    spread[0] = spread_state
    unrelated[0] = initial_noise

    for step in range(1, test_steps + 1):
        stressed = step >= regime_change
        common_scale = 0.02 if stressed else 0.01
        spread_scale = 0.02 if stressed else 0.008
        common_shock = rng.normal()
        spread_shock = rng.normal()
        if stress_scenario == "heavy_tail" and stressed:
            common_shock = rng.standard_t(3) / np.sqrt(3.0)
            spread_shock = rng.standard_t(3) / np.sqrt(3.0)
        common[step] = common[step - 1] + common_scale * common_shock
        if stress_scenario == "cointegration_break" and stressed:
            jump = 0.0
            if (step - regime_change) % 100 == 0:
                jump = float(rng.choice((-0.12, 0.12)))
            spread[step] = (
                spread[step - 1]
                + spread_scale * spread_shock
                + jump
            )
        else:
            spread[step] = (
                0.8 * spread[step - 1]
                + 0.2 * spread_mean
                + spread_scale * spread_shock
            )
        unrelated[step] = unrelated[step - 1] + rng.normal(0.0, 0.01)

    latent = np.column_stack(
        (
            pair_intercept + hedge_ratio * common + spread,
            common,
            unrelated,
        )
    )
    observed_tail = latent + rng.normal(0.0, 0.003, size=latent.shape)
    observed_tail[0] = training_logs[-1]
    full_log_prices = np.vstack((training_logs[:-1], observed_tail))
    latent_log_prices = np.vstack((training_logs[:-1], latent))
    with np.errstate(over="ignore"):
        prices = np.exp(full_log_prices)
    if (
        not np.all(np.isfinite(prices))
        or np.any(prices <= 0)
        or not np.all(np.isfinite(latent_log_prices))
    ):
        raise ValueError("synthetic Monte Carlo path produced invalid prices")
    return prices, latent_log_prices


def _continued_observations(
    training_observations: np.ndarray,
    *,
    pair_intercept: float,
    hedge_ratio: float,
    spread_mean: float,
    seed: int,
    test_steps: int,
    regime_change: int,
) -> np.ndarray:
    """Return observed prices for one seeded synthetic continuation."""
    prices, _ = _continued_paths(
        training_observations,
        pair_intercept=pair_intercept,
        hedge_ratio=hedge_ratio,
        spread_mean=spread_mean,
        seed=seed,
        test_steps=test_steps,
        regime_change=regime_change,
    )
    return prices


def _confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    critical = float(t.ppf(0.975, values.size - 1))
    return mean, mean - critical * standard_error, mean + critical * standard_error


def run_monte_carlo(
    simulations: int = 1_000,
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    first_seed: int = 10_000,
    execution_fee_aware: bool = True,
) -> None:
    if not isinstance(simulations, int) or simulations < 2:
        raise ValueError("simulations must be an integer of at least two")
    if not isinstance(first_seed, int) or first_seed < 0:
        raise ValueError("first_seed must be a nonnegative integer")

    training_count = TRAINING_OBSERVATIONS
    test_steps = TEST_STEPS
    initial_capital = INITIAL_CAPITAL
    training_market = _synthetic_observations(
        seed=TRAINING_SEED,
        count=training_count + test_steps,
        regime_change=training_count,
    )
    strategy_config, integration_config = default_research_configs(
        execution_seed=first_seed,
        execution_fee_aware=execution_fee_aware,
    )
    fitted = run_integrated_stat_arb(
        training_market,
        training_observations=training_count,
        initial_capital=initial_capital,
        strategy_config=strategy_config,
        integration_config=integration_config,
    )
    if len(fitted.model.pairs) != 1:
        raise RuntimeError(
            "the configured Monte Carlo benchmark expects exactly one fitted pair"
        )
    pair = fitted.model.pairs[0]
    threshold_policy = _ThresholdPairPolicy(
        strategy_config.entry_z,
        strategy_config.exit_z,
        integration_config.maximum_trading_rate,
        integration_config.maximum_inventory,
    )

    records: list[dict[str, float | int]] = []
    model_returns = np.empty(simulations, dtype=np.float64)
    baseline_returns = np.empty(simulations, dtype=np.float64)
    differences = np.empty(simulations, dtype=np.float64)
    model_drawdowns = np.empty(simulations, dtype=np.float64)
    baseline_drawdowns = np.empty(simulations, dtype=np.float64)
    raw_log_price_rmse = np.empty(simulations, dtype=np.float64)
    filtered_log_price_rmse = np.empty(simulations, dtype=np.float64)
    rmse_improvements = np.empty(simulations, dtype=np.float64)
    matched_fill_counts = 0
    divergent_position_runs = 0
    total_target_differences = 0
    absolute_differences: list[float] = []

    fixed_training_observations = training_market.mid_prices[:training_count]
    training_process_variances = fitted.filter_result.process_variances[
        :training_count
    ]
    for simulation_index in range(simulations):
        seed = first_seed + simulation_index
        regime_change = int(
            np.random.default_rng(seed + 50_000_000).integers(150, 351)
        )
        prices, latent_log_prices = _continued_paths(
            fixed_training_observations,
            pair_intercept=pair.intercept,
            hedge_ratio=pair.hedge_ratio,
            spread_mean=pair.spread_mean,
            seed=seed,
            test_steps=test_steps,
            regime_change=regime_change,
        )
        timestamps = tuple(f"{index:05d}" for index in range(prices.shape[0]))
        filter_result = integration_config.filter.filter(np.log(prices))
        test_slice = slice(training_count, None)
        raw_log_price_rmse[simulation_index] = float(
            np.sqrt(
                np.mean(
                    (
                        np.log(prices[test_slice])
                        - latent_log_prices[test_slice]
                    )
                    ** 2
                )
            )
        )
        filtered_log_price_rmse[simulation_index] = float(
            np.sqrt(
                np.mean(
                    (
                        filter_result.estimated_log_prices[test_slice]
                        - latent_log_prices[test_slice]
                    )
                    ** 2
                )
            )
        )
        rmse_improvements[simulation_index] = (
            raw_log_price_rmse[simulation_index]
            - filtered_log_price_rmse[simulation_index]
        )
        filtered_test = MarketData(
            timestamps[training_count - 1 :],
            training_market.symbols,
            filter_result.estimated_prices[training_count - 1 :],
        )
        test_variances = filter_result.process_variances[
            training_count - 1 :
        ]
        execution_arguments = {
            "training_process_variances": training_process_variances,
            "initial_capital": initial_capital,
            "maximum_inventory": integration_config.maximum_inventory,
            "dt": integration_config.execution_dt,
            "risk_aversion": integration_config.as_risk_aversion,
            "arrival_rate": integration_config.as_arrival_rate,
            "distance_sensitivity": integration_config.as_distance_sensitivity,
            "maximum_inventory_slots": integration_config.maximum_inventory_slots,
            "regime_threshold": integration_config.volatility_regime_threshold,
            "regime_multiplier_cap": integration_config.volatility_regime_cap,
            "seed": seed,
        }
        model_execution = execute_pair_targets_with_as(
            fitted.model,
            filtered_test,
            fitted.pair_pinns,
            test_variances,
            **execution_arguments,
        )
        baseline_execution = execute_pair_targets_with_as(
            fitted.model,
            filtered_test,
            (threshold_policy,),
            test_variances,
            **execution_arguments,
        )
        model_metrics = statistical_arbitrage_metrics(model_execution.portfolio)
        baseline_metrics = statistical_arbitrage_metrics(
            baseline_execution.portfolio
        )
        difference = model_metrics.net_pnl - baseline_metrics.net_pnl
        model_returns[simulation_index] = model_metrics.net_pnl
        baseline_returns[simulation_index] = baseline_metrics.net_pnl
        differences[simulation_index] = difference
        model_drawdowns[simulation_index] = model_metrics.maximum_drawdown_fraction
        baseline_drawdowns[simulation_index] = (
            baseline_metrics.maximum_drawdown_fraction
        )
        matched_fills = (
            np.array_equal(
                model_execution.bid_fills,
                baseline_execution.bid_fills,
            )
            and np.array_equal(
                model_execution.ask_fills,
                baseline_execution.ask_fills,
            )
        )
        matched_fill_counts += int(matched_fills)
        divergent_position_runs += int(
            not np.array_equal(
                model_execution.portfolio.pair_positions,
                baseline_execution.portfolio.pair_positions,
            )
        )
        position_steps_different = int(
            np.count_nonzero(
                model_execution.portfolio.pair_positions
                != baseline_execution.portfolio.pair_positions
            )
        )
        target_difference = float(
            np.mean(
                np.abs(
                    model_execution.target_inventory
                    - baseline_execution.target_inventory
                )
            )
        )
        total_target_differences += int(target_difference > 1e-8)
        absolute_differences.append(abs(difference))
        records.append(
            {
                "simulation": simulation_index + 1,
                "seed": seed,
                "regime_change_step": regime_change,
                "hjb_pinn_net_pnl": model_metrics.net_pnl,
                "threshold_net_pnl": baseline_metrics.net_pnl,
                "paired_pnl_difference": difference,
                "hjb_pinn_drawdown_fraction": model_metrics.maximum_drawdown_fraction,
                "threshold_drawdown_fraction": baseline_metrics.maximum_drawdown_fraction,
                "hjb_pinn_turnover": model_metrics.total_turnover,
                "threshold_turnover": baseline_metrics.total_turnover,
                "hjb_pinn_costs": model_metrics.total_transaction_costs,
                "threshold_costs": baseline_metrics.total_transaction_costs,
                "hjb_pinn_fills": model_execution.fill_count,
                "threshold_fills": baseline_execution.fill_count,
                "mean_absolute_target_difference": target_difference,
                "raw_log_price_rmse": raw_log_price_rmse[simulation_index],
                "filtered_log_price_rmse": filtered_log_price_rmse[
                    simulation_index
                ],
                "akf_log_price_rmse_improvement": rmse_improvements[
                    simulation_index
                ],
                "different_position_steps": position_steps_different,
                "same_fill_path": int(matched_fills),
            }
        )
        if (simulation_index + 1) % 100 == 0:
            print(f"completed {simulation_index + 1}/{simulations}")

    destination = Path(output_path)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    mean_delta, lower_delta, upper_delta = _confidence_interval(differences)
    bootstrap_rng = np.random.default_rng(first_seed + 90_000_000)
    bootstrap_means = np.empty(10_000, dtype=np.float64)
    for bootstrap_index in range(bootstrap_means.size):
        bootstrap_means[bootstrap_index] = np.mean(
            bootstrap_rng.choice(differences, size=simulations, replace=True)
        )
    bootstrap_lower, bootstrap_upper = np.quantile(
        bootstrap_means, (0.025, 0.975)
    )
    model_mean, model_lower, model_upper = _confidence_interval(model_returns)
    baseline_mean, baseline_lower, baseline_upper = _confidence_interval(
        baseline_returns
    )
    print(f"\nPaired synthetic Monte Carlo results: N={simulations}")
    print(
        f"HJB-PINN mean net P&L: {model_mean:.2f} "
        f"(95% CI {model_lower:.2f}, {model_upper:.2f}); "
        f"SD={np.std(model_returns, ddof=1):.2f}"
    )
    print(
        f"Threshold mean net P&L: {baseline_mean:.2f} "
        f"(95% CI {baseline_lower:.2f}, {baseline_upper:.2f}); "
        f"SD={np.std(baseline_returns, ddof=1):.2f}"
    )
    print(
        f"Paired HJB-PINN minus threshold P&L: {mean_delta:.2f} "
        f"(95% CI {lower_delta:.2f}, {upper_delta:.2f}); "
        f"bootstrap 95% CI ({bootstrap_lower:.2f}, {bootstrap_upper:.2f}); "
        f"SD={np.std(differences, ddof=1):.2f}; "
        f"median={np.median(differences):.2f}; "
        f"5th/95th percentiles={np.quantile(differences, 0.05):.2f}/"
        f"{np.quantile(differences, 0.95):.2f}; "
        f"win rate={np.mean(differences > 0):.2%}; "
        f"tie rate={np.mean(differences == 0):.2%}; "
        f"two-sided sign-test p="
        f"{binomtest(int(np.sum(differences > 0)), simulations, 0.5).pvalue:.4g}"
    )
    print(
        f"Mean max drawdown fraction: HJB-PINN={np.mean(model_drawdowns):.2%}, "
        f"threshold={np.mean(baseline_drawdowns):.2%}"
    )
    raw_rmse_mean, raw_rmse_lower, raw_rmse_upper = _confidence_interval(
        raw_log_price_rmse
    )
    filtered_rmse_mean, filtered_rmse_lower, filtered_rmse_upper = (
        _confidence_interval(filtered_log_price_rmse)
    )
    rmse_improvement_mean, rmse_improvement_lower, rmse_improvement_upper = (
        _confidence_interval(rmse_improvements)
    )
    print(
        f"Held-out log-price RMSE vs. latent truth: raw="
        f"{raw_rmse_mean:.6f} (95% CI {raw_rmse_lower:.6f}, "
        f"{raw_rmse_upper:.6f}); AKF="
        f"{filtered_rmse_mean:.6f} (95% CI {filtered_rmse_lower:.6f}, "
        f"{filtered_rmse_upper:.6f}); RMSE reduction="
        f"{1.0 - filtered_rmse_mean / raw_rmse_mean:.2%}; "
        f"paired absolute RMSE improvement="
        f"{rmse_improvement_mean:.7f} (95% CI "
        f"{rmse_improvement_lower:.7f}, {rmse_improvement_upper:.7f}); "
        f"AKF improves RMSE in {np.mean(rmse_improvements > 0):.2%} of runs"
    )
    print(
        f"Same fills in paired runs: {matched_fill_counts}/{simulations}; "
        f"runs with differing executed position paths: "
        f"{divergent_position_runs}/{simulations}; "
        f"runs with differing mean targets: {total_target_differences}/{simulations}"
    )
    print(
        f"Absolute paired P&L difference: median={np.median(absolute_differences):.2f}, "
        f"95th percentile={np.quantile(absolute_differences, 0.95):.2f}, "
        f"max={np.max(absolute_differences):.2f}"
    )
    print(f"Per-simulation results saved to {destination}")
    print(
        "Interpretation is limited to this fixed synthetic training model, "
        "regime-shift generator, and simplified AS fill simulator."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=1_000)
    parser.add_argument("--first-seed", type=int, default=10_000)
    parser.add_argument(
        "--no-execution-fee-awareness",
        action="store_false",
        dest="execution_fee_aware",
        help="use the pre-fee-penalty HJB controller",
    )
    parser.set_defaults(execution_fee_aware=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    arguments = parser.parse_args()
    run_monte_carlo(
        arguments.simulations,
        output_path=arguments.output,
        first_seed=arguments.first_seed,
        execution_fee_aware=arguments.execution_fee_aware,
    )


if __name__ == "__main__":
    main()
