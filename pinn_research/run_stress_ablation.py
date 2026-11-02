"""Matched stress scenarios and controller ablations for pair statistical arbitrage."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from math import ceil
from pathlib import Path

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
from .run_integrated_stat_arb import _ThresholdPairPolicy, _synthetic_observations
from .run_monte_carlo import _continued_paths, _confidence_interval


SCENARIOS = ("volatility_shift", "heavy_tail", "cointegration_break")
VARIANTS = (
    "threshold_no_variance_derisk",
    "threshold_akf_variance_derisk",
    "hjb_no_fee_akf_variance_derisk",
    "hjb_fee_aware_akf_variance_derisk",
)
VARIANT_LABELS = {
    "threshold_no_variance_derisk": "Threshold, no AKF de-risking",
    "threshold_akf_variance_derisk": "Threshold + AKF de-risking",
    "hjb_no_fee_akf_variance_derisk": "HJB-PINN, no fee term + AKF",
    "hjb_fee_aware_akf_variance_derisk": "HJB-PINN, fee-aware + AKF",
}
SCENARIO_LABELS = {
    "volatility_shift": "Volatility shift",
    "heavy_tail": "Heavy-tail",
    "cointegration_break": "Cointegration break",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _lower_tail_mean(values: np.ndarray, probability: float = 0.05) -> float:
    count = max(1, ceil(values.size * probability))
    return float(np.mean(np.sort(values)[:count]))


def _column_values(
    rows: list[dict[str, float | int | str]],
    column: str,
) -> np.ndarray:
    return np.asarray(
        [float(row[column]) for row in rows],
        dtype=np.float64,
    )


def _column_mean(
    rows: list[dict[str, float | int | str]],
    column: str,
) -> float:
    values = _column_values(rows, column)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _paired_tail_difference_interval(
    improved: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 2_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sample = rng.integers(0, improved.size, improved.size)
        differences[index] = _lower_tail_mean(
            improved[sample]
        ) - _lower_tail_mean(reference[sample])
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return float(lower), float(upper)


def _latex_table(rows: list[dict[str, float | int | str]]) -> str:
    """Format the abstract's synthetic diagnostics as a paste-ready LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Synthetic stress-test metrics; these are not live-market estimates.}",
        r"\label{tab:synthetic-abstract-metrics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\hline",
        (
            r"Scenario & Controller & Mean P\&L & Lower-tail 5\% P\&L & "
            r"Inventory variance & Spread capture & Quote/fill proxy & "
            r"Mean drawdown & 95th-pctl drawdown & Drawdown duration & "
            r"HJB residual \\"
        ),
        r"\hline",
    ]
    for row in rows:
        pnl = f"${float(row['mean_net_pnl']):,.0f}$"
        tail_pnl = f"${float(row['lower_tail_5pct_mean_pnl']):,.0f}$"
        inventory_variance = f"{float(row['mean_normalized_pair_inventory_variance']):.4f}"
        spread_capture = (
            f"{100 * float(row['mean_spread_capture_efficiency']):.3f}\\%"
        )
        quote_fill = (
            f"{float(row['mean_quote_opportunities_per_fill_proxy']):.2f}"
        )
        mean_drawdown = (
            f"{100 * float(row['mean_max_drawdown_fraction']):.2f}\\%"
        )
        p95_drawdown = (
            f"{100 * float(row['p95_max_drawdown_fraction']):.2f}\\%"
        )
        duration = f"{float(row['mean_max_drawdown_duration_steps']):.1f}"
        residual = row["hjb_mean_absolute_pde_residual"]
        residual_text = f"{float(residual):.5f}" if residual != "" else "---"
        lines.append(
            " & ".join(
                (
                    SCENARIO_LABELS[str(row["scenario"])],
                    VARIANT_LABELS[str(row["variant"])],
                    pnl,
                    tail_pnl,
                    inventory_variance,
                    spread_capture,
                    quote_fill,
                    mean_drawdown,
                    p95_drawdown,
                    duration,
                    residual_text,
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\par\smallskip",
            (
                r"\footnotesize Inventory variance is for normalized pair "
                r"positions. Spread capture and quote/fill are simulator "
                r"proxies; quote/fill is not an exchange order-to-trade ratio. "
                r"HJB residual is a training-fit PDE diagnostic and is not "
                r"applicable to threshold controllers. Results use one "
                r"synthetic training fit."
            ),
            r"\end{table}",
        )
    )
    return "\n".join(lines) + "\n"


def run_stress_ablation(
    simulations_per_scenario: int = 250,
    *,
    first_seed: int = 50_000,
    output_path: str | Path = PROJECT_ROOT / "results" / "stress_ablation_results.csv",
    summary_path: str | Path = PROJECT_ROOT / "results" / "stress_ablation_summary.csv",
    latex_path: str | Path = PROJECT_ROOT / "paper" / "abstract_metrics.tex",
) -> None:
    if (
        not isinstance(simulations_per_scenario, int)
        or simulations_per_scenario < 20
    ):
        raise ValueError("simulations_per_scenario must be an integer of at least 20")
    if not isinstance(first_seed, int) or first_seed < 0:
        raise ValueError("first_seed must be a nonnegative integer")

    training_count = TRAINING_OBSERVATIONS
    test_steps = TEST_STEPS
    initial_capital = INITIAL_CAPITAL
    strategy_config, base_config = default_research_configs(
        execution_seed=first_seed
    )
    training_market = _synthetic_observations(
        seed=TRAINING_SEED,
        count=training_count + test_steps,
        regime_change=training_count,
    )
    fee_aware_fit = run_integrated_stat_arb(
        training_market,
        training_observations=training_count,
        initial_capital=initial_capital,
        strategy_config=strategy_config,
        integration_config=base_config,
    )
    no_fee_config = replace(base_config, execution_fee_aware=False)
    no_fee_fit = run_integrated_stat_arb(
        training_market,
        training_observations=training_count,
        initial_capital=initial_capital,
        strategy_config=strategy_config,
        integration_config=no_fee_config,
    )
    if fee_aware_fit.model.pairs != no_fee_fit.model.pairs:
        raise RuntimeError("fee ablation changed the training-selected pair")
    if len(fee_aware_fit.model.pairs) != 1:
        raise RuntimeError("stress experiment expects exactly one fitted pair")

    pair = fee_aware_fit.model.pairs[0]
    threshold_policy = _ThresholdPairPolicy(
        strategy_config.entry_z,
        strategy_config.exit_z,
        base_config.maximum_trading_rate,
        base_config.maximum_inventory,
    )
    training_process_variances = fee_aware_fit.filter_result.process_variances[
        :training_count
    ]
    variance_baseline = np.median(training_process_variances, axis=0)
    if np.any(variance_baseline <= 0):
        raise RuntimeError("training AKF variance baseline must be positive")

    records: list[dict[str, float | int | str]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for simulation_index in range(simulations_per_scenario):
            seed = (
                first_seed
                + scenario_index * simulations_per_scenario
                + simulation_index
            )
            regime_change = int(
                np.random.default_rng(seed + 50_000_000).integers(150, 351)
            )
            prices, _ = _continued_paths(
                training_market.mid_prices[:training_count],
                pair_intercept=pair.intercept,
                hedge_ratio=pair.hedge_ratio,
                spread_mean=pair.spread_mean,
                seed=seed,
                test_steps=test_steps,
                regime_change=regime_change,
                stress_scenario=scenario,
            )
            timestamps = tuple(f"{index:05d}" for index in range(prices.shape[0]))
            filter_result = base_config.filter.filter(np.log(prices))
            filtered_test = MarketData(
                timestamps[training_count - 1 :],
                training_market.symbols,
                filter_result.estimated_prices[training_count - 1 :],
            )
            test_variances = filter_result.process_variances[
                training_count - 1 :
            ]
            no_derisk_variances = np.broadcast_to(
                variance_baseline,
                test_variances.shape,
            )
            common_execution = {
                "training_process_variances": training_process_variances,
                "initial_capital": initial_capital,
                "maximum_inventory": base_config.maximum_inventory,
                "dt": base_config.execution_dt,
                "risk_aversion": base_config.as_risk_aversion,
                "arrival_rate": base_config.as_arrival_rate,
                "distance_sensitivity": base_config.as_distance_sensitivity,
                "maximum_inventory_slots": base_config.maximum_inventory_slots,
                "regime_threshold": base_config.volatility_regime_threshold,
                "regime_multiplier_cap": base_config.volatility_regime_cap,
                "seed": seed,
            }
            variant_inputs = (
                (
                    "threshold_no_variance_derisk",
                    (threshold_policy,),
                    no_derisk_variances,
                ),
                (
                    "threshold_akf_variance_derisk",
                    (threshold_policy,),
                    test_variances,
                ),
                (
                    "hjb_no_fee_akf_variance_derisk",
                    no_fee_fit.pair_pinns,
                    test_variances,
                ),
                (
                    "hjb_fee_aware_akf_variance_derisk",
                    fee_aware_fit.pair_pinns,
                    test_variances,
                ),
            )
            row: dict[str, float | int | str] = {
                "scenario": scenario,
                "simulation": simulation_index + 1,
                "seed": seed,
                "regime_change_step": regime_change,
            }
            for variant, policies, variances in variant_inputs:
                execution = execute_pair_targets_with_as(
                    fee_aware_fit.model,
                    filtered_test,
                    policies,
                    variances,
                    **common_execution,
                )
                metrics = statistical_arbitrage_metrics(execution.portfolio)
                normalized_inventory_variance = float(
                    np.mean(
                        np.var(
                            execution.portfolio.pair_positions,
                            axis=0,
                            ddof=0,
                        )
                    )
                )
                spread_capture_efficiency = (
                    execution.spread_capture_after_fees
                    / metrics.total_turnover
                    if metrics.total_turnover > 0.0
                    else 0.0
                )
                quote_to_fill_proxy = (
                    execution.order_count / execution.fill_count
                    if execution.fill_count > 0
                    else float("nan")
                )
                if variant == "hjb_no_fee_akf_variance_derisk":
                    hjb_residual = (
                        no_fee_fit.pinn_comparisons[0].mean_absolute_residual
                    )
                elif variant == "hjb_fee_aware_akf_variance_derisk":
                    hjb_residual = (
                        fee_aware_fit.pinn_comparisons[0].mean_absolute_residual
                    )
                else:
                    hjb_residual = float("nan")
                values = {
                    "pnl": metrics.net_pnl,
                    "drawdown": metrics.maximum_drawdown_fraction,
                    "drawdown_duration": float(
                        metrics.maximum_drawdown_duration_steps
                    ),
                    "inventory_variance": normalized_inventory_variance,
                    "spread_capture_efficiency": spread_capture_efficiency,
                    "quote_to_fill_proxy": quote_to_fill_proxy,
                    "hjb_mean_absolute_residual": hjb_residual,
                }
                for metric, value in values.items():
                    row[f"{variant}_{metric}"] = value
            records.append(row)
        print(
            f"completed {scenario}: "
            f"{simulation_index + 1}/{simulations_per_scenario}"
        )

    destination = Path(output_path)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_records: list[dict[str, float | int | str]] = []
    for scenario in SCENARIOS:
        scenario_rows = [row for row in records if row["scenario"] == scenario]
        for variant in VARIANTS:
            pnl = np.asarray(
                [float(row[f"{variant}_pnl"]) for row in scenario_rows]
            )
            drawdown = np.asarray(
                [float(row[f"{variant}_drawdown"]) for row in scenario_rows]
            )
            summary_records.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "simulations": simulations_per_scenario,
                    "mean_net_pnl": float(np.mean(pnl)),
                    "lower_tail_5pct_mean_pnl": _lower_tail_mean(pnl),
                    "mean_normalized_pair_inventory_variance": _column_mean(
                        scenario_rows,
                        f"{variant}_inventory_variance",
                    ),
                    "mean_spread_capture_efficiency": _column_mean(
                        scenario_rows,
                        f"{variant}_spread_capture_efficiency",
                    ),
                    "mean_quote_opportunities_per_fill_proxy": _column_mean(
                        scenario_rows,
                        f"{variant}_quote_to_fill_proxy",
                    ),
                    "mean_max_drawdown_fraction": float(np.mean(drawdown)),
                    "p95_max_drawdown_fraction": float(
                        np.quantile(drawdown, 0.95)
                    ),
                    "mean_max_drawdown_duration_steps": _column_mean(
                        scenario_rows,
                        f"{variant}_drawdown_duration",
                    ),
                    "hjb_mean_absolute_pde_residual": (
                        _column_mean(
                            scenario_rows,
                            f"{variant}_hjb_mean_absolute_residual",
                        )
                        if variant.startswith("hjb_")
                        else ""
                    ),
                }
            )
    summary_destination = Path(summary_path)
    with summary_destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_records[0]),
        )
        writer.writeheader()
        writer.writerows(summary_records)
    latex_destination = Path(latex_path)
    latex_destination.write_text(
        _latex_table(summary_records),
        encoding="utf-8",
    )

    print(f"\nMatched stress-ablation results (N={simulations_per_scenario} per regime)")
    for scenario_index, scenario in enumerate(SCENARIOS):
        print(f"\n{scenario}:")
        scenario_rows = [row for row in records if row["scenario"] == scenario]
        for variant in VARIANTS:
            pnl = _column_values(scenario_rows, f"{variant}_pnl")
            drawdown = _column_values(scenario_rows, f"{variant}_drawdown")
            duration = _column_values(
                scenario_rows,
                f"{variant}_drawdown_duration",
            )
            print(
                f"  {variant}: mean P&L={np.mean(pnl):.2f}; "
                f"lower-tail 5% mean P&L={_lower_tail_mean(pnl):.2f}; "
                f"mean max drawdown={np.mean(drawdown):.2%}; "
                f"95th-percentile max drawdown={np.quantile(drawdown, 0.95):.2%}; "
                f"mean drawdown duration={np.mean(duration):.1f} steps"
            )
        comparisons = (
            (
                "AKF variance de-risking vs no de-risking (threshold)",
                "threshold_akf_variance_derisk",
                "threshold_no_variance_derisk",
            ),
            (
                "HJB fee-aware vs HJB without fee penalty",
                "hjb_fee_aware_akf_variance_derisk",
                "hjb_no_fee_akf_variance_derisk",
            ),
        )
        for comparison_index, (label, improved, reference) in enumerate(comparisons):
            improved_pnl = _column_values(scenario_rows, f"{improved}_pnl")
            reference_pnl = _column_values(scenario_rows, f"{reference}_pnl")
            tail_difference = _lower_tail_mean(improved_pnl) - _lower_tail_mean(
                reference_pnl
            )
            tail_lower, tail_upper = _paired_tail_difference_interval(
                improved_pnl,
                reference_pnl,
                seed=first_seed + scenario_index * 10_000 + comparison_index,
            )
            drawdown_difference = (
                _column_values(scenario_rows, f"{improved}_drawdown")
                - _column_values(scenario_rows, f"{reference}_drawdown")
            )
            mean_dd_diff, dd_lower, dd_upper = _confidence_interval(
                drawdown_difference
            )
            print(
                f"  {label}: lower-tail P&L difference={tail_difference:.2f}; "
                f"paired-bootstrap 95% CI "
                f"[{tail_lower:.2f}, {tail_upper:.2f}]; "
                f"paired mean drawdown difference={mean_dd_diff:.2%} "
                f"(95% CI {dd_lower:.2%}, {dd_upper:.2%})"
            )
    print(f"\nPer-simulation data saved to {destination}")
    print(f"Abstract-metric summary saved to {summary_destination}")
    print(f"LaTeX table saved to {latex_destination}")
    print(
        "All results are conditional on one synthetic training fit and a "
        "simplified AS simulator; they are stress diagnostics, not trading claims."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations-per-scenario", type=int, default=250)
    parser.add_argument("--first-seed", type=int, default=50_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "stress_ablation_results.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "stress_ablation_summary.csv",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=PROJECT_ROOT / "paper" / "abstract_metrics.tex",
    )
    arguments = parser.parse_args()
    run_stress_ablation(
        arguments.simulations_per_scenario,
        first_seed=arguments.first_seed,
        output_path=arguments.output,
        summary_path=arguments.summary_output,
        latex_path=arguments.latex_output,
    )


if __name__ == "__main__":
    main()
