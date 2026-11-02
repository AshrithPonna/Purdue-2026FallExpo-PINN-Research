"""Run the seeded synthetic AS versus HJB-PINN scenario evaluation."""

from __future__ import annotations

import numpy as np

from .baselines import ASParameters
from .evaluation import ScenarioConfig, evaluate_scenarios, paired_terminal_pnl_differences
from .hjb import InventoryHJBParameters, solve_inventory_hjb
from .metrics import market_making_metrics
from .pinn import PINNTrainingConfig, train_inventory_pinn


def main() -> None:
    config = ScenarioConfig(
        steps=252,
        dt=1 / 252,
        initial_price=100.0,
        volatility=0.2,
        jump_intensity=2.0,
        jump_scale=0.02,
        jump_shape=3.0,
        jump_log_cap=0.2,
        regime_change_fraction=0.5,
        regime_volatility_multiplier=2.0,
    )
    hjb_parameters = InventoryHJBParameters(
        risk_aversion=0.1,
        volatility=config.volatility,
        arrival_rate=5.0,
        distance_sensitivity=1.5,
        min_inventory=-5,
        max_inventory=5,
        fee_per_unit=0.001,
        horizon=config.steps * config.dt,
        max_quote_distance=2.0,
    )
    as_parameters = ASParameters(
        symbols=("ASSET",),
        risk_aversion=hjb_parameters.risk_aversion,
        volatility=np.array([hjb_parameters.volatility]),
        arrival_rate=np.array([hjb_parameters.arrival_rate]),
        distance_sensitivity=np.array([hjb_parameters.distance_sensitivity]),
        min_inventory=np.array([hjb_parameters.min_inventory]),
        max_inventory=np.array([hjb_parameters.max_inventory]),
        fee_per_unit=np.array([hjb_parameters.fee_per_unit]),
    )
    reference = solve_inventory_hjb(hjb_parameters, time_steps=501)
    pinn = train_inventory_pinn(
        hjb_parameters,
        PINNTrainingConfig(
            iterations=2_000,
            collocation_times=64,
            hidden_width=48,
            hidden_layers=2,
            learning_rate=1e-3,
            terminal_weight=10.0,
            seed=19,
        ),
    )
    print("PINN reference comparison:", pinn.compare(reference))
    runs = evaluate_scenarios(
        pinn,
        hjb_parameters,
        as_parameters,
        config,
        seeds=(11, 23, 37, 51, 79),
        initial_cash=10_000.0,
    )
    differences = paired_terminal_pnl_differences(runs)
    for scenario, policy in (
        ("ordinary", "AS"),
        ("ordinary", "HJB-PINN"),
        ("heavy_tail", "AS"),
        ("heavy_tail", "HJB-PINN"),
        ("regime_change", "AS"),
        ("regime_change", "HJB-PINN"),
    ):
        selected = [
            run for run in runs
            if run.scenario == scenario and run.policy == policy
        ]
        pnl = np.array([run.terminal_pnl for run in selected], dtype=np.float64)
        metrics = [
            market_making_metrics(
                run.backtest,
                run.mid_prices,
                as_parameters.fee_per_unit,
            )
            for run in selected
        ]
        print(
            f"{scenario:13s} {policy:8s} "
            f"mean_pnl={pnl.mean(): .6f} "
            f"std_pnl={pnl.std(ddof=1): .6f} "
            f"mean_max_drawdown={np.mean([m.maximum_drawdown for m in metrics]): .6f} "
            f"mean_drawdown_fraction={np.mean([m.maximum_drawdown_fraction for m in metrics]): .6f} "
            f"mean_inventory_variance={np.mean([np.mean(m.inventory_variance_by_asset) for m in metrics]): .6f} "
            f"mean_spread_capture={np.mean([m.spread_capture_after_fees for m in metrics]): .6f} "
            f"mean_quote_opportunities_per_trade={np.mean([m.quote_opportunity_to_trade_ratio for m in metrics if m.quote_opportunity_to_trade_ratio is not None]) if any(m.quote_opportunity_to_trade_ratio is not None for m in metrics) else float('nan'): .2f}"
        )
    for scenario, paired in differences.items():
        print(
            f"{scenario:13s} paired(HJB-PINN - AS) "
            f"mean={paired.mean(): .6f} "
            f"std={paired.std(ddof=1): .6f}"
        )


if __name__ == "__main__":
    main()
