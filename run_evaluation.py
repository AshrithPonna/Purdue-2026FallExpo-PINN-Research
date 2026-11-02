"""Run the seeded synthetic AS versus HJB-PINN scenario evaluation."""

from __future__ import annotations

import numpy as np

from baselines import ASParameters
from evaluation import ScenarioConfig, evaluate_scenarios, paired_terminal_pnl_differences
from hjb import InventoryHJBParameters, solve_inventory_hjb
from pinn import PINNTrainingConfig, train_inventory_pinn


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
        fills = np.array([run.fills for run in selected], dtype=np.float64)
        print(
            f"{scenario:13s} {policy:8s} "
            f"mean_pnl={pnl.mean(): .6f} "
            f"std_pnl={pnl.std(ddof=1): .6f} "
            f"mean_fills={fills.mean(): .2f}"
        )
    for scenario, paired in differences.items():
        print(
            f"{scenario:13s} paired(HJB-PINN - AS) "
            f"mean={paired.mean(): .6f} "
            f"std={paired.std(ddof=1): .6f}"
        )


if __name__ == "__main__":
    main()
