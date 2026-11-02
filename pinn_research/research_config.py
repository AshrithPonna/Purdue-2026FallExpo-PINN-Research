"""Shared default settings for reproducible synthetic research experiments."""

from __future__ import annotations

from .filters import AdaptiveKalmanFilter
from .integrated_stat_arb import IntegratedStatArbConfig
from .pair_pinn import PairPINNTrainingConfig
from .stat_arb import StatArbConfig

TRAINING_OBSERVATIONS = 750
TEST_STEPS = 500
INITIAL_CAPITAL = 100_000.0
TRAINING_SEED = 23


def default_research_configs(
    *,
    execution_seed: int = 42,
    execution_fee_aware: bool = True,
) -> tuple[StatArbConfig, IntegratedStatArbConfig]:
    """Return the common pair-selection and AKF/HJB-PINN/AS configuration."""
    strategy = StatArbConfig(
        entry_z=2.0,
        exit_z=0.5,
        cointegration_fdr=0.05,
        min_half_life=1.0,
        max_half_life=100.0,
        max_pairs=1,
        gross_leverage=1.0,
        transaction_cost_bps=1.0,
    )
    integration = IntegratedStatArbConfig(
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
        as_distance_sensitivity=100.0,
        execution_seed=execution_seed,
        execution_fee_aware=execution_fee_aware,
    )
    return strategy, integration
