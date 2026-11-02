"""Causal AKF and pair-level HJB-PINN statistical-arbitrage pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .combined_execution import ASPairExecutionResult, execute_pair_targets_with_as
from .filters import AdaptiveKalmanFilter, FilterResult
from .market_data import MarketData
from .pair_hjb import PairHJBParameters, PairHJBResult, solve_pair_hjb
from .pair_pinn import (
    PairInventoryPINN,
    PairPINNComparison,
    PairPINNTrainingConfig,
    train_pair_pinn,
)
from .stat_arb import (
    PairCalibration,
    StatArbConfig,
    StatArbModel,
    StatArbResult,
    fit_cointegrated_pairs,
)


@dataclass(frozen=True)
class IntegratedStatArbConfig:
    """Fixed estimator and reduced-control parameters for an experiment."""

    filter: AdaptiveKalmanFilter = field(
        default_factory=lambda: AdaptiveKalmanFilter(
            initial_process_variance=1e-4,
            measurement_variance=1e-4,
            initial_state_variance=1e-3,
            adaptation_rate=0.05,
        )
    )
    pinn: PairPINNTrainingConfig = field(default_factory=PairPINNTrainingConfig)
    risk_penalty: float = 0.01
    trading_penalty: float = 0.05
    maximum_inventory: float = 1.0
    maximum_trading_rate: float = 0.25
    z_limit: float = 8.0
    hjb_time_steps: int = 51
    hjb_z_points: int = 41
    hjb_inventory_points: int = 11
    execution_dt: float = 1.0
    as_risk_aversion: float = 0.1
    as_arrival_rate: float = 1.0
    as_distance_sensitivity: float = 100.0
    execution_fee_aware: bool = True
    maximum_inventory_slots: int = 10
    volatility_regime_threshold: float = 1.5
    volatility_regime_cap: float = 3.0
    execution_seed: int = 42

    def __post_init__(self) -> None:
        positive = (
            self.risk_penalty,
            self.trading_penalty,
            self.maximum_inventory,
            self.maximum_trading_rate,
            self.z_limit,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("integrated-control parameters must be finite and positive")
        execution_values = (
            self.execution_dt,
            self.as_risk_aversion,
            self.as_arrival_rate,
            self.as_distance_sensitivity,
            self.volatility_regime_threshold,
            self.volatility_regime_cap,
        )
        if any(not np.isfinite(value) or value <= 0 for value in execution_values):
            raise ValueError("AS execution parameters must be finite and positive")
        if (
            not isinstance(self.maximum_inventory_slots, int)
            or self.maximum_inventory_slots < 1
        ):
            raise ValueError("maximum_inventory_slots must be a positive integer")
        if not isinstance(self.execution_fee_aware, bool):
            raise ValueError("execution_fee_aware must be a boolean")
        if self.volatility_regime_threshold <= 1:
            raise ValueError("volatility_regime_threshold must exceed one")
        if self.volatility_regime_cap < self.volatility_regime_threshold:
            raise ValueError("volatility_regime_cap must not be below its threshold")
        if not isinstance(self.execution_seed, int) or self.execution_seed < 0:
            raise ValueError("execution_seed must be a nonnegative integer")
        for value, name, minimum in (
            (self.hjb_time_steps, "hjb_time_steps", 3),
            (self.hjb_z_points, "hjb_z_points", 5),
            (self.hjb_inventory_points, "hjb_inventory_points", 5),
        ):
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")


@dataclass(frozen=True)
class IntegratedStatArbRun:
    """Artifacts and held-out results from one integrated experiment."""

    model: StatArbModel
    result: StatArbResult
    filter_result: FilterResult
    pair_hjb_references: tuple[PairHJBResult, ...]
    pair_pinns: tuple[PairInventoryPINN, ...]
    pinn_comparisons: tuple[PairPINNComparison, ...]
    execution: ASPairExecutionResult


def _pair_hjb_parameters(
    pair: PairCalibration,
    training_data: MarketData,
    horizon: float,
    config: IntegratedStatArbConfig,
    execution_fee_rate: float,
) -> PairHJBParameters:
    symbol_indices = {symbol: index for index, symbol in enumerate(training_data.symbols)}
    log_prices = np.log(training_data.mid_prices)
    spread = (
        log_prices[:, symbol_indices[pair.asset_y]]
        - pair.intercept
        - pair.hedge_ratio * log_prices[:, symbol_indices[pair.asset_x]]
    )
    zscores = (spread - pair.spread_mean) / pair.spread_std
    lagged = zscores[:-1]
    changes = np.diff(zscores)
    denominator = float(lagged @ lagged)
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError(f"degenerate spread dynamics for {pair.asset_y}/{pair.asset_x}")
    persistence = float(lagged @ zscores[1:] / denominator)
    persistence = float(np.clip(persistence, 1e-6, 1.0 - 1e-6))
    mean_reversion_speed = -float(np.log(persistence))
    innovations = changes - (persistence - 1.0) * lagged
    spread_volatility = float(np.std(innovations, ddof=1))
    if not np.isfinite(spread_volatility) or spread_volatility <= 0:
        raise ValueError(f"invalid spread volatility for {pair.asset_y}/{pair.asset_x}")
    return PairHJBParameters(
        mean_reversion_speed=mean_reversion_speed,
        spread_volatility=spread_volatility,
        spread_scale=pair.spread_std,
        risk_penalty=config.risk_penalty,
        trading_penalty=config.trading_penalty,
        execution_fee_rate=execution_fee_rate,
        maximum_inventory=config.maximum_inventory,
        maximum_trading_rate=config.maximum_trading_rate,
        z_limit=config.z_limit,
        horizon=horizon,
    )


def run_integrated_stat_arb(
    observed_data: MarketData,
    *,
    training_observations: int,
    initial_capital: float,
    strategy_config: StatArbConfig = StatArbConfig(),
    integration_config: IntegratedStatArbConfig = IntegratedStatArbConfig(),
) -> IntegratedStatArbRun:
    """Fit on filtered training data and trade causal filtered held-out prices."""
    if (
        not isinstance(training_observations, int)
        or training_observations < 101
        or training_observations >= len(observed_data.timestamps)
    ):
        raise ValueError(
            "training_observations must be at least 101 and leave held-out observations"
        )
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive")

    filter_result = integration_config.filter.filter(np.log(observed_data.mid_prices))
    filtered_data = MarketData(
        observed_data.timestamps,
        observed_data.symbols,
        filter_result.estimated_prices,
    )
    training_data = MarketData(
        filtered_data.timestamps[:training_observations],
        filtered_data.symbols,
        filtered_data.mid_prices[:training_observations],
    )
    test_data = MarketData(
        filtered_data.timestamps[training_observations - 1 :],
        filtered_data.symbols,
        filtered_data.mid_prices[training_observations - 1 :],
    )
    model = fit_cointegrated_pairs(training_data, strategy_config)
    references: list[PairHJBResult] = []
    pinns: list[PairInventoryPINN] = []
    comparisons: list[PairPINNComparison] = []
    for pair_index, pair in enumerate(model.pairs):
        pair_fee_rate = 0.0
        if integration_config.execution_fee_aware:
            pair_fee_rate = (
                strategy_config.transaction_cost_bps
                / 10_000.0
                * strategy_config.gross_leverage
                / len(model.pairs)
                / integration_config.maximum_inventory
            )
        parameters = _pair_hjb_parameters(
            pair,
            training_data,
            float(test_data.steps * integration_config.execution_dt),
            integration_config,
            pair_fee_rate,
        )
        reference = solve_pair_hjb(
            parameters,
            time_steps=integration_config.hjb_time_steps,
            z_points=integration_config.hjb_z_points,
            inventory_points=integration_config.hjb_inventory_points,
        )
        pin_config = PairPINNTrainingConfig(
            iterations=integration_config.pinn.iterations,
            collocation_points=integration_config.pinn.collocation_points,
            data_batch_size=integration_config.pinn.data_batch_size,
            hidden_width=integration_config.pinn.hidden_width,
            hidden_layers=integration_config.pinn.hidden_layers,
            learning_rate=integration_config.pinn.learning_rate,
            data_weight=integration_config.pinn.data_weight,
            terminal_weight=integration_config.pinn.terminal_weight,
            seed=integration_config.pinn.seed + pair_index,
        )
        pinn = train_pair_pinn(parameters, pin_config, reference=reference)
        references.append(reference)
        pinns.append(pinn)
        comparisons.append(pinn.compare(reference))

    execution = execute_pair_targets_with_as(
        model,
        test_data,
        tuple(pinns),
        filter_result.process_variances[
            training_observations - 1 :
        ],
        training_process_variances=filter_result.process_variances[
            :training_observations
        ],
        initial_capital=initial_capital,
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
    return IntegratedStatArbRun(
        model=model,
        result=execution.portfolio,
        filter_result=filter_result,
        pair_hjb_references=tuple(references),
        pair_pinns=tuple(pinns),
        pinn_comparisons=tuple(comparisons),
        execution=execution,
    )
