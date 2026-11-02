"""Causal AKF and pair-level HJB-PINN statistical-arbitrage pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from filters import AdaptiveKalmanFilter, FilterResult
from market_data import MarketData
from model_types import FloatArray
from pair_hjb import PairHJBParameters, PairHJBResult, solve_pair_hjb
from pair_pinn import (
    PairInventoryPINN,
    PairPINNComparison,
    PairPINNTrainingConfig,
    train_pair_pinn,
)
from stat_arb import (
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


def _pair_hjb_parameters(
    pair: PairCalibration,
    training_data: MarketData,
    horizon: float,
    config: IntegratedStatArbConfig,
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
        parameters = _pair_hjb_parameters(
            pair,
            training_data,
            float(test_data.steps),
            integration_config,
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

    result = _backtest_hjb_pairs(
        model,
        test_data,
        pinns,
        initial_capital=initial_capital,
        maximum_inventory=integration_config.maximum_inventory,
    )
    return IntegratedStatArbRun(
        model=model,
        result=result,
        filter_result=filter_result,
        pair_hjb_references=tuple(references),
        pair_pinns=tuple(pinns),
        pinn_comparisons=tuple(comparisons),
    )


def _backtest_hjb_pairs(
    model: StatArbModel,
    test_data: MarketData,
    policies: list[PairInventoryPINN],
    *,
    initial_capital: float,
    maximum_inventory: float,
) -> StatArbResult:
    if test_data.symbols != model.symbols:
        raise ValueError("test symbols must match training symbols in order")
    if len(policies) != len(model.pairs):
        raise ValueError("one HJB-PINN policy is required per selected pair")
    symbol_indices = {symbol: index for index, symbol in enumerate(test_data.symbols)}
    pair_count = len(model.pairs)
    time_count, asset_count = test_data.mid_prices.shape
    equity = np.empty(time_count, dtype=np.float64)
    cash_history = np.empty(time_count, dtype=np.float64)
    holdings_history = np.zeros((time_count, asset_count), dtype=np.float64)
    gross_exposure = np.zeros(time_count, dtype=np.float64)
    zscores = np.empty((time_count, pair_count), dtype=np.float64)
    pair_positions = np.zeros((time_count, pair_count), dtype=np.float64)
    turnover = np.zeros(time_count, dtype=np.float64)
    transaction_costs = np.zeros(time_count, dtype=np.float64)
    pair_holdings = np.zeros((asset_count, pair_count), dtype=np.float64)
    holdings = np.zeros(asset_count, dtype=np.float64)
    cash = float(initial_capital)
    pair_budget = initial_capital * model.config.gross_leverage / pair_count
    fee_rate = model.config.transaction_cost_bps / 10_000.0
    positions = np.zeros(pair_count, dtype=np.float64)
    horizon = float(test_data.steps)

    for time_index, prices in enumerate(test_data.mid_prices):
        for pair_index, (pair, policy) in enumerate(zip(model.pairs, policies)):
            y_index = symbol_indices[pair.asset_y]
            x_index = symbol_indices[pair.asset_x]
            spread = (
                np.log(prices[y_index])
                - pair.intercept
                - pair.hedge_ratio * np.log(prices[x_index])
            )
            zscore = (spread - pair.spread_mean) / pair.spread_std
            zscores[time_index, pair_index] = zscore
            if time_index == test_data.steps:
                position = 0.0
            else:
                time_to_horizon = horizon - time_index
                rate = policy.trading_rate(
                    time_to_horizon,
                    zscore,
                    positions[pair_index],
                )
                position = float(
                    np.clip(
                        positions[pair_index] + rate,
                        -maximum_inventory,
                        maximum_inventory,
                    )
                )
            positions[pair_index] = position
            pair_positions[time_index, pair_index] = position
            y_notional = pair_budget / (1.0 + abs(pair.hedge_ratio))
            pair_holdings[:, pair_index] = 0.0
            pair_holdings[y_index, pair_index] = (
                position * y_notional / prices[y_index]
            )
            pair_holdings[x_index, pair_index] = (
                -position * pair.hedge_ratio * y_notional / prices[x_index]
            )

        target_holdings = pair_holdings.sum(axis=1)
        trade_delta = target_holdings - holdings
        step_turnover = float(np.sum(np.abs(trade_delta * prices)))
        step_cost = fee_rate * step_turnover
        cash -= float(trade_delta @ prices) + step_cost
        holdings = target_holdings
        equity[time_index] = cash + float(holdings @ prices)
        gross_exposure[time_index] = float(np.sum(np.abs(holdings * prices)))
        cash_history[time_index] = cash
        holdings_history[time_index] = holdings
        turnover[time_index] = step_turnover
        transaction_costs[time_index] = step_cost

    for array in (
        equity,
        cash_history,
        holdings_history,
        gross_exposure,
        zscores,
        pair_positions,
        turnover,
        transaction_costs,
    ):
        array.setflags(write=False)
    return StatArbResult(
        timestamps=test_data.timestamps,
        symbols=test_data.symbols,
        selected_pairs=model.pairs,
        initial_capital=float(initial_capital),
        equity=equity,
        cash=cash_history,
        holdings=holdings_history,
        gross_exposure=gross_exposure,
        zscores=zscores,
        pair_positions=pair_positions,
        turnover=turnover,
        transaction_costs=transaction_costs,
    )
