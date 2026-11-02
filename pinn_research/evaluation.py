"""Seeded single-asset scenario evaluation for AS and reduced-HJB policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .backtest import BacktestResult, run_backtest
from .baselines import ASParameters, AvellanedaStoikov
from .hjb import InventoryHJBParameters
from .market_data import MarketData
from .model_types import FloatArray
from .pinn import HJBQuotingPolicy, InventoryPINN

ScenarioName = Literal["ordinary", "heavy_tail", "regime_change"]
PolicyName = Literal["AS", "HJB-PINN"]


@dataclass(frozen=True)
class ScenarioConfig:
    steps: int
    dt: float
    initial_price: float
    volatility: float
    drift: float = 0.0
    jump_intensity: float = 0.0
    jump_scale: float = 0.0
    jump_shape: float = 3.0
    jump_log_cap: float = 0.2
    regime_change_fraction: float = 0.5
    regime_volatility_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.steps, int) or self.steps < 2:
            raise ValueError("steps must be an integer of at least 2")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(self.initial_price) or self.initial_price <= 0:
            raise ValueError("initial_price must be finite and positive")
        if not np.isfinite(self.volatility) or self.volatility <= 0:
            raise ValueError("volatility must be finite and positive")
        if not np.isfinite(self.drift):
            raise ValueError("drift must be finite")
        if not np.isfinite(self.jump_intensity) or self.jump_intensity < 0:
            raise ValueError("jump_intensity must be finite and nonnegative")
        if not np.isfinite(self.jump_scale) or self.jump_scale < 0:
            raise ValueError("jump_scale must be finite and nonnegative")
        if not np.isfinite(self.jump_shape) or self.jump_shape <= 0:
            raise ValueError("jump_shape must be finite and positive")
        if not np.isfinite(self.jump_log_cap) or self.jump_log_cap <= 0:
            raise ValueError("jump_log_cap must be finite and positive")
        if not 0 < self.regime_change_fraction < 1:
            raise ValueError("regime_change_fraction must be between 0 and 1")
        if (
            not np.isfinite(self.regime_volatility_multiplier)
            or self.regime_volatility_multiplier <= 0
        ):
            raise ValueError("regime_volatility_multiplier must be finite and positive")


@dataclass(frozen=True)
class EvaluationRun:
    scenario: ScenarioName
    seed: int
    policy: PolicyName
    terminal_pnl: float
    fills: int
    mid_prices: FloatArray
    backtest: BacktestResult


def simulate_scenario(
    scenario: ScenarioName,
    config: ScenarioConfig,
    seed: int,
) -> MarketData:
    """Generate an arithmetic-diffusion path with optional jump/regime stress."""
    if scenario not in ("ordinary", "heavy_tail", "regime_change"):
        raise ValueError(f"unknown scenario: {scenario}")
    if scenario == "heavy_tail" and (config.jump_intensity <= 0 or config.jump_scale <= 0):
        raise ValueError("heavy_tail scenario requires positive jump intensity and scale")

    rng = np.random.default_rng(seed)
    prices = np.empty((config.steps + 1, 1), dtype=np.float64)
    prices[0, 0] = config.initial_price
    regime_step = int(config.steps * config.regime_change_fraction)
    for step in range(config.steps):
        volatility = config.volatility
        if scenario == "regime_change" and step >= regime_step:
            volatility *= config.regime_volatility_multiplier
        increment = (
            config.drift * config.dt
            + volatility * np.sqrt(config.dt) * rng.standard_normal()
        )
        if scenario == "heavy_tail":
            jump_count = rng.poisson(config.jump_intensity * config.dt)
            if jump_count:
                magnitudes = config.jump_scale * (
                    1.0 + rng.pareto(config.jump_shape, size=jump_count)
                )
                magnitudes = np.minimum(magnitudes, config.jump_log_cap)
                signs = rng.choice(np.array([-1.0, 1.0]), size=jump_count)
                log_jump = float(np.sum(signs * magnitudes))
                increment += prices[step, 0] * np.expm1(log_jump)
        prices[step + 1, 0] = prices[step, 0] + increment
        if not np.isfinite(prices[step + 1, 0]) or prices[step + 1, 0] <= 0:
            raise RuntimeError(
                f"scenario generated nonpositive price at step {step + 1}; "
                "reduce the stress parameters"
            )

    timestamps = tuple(f"{index:06d}" for index in range(config.steps + 1))
    return MarketData(timestamps, ("ASSET",), prices)


def _validate_hjb_compatibility(
    parameters: InventoryHJBParameters,
    as_parameters: ASParameters,
    config: ScenarioConfig,
) -> None:
    if as_parameters.symbols != ("ASSET",):
        raise ValueError("evaluation currently supports one asset named ASSET")
    if as_parameters.min_inventory.shape != (1,):
        raise ValueError("AS parameters must contain exactly one asset")
    if not np.isclose(parameters.horizon, config.steps * config.dt):
        raise ValueError("HJB horizon must equal scenario steps multiplied by dt")
    if not np.isclose(parameters.volatility, config.volatility):
        raise ValueError("HJB volatility must match ordinary scenario volatility")
    if not np.isclose(parameters.arrival_rate, as_parameters.arrival_rate[0]):
        raise ValueError("HJB arrival rate must match AS arrival rate")
    if not np.isclose(parameters.distance_sensitivity, as_parameters.distance_sensitivity[0]):
        raise ValueError("HJB distance sensitivity must match AS parameters")
    if not np.isclose(parameters.fee_per_unit, as_parameters.fee_per_unit[0]):
        raise ValueError("HJB fee must match AS parameters")
    if (
        parameters.min_inventory != int(as_parameters.min_inventory[0])
        or parameters.max_inventory != int(as_parameters.max_inventory[0])
    ):
        raise ValueError("HJB and AS inventory limits must match")


def evaluate_scenarios(
    pinn: InventoryPINN,
    hjb_parameters: InventoryHJBParameters,
    as_parameters: ASParameters,
    config: ScenarioConfig,
    *,
    seeds: tuple[int, ...],
    initial_cash: float,
) -> tuple[EvaluationRun, ...]:
    """Compare AS and PINN policies on paired seeded scenario paths."""
    if not seeds:
        raise ValueError("at least one random seed is required")
    if not np.isfinite(initial_cash) or initial_cash < 0:
        raise ValueError("initial_cash must be finite and nonnegative")
    _validate_hjb_compatibility(hjb_parameters, as_parameters, config)
    hjb_policy = HJBQuotingPolicy(pinn)
    as_policy = AvellanedaStoikov(as_parameters)
    policies = (("AS", as_policy), ("HJB-PINN", hjb_policy))
    results: list[EvaluationRun] = []
    for scenario in ("ordinary", "heavy_tail", "regime_change"):
        for seed in seeds:
            market = simulate_scenario(scenario, config, seed)
            for policy_name, policy in policies:
                backtest = run_backtest(
                    market,
                    policy,
                    as_parameters,
                    initial_cash,
                    config.dt,
                    seed=seed,
                )
                results.append(
                    EvaluationRun(
                        scenario=scenario,
                        seed=seed,
                        policy=policy_name,
                        terminal_pnl=float(backtest.equity[-1] - initial_cash),
                        fills=int(backtest.bid_fills.sum() + backtest.ask_fills.sum()),
                        mid_prices=market.mid_prices,
                        backtest=backtest,
                    )
                )
    return tuple(results)


def paired_terminal_pnl_differences(
    runs: tuple[EvaluationRun, ...],
) -> dict[ScenarioName, FloatArray]:
    """Return per-seed HJB-PINN minus AS terminal P&L differences."""
    differences: dict[ScenarioName, FloatArray] = {}
    for scenario in ("ordinary", "heavy_tail", "regime_change"):
        by_seed: dict[int, dict[PolicyName, float]] = {}
        for run in runs:
            if run.scenario == scenario:
                policy_results = by_seed.setdefault(run.seed, {})
                if run.policy in policy_results:
                    raise ValueError(
                        f"duplicate {run.policy} result for {scenario}, seed {run.seed}"
                    )
                policy_results[run.policy] = run.terminal_pnl
        if not by_seed:
            continue
        if any(set(values) != {"AS", "HJB-PINN"} for values in by_seed.values()):
            raise ValueError(f"scenario {scenario} must have paired results for both policies")
        differences[scenario] = np.asarray(
            [values["HJB-PINN"] - values["AS"] for _, values in sorted(by_seed.items())],
            dtype=np.float64,
        )
    return differences
