"""Reduced HJB reference solver for mean-reverting pair spreads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model_types import FloatArray


@dataclass(frozen=True)
class PairHJBParameters:
    """OU spread and bounded-inventory control parameters."""

    mean_reversion_speed: float
    spread_volatility: float
    spread_scale: float
    risk_penalty: float
    trading_penalty: float
    maximum_inventory: float
    maximum_trading_rate: float
    z_limit: float
    horizon: float
    execution_fee_rate: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.mean_reversion_speed,
            self.spread_volatility,
            self.spread_scale,
            self.risk_penalty,
            self.trading_penalty,
            self.maximum_inventory,
            self.maximum_trading_rate,
            self.z_limit,
            self.horizon,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("pair HJB parameters must be finite and positive")
        if not np.isfinite(self.execution_fee_rate) or self.execution_fee_rate < 0:
            raise ValueError("execution_fee_rate must be finite and nonnegative")


@dataclass(frozen=True)
class PairHJBResult:
    """Value function over time, standardized spread, and inventory."""

    times: FloatArray
    z_grid: FloatArray
    inventory_grid: FloatArray
    value: FloatArray
    trading_rate: FloatArray
    residual: FloatArray

    def rate(self, time_to_horizon: float, zscore: float, inventory: float) -> float:
        """Interpolate the optimal bounded inventory rate on the solved grid."""
        if (
            not np.isfinite(time_to_horizon)
            or time_to_horizon < 0
            or time_to_horizon > self.times[-1]
            or not np.isfinite(zscore)
            or not np.isfinite(inventory)
        ):
            raise ValueError("policy state must be finite and within the HJB horizon")
        time = self.times[-1] - time_to_horizon
        z = float(np.clip(zscore, self.z_grid[0], self.z_grid[-1]))
        q = float(np.clip(inventory, self.inventory_grid[0], self.inventory_grid[-1]))
        at_z = np.empty((self.times.size, self.inventory_grid.size), dtype=np.float64)
        for time_index in range(self.times.size):
            for inventory_index in range(self.inventory_grid.size):
                at_z[time_index, inventory_index] = np.interp(
                    z,
                    self.z_grid,
                    self.trading_rate[time_index, :, inventory_index],
                )
        at_time = np.empty(self.inventory_grid.size, dtype=np.float64)
        for inventory_index in range(self.inventory_grid.size):
            at_time[inventory_index] = np.interp(
                time,
                self.times,
                at_z[:, inventory_index],
            )
        return float(np.interp(q, self.inventory_grid, at_time))


def _operator(
    value: FloatArray,
    z_grid: FloatArray,
    inventory_grid: FloatArray,
    parameters: PairHJBParameters,
) -> tuple[FloatArray, FloatArray]:
    drift = -parameters.mean_reversion_speed * z_grid
    value_z = np.gradient(value, z_grid, axis=0, edge_order=2)
    value_zz = np.gradient(value_z, z_grid, axis=0, edge_order=2)
    value_q = np.gradient(value, inventory_grid, axis=1, edge_order=2)
    marginal_value = np.sign(value_q) * np.maximum(
        np.abs(value_q) - parameters.execution_fee_rate,
        0.0,
    )
    rate = np.clip(
        marginal_value / parameters.trading_penalty,
        -parameters.maximum_trading_rate,
        parameters.maximum_trading_rate,
    )
    rate[:, 0] = np.maximum(rate[:, 0], 0.0)
    rate[:, -1] = np.minimum(rate[:, -1], 0.0)
    objective = (
        rate * value_q
        - 0.5 * parameters.trading_penalty * rate**2
        - parameters.execution_fee_rate * np.abs(rate)
        + inventory_grid[np.newaxis, :]
        * parameters.spread_scale
        * drift[:, np.newaxis]
        - 0.5 * parameters.risk_penalty * inventory_grid[np.newaxis, :] ** 2
    )
    return (
        drift[:, np.newaxis] * value_z
        + 0.5 * parameters.spread_volatility**2 * value_zz
        + objective,
        rate,
    )


def solve_pair_hjb(
    parameters: PairHJBParameters,
    *,
    time_steps: int = 61,
    z_points: int = 81,
    inventory_points: int = 21,
) -> PairHJBResult:
    """Solve the risk-adjusted OU-spread HJB by the method of lines.

    The value maximizes expected spread P&L less quadratic inventory risk and
    trading-rate penalties plus a linear fee calibrated to simulated turnover
    fees. It does not model AS fill probabilities, price improvement, or market
    impact.
    """
    if not isinstance(time_steps, int) or time_steps < 3:
        raise ValueError("time_steps must be an integer of at least 3")
    if not isinstance(z_points, int) or z_points < 5:
        raise ValueError("z_points must be an integer of at least 5")
    if not isinstance(inventory_points, int) or inventory_points < 5:
        raise ValueError("inventory_points must be an integer of at least 5")
    z_grid = np.linspace(-parameters.z_limit, parameters.z_limit, z_points)
    inventory_grid = np.linspace(
        -parameters.maximum_inventory,
        parameters.maximum_inventory,
        inventory_points,
    )
    terminal_value = (
        -0.5 * parameters.risk_penalty * inventory_grid**2
    )
    reverse_times = np.linspace(0.0, parameters.horizon, time_steps)

    def backward_rhs(_elapsed: float, flat_value: FloatArray) -> FloatArray:
        value = flat_value.reshape(z_points, inventory_points)
        operator, _ = _operator(value, z_grid, inventory_grid, parameters)
        return operator.ravel()

    solution = solve_ivp(
        backward_rhs,
        (0.0, parameters.horizon),
        np.broadcast_to(terminal_value, (z_points, inventory_points)).ravel().copy(),
        t_eval=reverse_times,
        method="DOP853",
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(f"pair HJB solve failed: {solution.message}")
    if not np.all(np.isfinite(solution.y)):
        raise RuntimeError("pair HJB solve produced non-finite values")

    times = parameters.horizon - reverse_times[::-1]
    value = solution.y[:, ::-1].T.reshape(
        time_steps, z_points, inventory_points
    ).copy()
    trading_rate = np.empty_like(value)
    residual = np.empty_like(value)
    for time_index, value_slice in enumerate(value):
        operator, rate = _operator(
            value_slice,
            z_grid,
            inventory_grid,
            parameters,
        )
        trading_rate[time_index] = rate
        residual[time_index] = operator
    residual += np.gradient(value, times, axis=0, edge_order=2)
    for array in (times, z_grid, inventory_grid, value, trading_rate, residual):
        array.setflags(write=False)
    return PairHJBResult(
        times=times,
        z_grid=z_grid,
        inventory_grid=inventory_grid,
        value=value,
        trading_rate=trading_rate,
        residual=residual,
    )
