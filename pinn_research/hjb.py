"""Low-dimensional finite-inventory HJB reference solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model_types import FloatArray, IntArray


@dataclass(frozen=True)
class InventoryHJBParameters:
    """Single-asset CARA market-making parameters for the reference HJB."""

    risk_aversion: float
    volatility: float
    arrival_rate: float
    distance_sensitivity: float
    min_inventory: int
    max_inventory: int
    fee_per_unit: float
    horizon: float
    max_quote_distance: float

    def __post_init__(self) -> None:
        positive = (
            self.risk_aversion,
            self.arrival_rate,
            self.distance_sensitivity,
            self.horizon,
            self.max_quote_distance,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("risk, arrival, sensitivity, horizon and quote distance must be positive")
        if not np.isfinite(self.volatility) or self.volatility < 0:
            raise ValueError("volatility must be finite and nonnegative")
        if not np.isfinite(self.fee_per_unit) or self.fee_per_unit < 0:
            raise ValueError("fee_per_unit must be finite and nonnegative")
        if (
            not isinstance(self.min_inventory, (int, np.integer))
            or not isinstance(self.max_inventory, (int, np.integer))
        ):
            raise ValueError("inventory limits must be integers")
        if self.min_inventory > self.max_inventory:
            raise ValueError("minimum inventory cannot exceed maximum inventory")


@dataclass(frozen=True)
class InventoryHJBResult:
    """Value factors and optimal offsets on the finite time/inventory grid."""

    times: FloatArray
    inventory: IntArray
    value_factor: FloatArray
    bid_offsets: FloatArray
    ask_offsets: FloatArray
    residual: FloatArray
    risk_aversion: float

    def value(self, cash: float, mid_price: float) -> FloatArray:
        """Recover V(t,c,q,S) = -exp(-eta(c+qS)) * value_factor."""
        if not np.isfinite(cash) or not np.isfinite(mid_price) or mid_price <= 0:
            raise ValueError("cash must be finite and mid_price must be finite and positive")
        wealth = cash + self.inventory[np.newaxis, :] * mid_price
        return -np.exp(-self.risk_aversion * wealth) * self.value_factor


def _optimal_offset(
    value_at_inventory: float,
    value_at_next_inventory: float,
    parameters: InventoryHJBParameters,
) -> float:
    if (
        not np.isfinite(value_at_inventory)
        or not np.isfinite(value_at_next_inventory)
        or value_at_inventory <= 0
        or value_at_next_inventory <= 0
    ):
        raise ValueError("HJB value factors must be finite and positive")
    eta = parameters.risk_aversion
    candidate = parameters.fee_per_unit + np.log(
        (parameters.distance_sensitivity + eta)
        * value_at_next_inventory
        / (parameters.distance_sensitivity * value_at_inventory)
    ) / eta
    return float(np.clip(candidate, 0.0, parameters.max_quote_distance))


def _inventory_operator(
    value_factor: FloatArray,
    inventory: IntArray,
    parameters: InventoryHJBParameters,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    eta = parameters.risk_aversion
    gamma = parameters.distance_sensitivity
    q_variance = (eta * parameters.volatility) ** 2 * inventory.astype(np.float64) ** 2
    operator = 0.5 * q_variance * value_factor
    bid_offsets = np.zeros_like(value_factor)
    ask_offsets = np.zeros_like(value_factor)

    for index, position in enumerate(inventory):
        if position > parameters.min_inventory:
            lower_value = value_factor[index - 1]
            offset = _optimal_offset(value_factor[index], lower_value, parameters)
            intensity = parameters.arrival_rate * np.exp(-gamma * offset)
            continuation = np.exp(-eta * (offset - parameters.fee_per_unit)) * lower_value
            operator[index] += intensity * (continuation - value_factor[index])
            bid_offsets[index] = offset

        if position < parameters.max_inventory:
            upper_value = value_factor[index + 1]
            offset = _optimal_offset(value_factor[index], upper_value, parameters)
            intensity = parameters.arrival_rate * np.exp(-gamma * offset)
            continuation = np.exp(-eta * (offset - parameters.fee_per_unit)) * upper_value
            operator[index] += intensity * (continuation - value_factor[index])
            ask_offsets[index] = offset

    return operator, bid_offsets, ask_offsets


def solve_inventory_hjb(
    parameters: InventoryHJBParameters,
    *,
    time_steps: int = 201,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
) -> InventoryHJBResult:
    """Solve the one-asset diffusion/fill HJB after CARA factorization.

    The price follows arithmetic Brownian motion, dS = volatility*dW.
    Heavy-tail jumps, log-price dynamics, and multi-asset coupling are not
    included in this reference problem.
    """
    if not isinstance(time_steps, int) or time_steps < 3:
        raise ValueError("time_steps must be an integer of at least 3")
    if (
        not np.isfinite(relative_tolerance)
        or relative_tolerance <= 0
        or not np.isfinite(absolute_tolerance)
        or absolute_tolerance <= 0
    ):
        raise ValueError("solver tolerances must be finite and positive")

    inventory = np.arange(
        parameters.min_inventory,
        parameters.max_inventory + 1,
        dtype=np.int64,
    )
    initial = np.ones(inventory.size, dtype=np.float64)
    reverse_times = np.linspace(0.0, parameters.horizon, time_steps)

    def backward_rhs(_elapsed: float, value_factor: FloatArray) -> FloatArray:
        operator, _, _ = _inventory_operator(value_factor, inventory, parameters)
        return operator

    solution = solve_ivp(
        backward_rhs,
        (0.0, parameters.horizon),
        initial,
        method="DOP853",
        t_eval=reverse_times,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    if not solution.success:
        raise RuntimeError(f"HJB reference solve failed: {solution.message}")
    if not np.all(np.isfinite(solution.y)) or np.any(solution.y <= 0):
        raise RuntimeError("HJB reference solve produced invalid value factors")

    times = parameters.horizon - reverse_times[::-1]
    value_factor = solution.y[:, ::-1].T.copy()
    bid_offsets = np.empty_like(value_factor)
    ask_offsets = np.empty_like(value_factor)
    operators = np.empty_like(value_factor)
    for time_index, factors in enumerate(value_factor):
        operator, bids, asks = _inventory_operator(factors, inventory, parameters)
        operators[time_index] = operator
        bid_offsets[time_index] = bids
        ask_offsets[time_index] = asks

    time_derivative = np.gradient(value_factor, times, axis=0, edge_order=2)
    residual = time_derivative + operators
    for array in (times, inventory, value_factor, bid_offsets, ask_offsets, residual):
        array.setflags(write=False)
    return InventoryHJBResult(
        times=times,
        inventory=inventory,
        value_factor=value_factor,
        bid_offsets=bid_offsets,
        ask_offsets=ask_offsets,
        residual=residual,
        risk_aversion=parameters.risk_aversion,
    )
