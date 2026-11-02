"""Physics-informed approximation for the pair-level mean-reversion HJB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from model_types import FloatArray
from pair_hjb import PairHJBParameters, PairHJBResult, solve_pair_hjb


@dataclass(frozen=True)
class PairPINNTrainingConfig:
    iterations: int = 1_000
    collocation_points: int = 512
    data_batch_size: int = 1_024
    hidden_width: int = 48
    hidden_layers: int = 3
    learning_rate: float = 1e-3
    data_weight: float = 1.0
    terminal_weight: float = 10.0
    seed: int = 7

    def __post_init__(self) -> None:
        integer_values = (
            self.iterations,
            self.collocation_points,
            self.data_batch_size,
            self.hidden_width,
            self.hidden_layers,
        )
        if any(not isinstance(value, int) or value < 1 for value in integer_values):
            raise ValueError("PINN training sizes must be positive integers")
        if self.collocation_points < 2 or self.hidden_width < 4:
            raise ValueError("PINN needs at least two collocation points and width four")
        positive = (self.learning_rate, self.data_weight, self.terminal_weight)
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("PINN loss weights and learning rate must be positive")


class _PairValueNetwork(nn.Module):
    def __init__(self, hidden_width: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_width = 3
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_width, hidden_width), nn.Tanh()))
            input_width = hidden_width
        layers.append(nn.Linear(input_width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


@dataclass(frozen=True)
class PairPINNComparison:
    value_rmse: float
    value_max_error: float
    mean_absolute_residual: float
    max_absolute_residual: float
    terminal_rmse: float


class PairInventoryPINN:
    """Value network and gradient policy for a reduced cointegrated pair."""

    def __init__(
        self,
        parameters: PairHJBParameters,
        network: _PairValueNetwork,
    ) -> None:
        self.parameters = parameters
        self._network = network
        self._network.eval()

    def _normalized_inputs(
        self, times: FloatArray, zscores: FloatArray, inventory: FloatArray
    ) -> Tensor:
        return torch.as_tensor(
            np.column_stack(
                (
                    times / self.parameters.horizon,
                    zscores / self.parameters.z_limit,
                    inventory / self.parameters.maximum_inventory,
                )
            ),
            dtype=torch.float64,
        )

    def predict_value(self, times: FloatArray, zscores: FloatArray, inventory: FloatArray) -> FloatArray:
        time_values, z_values, inventory_values = np.broadcast_arrays(
            np.asarray(times, dtype=np.float64),
            np.asarray(zscores, dtype=np.float64),
            np.asarray(inventory, dtype=np.float64),
        )
        if not (
            np.all(np.isfinite(time_values))
            and np.all(np.isfinite(z_values))
            and np.all(np.isfinite(inventory_values))
        ):
            raise ValueError("PINN states must be finite")
        if np.any(time_values < 0) or np.any(time_values > self.parameters.horizon):
            raise ValueError("times must lie within the pair HJB horizon")
        clipped_z = np.clip(
            z_values.ravel(), -self.parameters.z_limit, self.parameters.z_limit
        )
        clipped_inventory = np.clip(
            inventory_values.ravel(),
            -self.parameters.maximum_inventory,
            self.parameters.maximum_inventory,
        )
        inputs = self._normalized_inputs(
            time_values.ravel(), clipped_z, clipped_inventory
        )
        with torch.no_grad():
            values = self._network(inputs).squeeze(1).cpu().numpy()
        return values.reshape(time_values.shape)

    def trading_rate(self, time_to_horizon: float, zscore: float, inventory: float) -> float:
        if (
            not np.isfinite(time_to_horizon)
            or time_to_horizon < 0
            or time_to_horizon > self.parameters.horizon
            or not np.isfinite(zscore)
            or not np.isfinite(inventory)
        ):
            raise ValueError("policy state must be finite and within the pair HJB horizon")
        time = self.parameters.horizon - time_to_horizon
        z = float(np.clip(zscore, -self.parameters.z_limit, self.parameters.z_limit))
        q = float(
            np.clip(
                inventory,
                -self.parameters.maximum_inventory,
                self.parameters.maximum_inventory,
            )
        )
        inputs = torch.tensor(
            [[
                time / self.parameters.horizon,
                z / self.parameters.z_limit,
                q / self.parameters.maximum_inventory,
            ]],
            dtype=torch.float64,
            requires_grad=True,
        )
        value = self._network(inputs)
        gradient = torch.autograd.grad(value.sum(), inputs)[0][0, 2]
        value_inventory_derivative = (
            float(gradient.detach())
            / self.parameters.maximum_inventory
        )
        rate = np.clip(
            value_inventory_derivative / self.parameters.trading_penalty,
            -self.parameters.maximum_trading_rate,
            self.parameters.maximum_trading_rate,
        )
        if q <= -self.parameters.maximum_inventory + 1e-10:
            rate = max(rate, 0.0)
        elif q >= self.parameters.maximum_inventory - 1e-10:
            rate = min(rate, 0.0)
        return float(rate)

    def compare(self, reference: PairHJBResult) -> PairPINNComparison:
        if not np.isclose(reference.times[-1], self.parameters.horizon):
            raise ValueError("reference horizon must match PINN horizon")
        time_indices = np.linspace(
            0, reference.times.size - 1, min(15, reference.times.size), dtype=int
        )
        z_indices = np.linspace(
            0, reference.z_grid.size - 1, min(21, reference.z_grid.size), dtype=int
        )
        q_indices = np.linspace(
            0,
            reference.inventory_grid.size - 1,
            min(11, reference.inventory_grid.size),
            dtype=int,
        )
        times, zscores, inventories = np.meshgrid(
            reference.times[time_indices],
            reference.z_grid[z_indices],
            reference.inventory_grid[q_indices],
            indexing="ij",
        )
        prediction = self.predict_value(times, zscores, inventories)
        target = reference.value[
            np.ix_(time_indices, z_indices, q_indices)
        ]
        errors = prediction - target
        residual = _pde_residual(
            self._network,
            self.parameters,
            times.ravel(),
            zscores.ravel(),
            inventories.ravel(),
        ).detach().cpu().numpy()
        return PairPINNComparison(
            value_rmse=float(np.sqrt(np.mean(errors**2))),
            value_max_error=float(np.max(np.abs(errors))),
            mean_absolute_residual=float(np.mean(np.abs(residual))),
            max_absolute_residual=float(np.max(np.abs(residual))),
            terminal_rmse=float(np.sqrt(np.mean(errors[-1] ** 2))),
        )


def _pde_residual(
    network: _PairValueNetwork,
    parameters: PairHJBParameters,
    times: FloatArray,
    zscores: FloatArray,
    inventory: FloatArray,
) -> Tensor:
    inputs = torch.tensor(
        np.column_stack(
            (
                times / parameters.horizon,
                zscores / parameters.z_limit,
                inventory / parameters.maximum_inventory,
            )
        ),
        dtype=torch.float64,
        requires_grad=True,
    )
    values = network(inputs)
    gradients = torch.autograd.grad(
        values.sum(), inputs, create_graph=True
    )[0]
    value_time = gradients[:, 0] / parameters.horizon
    value_z = gradients[:, 1] / parameters.z_limit
    value_q = gradients[:, 2] / parameters.maximum_inventory
    value_zz = torch.autograd.grad(
        value_z.sum(), inputs, create_graph=True
    )[0][:, 1] / parameters.z_limit
    drift = -parameters.mean_reversion_speed * torch.as_tensor(
        zscores, dtype=torch.float64
    )
    rate = torch.clamp(
        value_q / parameters.trading_penalty,
        min=-parameters.maximum_trading_rate,
        max=parameters.maximum_trading_rate,
    )
    rate = torch.where(
        torch.as_tensor(inventory, dtype=torch.float64)
        <= -parameters.maximum_inventory + 1e-10,
        torch.clamp(rate, min=0.0),
        rate,
    )
    rate = torch.where(
        torch.as_tensor(inventory, dtype=torch.float64)
        >= parameters.maximum_inventory - 1e-10,
        torch.clamp(rate, max=0.0),
        rate,
    )
    q = torch.as_tensor(inventory, dtype=torch.float64)
    return (
        value_time
        + drift * value_z
        + 0.5 * parameters.spread_volatility**2 * value_zz
        + rate * value_q
        - 0.5 * parameters.trading_penalty * rate**2
        + q * parameters.spread_scale * drift
        - 0.5 * parameters.risk_penalty * q**2
    )


def train_pair_pinn(
    parameters: PairHJBParameters,
    config: PairPINNTrainingConfig = PairPINNTrainingConfig(),
    *,
    reference: PairHJBResult | None = None,
) -> PairInventoryPINN:
    """Train against the HJB residual, terminal condition, and reference grid."""
    torch.manual_seed(config.seed)
    solved = reference or solve_pair_hjb(parameters)
    network = _PairValueNetwork(config.hidden_width, config.hidden_layers).to(
        dtype=torch.float64
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    reference_inputs = np.stack(
        np.meshgrid(solved.times, solved.z_grid, solved.inventory_grid, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    reference_values = solved.value.reshape(-1, 1)
    normalized_reference = torch.as_tensor(
        reference_inputs
        / np.array(
            [
                parameters.horizon,
                parameters.z_limit,
                parameters.maximum_inventory,
            ]
        ),
        dtype=torch.float64,
    )
    target_reference = torch.tensor(reference_values.copy(), dtype=torch.float64)
    terminal_z = torch.rand(config.collocation_points, dtype=torch.float64) * 2 - 1
    terminal_q = torch.rand(config.collocation_points, dtype=torch.float64) * 2 - 1
    terminal_inputs = torch.column_stack(
        (
            torch.ones(config.collocation_points, dtype=torch.float64),
            terminal_z,
            terminal_q,
        )
    )
    terminal_target = (
        -0.5
        * parameters.risk_penalty
        * (terminal_q * parameters.maximum_inventory) ** 2
    ).unsqueeze(1)

    for _ in range(config.iterations):
        time = torch.rand(config.collocation_points, dtype=torch.float64)
        z = (
            torch.rand(config.collocation_points, dtype=torch.float64) * 2 - 1
        ) * parameters.z_limit
        q = (
            torch.rand(config.collocation_points, dtype=torch.float64) * 2 - 1
        ) * parameters.maximum_inventory
        residual = _pde_residual(
            network,
            parameters,
            time.numpy() * parameters.horizon,
            z.numpy(),
            q.numpy(),
        )
        residual_loss = residual.square().mean()
        data_indices = torch.randint(
            normalized_reference.shape[0],
            (min(config.data_batch_size, normalized_reference.shape[0]),),
        )
        data_prediction = network(normalized_reference[data_indices])
        data_loss = (
            data_prediction - target_reference[data_indices]
        ).square().mean()
        terminal_loss = (
            network(terminal_inputs) - terminal_target
        ).square().mean()
        loss = (
            residual_loss
            + config.data_weight * data_loss
            + config.terminal_weight * terminal_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("pair PINN training produced non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return PairInventoryPINN(parameters, network)
