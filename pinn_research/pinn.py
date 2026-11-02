"""Physics-informed neural approximation for the reduced inventory HJB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .hjb import InventoryHJBParameters, InventoryHJBResult
from .model_types import FloatArray, IntArray
from .baselines import Quotes


@dataclass(frozen=True)
class PINNTrainingConfig:
    iterations: int = 2_000
    collocation_times: int = 64
    hidden_width: int = 64
    hidden_layers: int = 3
    learning_rate: float = 1e-3
    terminal_weight: float = 10.0
    seed: int = 7

    def __post_init__(self) -> None:
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(self.collocation_times, int) or self.collocation_times < 2:
            raise ValueError("collocation_times must be an integer of at least 2")
        if not isinstance(self.hidden_width, int) or self.hidden_width < 4:
            raise ValueError("hidden_width must be an integer of at least 4")
        if not isinstance(self.hidden_layers, int) or self.hidden_layers < 1:
            raise ValueError("hidden_layers must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.terminal_weight) or self.terminal_weight <= 0:
            raise ValueError("terminal_weight must be finite and positive")


class _ValueNetwork(nn.Module):
    def __init__(self, hidden_width: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_width = 2
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_width, hidden_width), nn.Tanh()))
            input_width = hidden_width
        output = nn.Linear(input_width, 1)
        layers.append(output)
        self.layers = nn.Sequential(*layers)
        nn.init.constant_(output.bias, float(np.log(np.expm1(1.0))))
        self.positive = nn.Softplus()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.positive(self.layers(inputs))


@dataclass(frozen=True)
class PINNComparison:
    """Error and residual summaries against the numerical reference solution."""

    value_factor_rmse: float
    value_factor_max_error: float
    mean_absolute_residual: float
    max_absolute_residual: float
    terminal_rmse: float


class InventoryPINN:
    """Trained positive value-factor network for the reduced inventory HJB."""

    def __init__(
        self,
        parameters: InventoryHJBParameters,
        network: _ValueNetwork,
    ) -> None:
        self.parameters = parameters
        self._network = network
        self._network.eval()
        self._inventory = torch.arange(
            parameters.min_inventory,
            parameters.max_inventory + 1,
            dtype=torch.float64,
        )
        self._inventory_scale = max(
            1.0, abs(parameters.min_inventory), abs(parameters.max_inventory)
        )

    def predict_value_factor(self, times: FloatArray) -> FloatArray:
        """Evaluate v(t,q) for every finite inventory state."""
        requested_times = np.asarray(times, dtype=np.float64)
        if requested_times.ndim != 1 or not np.all(np.isfinite(requested_times)):
            raise ValueError("times must be a finite one-dimensional array")
        if np.any(requested_times < 0) or np.any(requested_times > self.parameters.horizon):
            raise ValueError("times must lie within the HJB horizon")
        time_tensor = torch.tensor(requested_times.copy(), dtype=torch.float64)
        inventory = self._inventory
        time_input = (
            time_tensor[:, None, None]
            .expand(-1, inventory.numel(), 1)
            .reshape(-1, 1)
            / self.parameters.horizon
        )
        inventory_input = (
            inventory[None, :, None]
            .expand(time_tensor.numel(), -1, -1)
            .reshape(-1, 1)
            / self._inventory_scale
        )
        with torch.no_grad():
            prediction = self._network(torch.cat((time_input, inventory_input), dim=1))
        return prediction.reshape(time_tensor.numel(), inventory.numel()).cpu().numpy()

    def compare(self, reference: InventoryHJBResult) -> PINNComparison:
        """Compare predictions and autograd residuals on a reference grid."""
        if not np.array_equal(reference.inventory, np.asarray(self._inventory, dtype=np.int64)):
            raise ValueError("reference inventory grid must match PINN inventory grid")
        if not np.isclose(reference.times[-1], self.parameters.horizon):
            raise ValueError("reference horizon must match PINN horizon")
        prediction = self.predict_value_factor(reference.times)
        error = prediction - reference.value_factor
        residual = _hjb_residual(
            self._network,
            self.parameters,
            self._inventory,
            self._inventory_scale,
            torch.as_tensor(
                reference.times[:-1] / self.parameters.horizon,
                dtype=torch.float64,
            ),
        ).detach().cpu().numpy()
        return PINNComparison(
            value_factor_rmse=float(np.sqrt(np.mean(error**2))),
            value_factor_max_error=float(np.max(np.abs(error))),
            mean_absolute_residual=float(np.mean(np.abs(residual))),
            max_absolute_residual=float(np.max(np.abs(residual))),
            terminal_rmse=float(np.sqrt(np.mean(error[-1] ** 2))),
        )

    def quoting_policy(self, *, time_nodes: int = 201) -> HJBQuotingPolicy:
        """Create a policy using this PINN's learned optimal offsets."""
        return HJBQuotingPolicy(self, time_nodes=time_nodes)


class HJBQuotingPolicy:
    """Map learned finite-inventory offsets into the shared quoting interface."""

    def __init__(self, pinn: InventoryPINN, *, time_nodes: int = 201) -> None:
        if not isinstance(time_nodes, int) or time_nodes < 2:
            raise ValueError("time_nodes must be an integer of at least 2")
        self._pinn = pinn
        self._parameters = pinn.parameters
        self._times = np.linspace(0.0, self._parameters.horizon, time_nodes)
        self._factors = pinn.predict_value_factor(self._times)
        self._inventory = np.arange(
            self._parameters.min_inventory,
            self._parameters.max_inventory + 1,
            dtype=np.int64,
        )

    def quote(
        self, mid_prices: FloatArray, inventory: IntArray, time_to_horizon: float
    ) -> Quotes:
        mids = np.asarray(mid_prices, dtype=np.float64)
        positions = np.asarray(inventory, dtype=np.int64)
        if mids.shape != (1,) or positions.shape != (1,):
            raise ValueError("the reduced HJB policy supports exactly one asset")
        if not np.all(np.isfinite(mids)) or mids[0] <= 0:
            raise ValueError("mid-price must be finite and positive")
        if (
            not np.isfinite(time_to_horizon)
            or time_to_horizon < 0
            or time_to_horizon > self._parameters.horizon
        ):
            raise ValueError("time_to_horizon must lie within the HJB horizon")
        position = int(positions[0])
        if position < self._parameters.min_inventory or position > self._parameters.max_inventory:
            raise ValueError("inventory is outside configured HJB bounds")

        model_time = self._parameters.horizon - time_to_horizon
        factors = np.array(
            [
                np.interp(model_time, self._times, self._factors[:, index])
                for index in range(self._inventory.size)
            ],
            dtype=np.float64,
        )
        index = position - self._parameters.min_inventory
        eta = self._parameters.risk_aversion
        k = self._parameters.distance_sensitivity
        fee = self._parameters.fee_per_unit
        maximum_distance = self._parameters.max_quote_distance
        bid_offset = 0.0
        ask_offset = 0.0
        bid_enabled = position < self._parameters.max_inventory
        ask_enabled = position > self._parameters.min_inventory
        if position > self._parameters.min_inventory:
            bid_offset = float(
                np.clip(
                    fee + np.log((k + eta) * factors[index - 1] / (k * factors[index])) / eta,
                    0.0,
                    maximum_distance,
                )
            )
        if position < self._parameters.max_inventory:
            ask_offset = float(
                np.clip(
                    fee + np.log((k + eta) * factors[index + 1] / (k * factors[index])) / eta,
                    0.0,
                    maximum_distance,
                )
            )
        bid = np.array([max(0.0, mids[0] - bid_offset)], dtype=np.float64)
        ask = np.array([mids[0] + ask_offset], dtype=np.float64)
        return Quotes(
            bid=bid,
            ask=ask,
            bid_enabled=np.array([bid_enabled], dtype=np.bool_),
            ask_enabled=np.array([ask_enabled], dtype=np.bool_),
        )


def _hjb_residual(
    network: _ValueNetwork,
    parameters: InventoryHJBParameters,
    inventory: Tensor,
    inventory_scale: float,
    normalized_times: Tensor,
) -> Tensor:
    """Return PDE residuals for all inventory states at sampled times."""
    time_values = normalized_times.detach().clone().requires_grad_(True)
    batch_size = time_values.numel()
    state_count = inventory.numel()
    time_input = time_values.repeat_interleave(state_count).unsqueeze(1)
    inventory_input = (
        inventory.repeat(batch_size).unsqueeze(1) / inventory_scale
    )
    factors = network(torch.cat((time_input, inventory_input), dim=1)).reshape(
        batch_size, state_count
    )
    time_derivative_normalized = torch.autograd.grad(
        factors.sum(), time_values, create_graph=True
    )[0]
    time_derivative = (
        time_derivative_normalized.unsqueeze(1) / parameters.horizon
    )

    eta = parameters.risk_aversion
    k = parameters.distance_sensitivity
    diffusion = (
        0.5
        * (eta * parameters.volatility) ** 2
        * inventory.square()
    )
    operator = factors * diffusion.unsqueeze(0)
    fee = parameters.fee_per_unit
    intensity_base = parameters.arrival_rate
    maximum_distance = parameters.max_quote_distance

    for index, position in enumerate(range(parameters.min_inventory, parameters.max_inventory + 1)):
        if position > parameters.min_inventory:
            lower = factors[:, index - 1]
            current = factors[:, index]
            offset = fee + torch.log((k + eta) * lower / (k * current)) / eta
            offset = torch.clamp(offset, min=0.0, max=maximum_distance)
            intensity = intensity_base * torch.exp(-k * offset)
            continuation = torch.exp(-eta * (offset - fee)) * lower
            operator[:, index] = operator[:, index] + intensity * (continuation - current)
        if position < parameters.max_inventory:
            upper = factors[:, index + 1]
            current = factors[:, index]
            offset = fee + torch.log((k + eta) * upper / (k * current)) / eta
            offset = torch.clamp(offset, min=0.0, max=maximum_distance)
            intensity = intensity_base * torch.exp(-k * offset)
            continuation = torch.exp(-eta * (offset - fee)) * upper
            operator[:, index] = operator[:, index] + intensity * (continuation - current)

    return time_derivative + operator


def train_inventory_pinn(
    parameters: InventoryHJBParameters,
    config: PINNTrainingConfig = PINNTrainingConfig(),
) -> InventoryPINN:
    """Train a PINN using only the HJB residual and terminal condition."""
    torch.manual_seed(config.seed)
    network = _ValueNetwork(config.hidden_width, config.hidden_layers).to(
        dtype=torch.float64
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    inventory = torch.arange(
        parameters.min_inventory,
        parameters.max_inventory + 1,
        dtype=torch.float64,
    )
    inventory_scale = max(
        1.0, abs(parameters.min_inventory), abs(parameters.max_inventory)
    )
    normalized_terminal_times = torch.ones(config.collocation_times, dtype=torch.float64)

    for _ in range(config.iterations):
        normalized_times = torch.rand(config.collocation_times, dtype=torch.float64)
        residual = _hjb_residual(
            network,
            parameters,
            inventory,
            inventory_scale,
            normalized_times,
        )
        pde_loss = torch.mean(residual.square())

        terminal_input = torch.stack(
            (
                normalized_terminal_times.repeat_interleave(inventory.numel()),
                inventory.repeat(config.collocation_times) / inventory_scale,
            ),
            dim=1,
        )
        terminal_prediction = network(terminal_input)
        terminal_loss = torch.mean((terminal_prediction - 1.0).square())
        loss = pde_loss + config.terminal_weight * terminal_loss
        if not torch.isfinite(loss):
            raise RuntimeError("PINN training produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return InventoryPINN(parameters, network)


def compare_inventory_pinn(
    pinn: InventoryPINN, reference: InventoryHJBResult
) -> PINNComparison:
    """Public convenience function for numerical-reference comparison."""
    return pinn.compare(reference)
