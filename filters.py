"""Adaptive state estimators for noisy market observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_types import FloatArray


@dataclass(frozen=True)
class FilterResult:
    """Causal latent log-price estimates and adaptive process-noise history."""

    estimated_log_prices: FloatArray
    estimated_prices: FloatArray
    innovations: FloatArray
    process_variances: FloatArray


@dataclass(frozen=True)
class AdaptiveKalmanFilter:
    """Independent scalar adaptive Kalman filters for each asset's log-price."""

    initial_process_variance: float
    measurement_variance: float
    initial_state_variance: float
    adaptation_rate: float
    minimum_process_variance: float = 1e-12

    def __post_init__(self) -> None:
        positive_values = (
            self.initial_process_variance,
            self.measurement_variance,
            self.initial_state_variance,
            self.minimum_process_variance,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError("filter variances must be finite and positive")
        if (
            not np.isfinite(self.adaptation_rate)
            or not 0 < self.adaptation_rate <= 1
        ):
            raise ValueError("adaptation_rate must be in (0, 1]")
        if self.minimum_process_variance > self.initial_process_variance:
            raise ValueError("minimum process variance cannot exceed initial variance")

    def filter(self, observed_log_prices: FloatArray) -> FilterResult:
        """Filter time-by-asset log-price observations without future data."""
        observations = np.asarray(observed_log_prices, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[0] < 2 or observations.shape[1] < 1:
            raise ValueError("observations must have shape (at least 2 times, at least 1 asset)")
        if not np.all(np.isfinite(observations)):
            raise ValueError("observations must be finite log-prices")

        time_count, asset_count = observations.shape
        estimates = np.empty_like(observations)
        innovations = np.zeros_like(observations)
        process_variances = np.empty_like(observations)
        state = observations[0].copy()
        posterior_variance = np.full(asset_count, self.initial_state_variance)
        process_variance = np.full(asset_count, self.initial_process_variance)
        estimates[0] = state
        process_variances[0] = process_variance

        for time_index in range(1, time_count):
            prior_variance = posterior_variance.copy()
            predicted_variance = prior_variance + process_variance
            innovation = observations[time_index] - state
            innovation_variance = predicted_variance + self.measurement_variance
            gain = predicted_variance / innovation_variance
            state = state + gain * innovation
            posterior_variance = (1.0 - gain) * predicted_variance

            adaptive_estimate = (
                gain**2 * innovation**2
                + posterior_variance
                - prior_variance
            )
            if not np.all(np.isfinite(adaptive_estimate)):
                raise ValueError("adaptive process-variance estimate is not finite")
            process_variance = np.maximum(
                self.minimum_process_variance,
                (1.0 - self.adaptation_rate) * process_variance
                + self.adaptation_rate * adaptive_estimate,
            )
            if not np.all(np.isfinite(process_variance)):
                raise ValueError("adaptive process variance is not finite")
            estimates[time_index] = state
            innovations[time_index] = innovation
            process_variances[time_index] = process_variance

        with np.errstate(over="ignore"):
            estimated_prices = np.exp(estimates)
        if not np.all(np.isfinite(estimated_prices)) or np.any(estimated_prices <= 0):
            raise ValueError("filtered log-prices are outside the representable price range")
        for array in (estimates, estimated_prices, innovations, process_variances):
            array.setflags(write=False)
        return FilterResult(estimates, estimated_prices, innovations, process_variances)
