"""LIGC unified potential alignment for multi-objective consistency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class LigcConfig:
    grid_shape: Tuple[int, int, int] = (8, 8, 8)
    curvature_weight: float = 1.0
    entropy_weight: float = 1.0
    energy_weight: float = 1.0
    gamma_prior: float | None = None
    delta_prior: float | None = None


@dataclass
class LigcResult:
    gamma: float
    delta: float
    variance: float
    stability: float
    status: str
    gamma_prior: float | None = None
    delta_prior: float | None = None
    gamma_deviation: float | None = None
    delta_deviation: float | None = None


class ExperimentRecordProtocol:
    ricci_field: np.ndarray
    entropy_field: np.ndarray
    energy_field: np.ndarray


def _flatten_fields(experiment: ExperimentRecordProtocol, cfg: LigcConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeros = np.zeros(cfg.grid_shape, dtype=float)
    ricci = np.asarray(getattr(experiment, "ricci_field", zeros), dtype=float)
    entropy = np.asarray(getattr(experiment, "entropy_field", zeros), dtype=float)
    energy = np.asarray(getattr(experiment, "energy_field", zeros), dtype=float)
    return ricci.ravel(), entropy.ravel(), energy.ravel()


def compute_ligc_for_experiment(experiment: ExperimentRecordProtocol, config: LigcConfig) -> LigcResult:
    ricci, entropy, energy = _flatten_fields(experiment, config)
    S = np.vstack([config.entropy_weight * entropy, config.energy_weight * energy]).T
    target = -config.curvature_weight * ricci
    if S.shape[0] == 0 or not np.any(S):
        return LigcResult(0.0, 0.0, 0.0, 0.0, status="empty")
    if config.gamma_prior is not None and config.delta_prior is not None:
        reg = 1e-2
        S_reg = np.vstack(
            [
                S,
                np.sqrt(reg) * np.array([1.0, 0.0]),
                np.sqrt(reg) * np.array([0.0, 1.0]),
            ]
        )
        target_reg = np.concatenate(
            [
                target,
                np.sqrt(reg) * np.array([config.gamma_prior]),
                np.sqrt(reg) * np.array([config.delta_prior]),
            ]
        )
        coeffs, _, _, _ = np.linalg.lstsq(S_reg, target_reg, rcond=None)
    else:
        coeffs, _, _, _ = np.linalg.lstsq(S, target, rcond=None)
    gamma, delta = coeffs[:2]
    unified = ricci + gamma * entropy + delta * energy
    variance = float(np.var(unified))
    stability = float(np.std(coeffs))
    if variance < 1e-2:
        status = "tight"
    elif variance < 1.0:
        status = "stable"
    else:
        status = "marginal"
    gamma_prior = config.gamma_prior if config.gamma_prior is not None else None
    delta_prior = config.delta_prior if config.delta_prior is not None else None
    return LigcResult(
        float(gamma),
        float(delta),
        variance,
        stability,
        status,
        gamma_prior=gamma_prior,
        delta_prior=delta_prior,
        gamma_deviation=None if gamma_prior is None else float(gamma - gamma_prior),
        delta_deviation=None if delta_prior is None else float(delta - delta_prior),
    )
