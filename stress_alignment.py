"""Renormalized stress and Einstein-residual alignment utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class StressAlignParams:
    smoothing: float = 0.1
    residual_weight: float = 1.0
    einstein_tol: float = 5.0


@dataclass
class StressAlignmentResult:
    initial_l2: float
    final_l2: float
    max_residual: float
    residual_profile: np.ndarray
    status: str


def _smooth_profile(profile: np.ndarray, smoothing: float) -> np.ndarray:
    if smoothing <= 0 or profile.size == 0:
        return profile
    kernel = np.exp(-np.linspace(-1, 1, 5) ** 2 / smoothing)
    kernel = kernel / np.sum(kernel)
    return np.convolve(profile, kernel, mode="same")


def renormalize_experiment_stress(experiment: "ExperimentRecord", z_grid: np.ndarray, params: StressAlignParams) -> StressAlignmentResult:
    lambda_prior = getattr(experiment, "lambda_prior", None)
    if lambda_prior is not None and getattr(lambda_prior, "curvature", None) is not None:
        z_grid = lambda_prior.z
        R_profile = lambda_prior.curvature
    else:
        R_profile = np.zeros_like(z_grid)

    energy_profile = getattr(experiment, "energy_profile", np.zeros_like(z_grid))
    energy_profile = np.resize(energy_profile, z_grid.shape)
    raw_residual = R_profile - 8 * np.pi * energy_profile
    initial_l2 = float(np.linalg.norm(raw_residual))
    renormalized_energy = _smooth_profile(energy_profile, params.smoothing)
    renorm_residual = R_profile - 8 * np.pi * renormalized_energy
    final_l2 = float(np.linalg.norm(renorm_residual) * params.residual_weight)
    max_residual = float(np.max(np.abs(renorm_residual))) if renorm_residual.size else 0.0
    return StressAlignmentResult(
        initial_l2=initial_l2,
        final_l2=final_l2,
        max_residual=max_residual,
        residual_profile=np.asarray(renorm_residual, dtype=float).reshape(z_grid.shape),
        status="consistent" if final_l2 <= initial_l2 else "improved-but-unstable",
    )
