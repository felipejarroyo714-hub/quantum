"""Phase 5 spectral unification bridge for covariant priors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class SpatialGrid:
    coordinates: np.ndarray
    values: Optional[np.ndarray] = None


@dataclass
class Phase5Params:
    max_l: int = 4
    radial_points: int = 32
    regularization: float = 1e-6


@dataclass
class Phase5Prior:
    K_covariant: np.ndarray
    einstein_residual: Dict[str, float]
    boundary_invariance: float
    covariance_residual: float
    metadata: Dict[str, float] = field(default_factory=dict)


def _build_spherical_harmonics(max_l: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    harmonics = []
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            harmonics.append(np.cos(m * phi) * (np.sin(theta) ** l))
    return np.vstack(harmonics)


def _build_radial_covariance(radial_nodes: np.ndarray, regularization: float) -> np.ndarray:
    distances = np.abs(radial_nodes[:, None] - radial_nodes[None, :])
    kernel = np.exp(-distances)
    kernel += regularization * np.eye(len(radial_nodes))
    return kernel


def build_phase5_covariant_prior(complex_grid: SpatialGrid, params: Phase5Params) -> Phase5Prior:
    coords = complex_grid.coordinates
    if coords.size == 0:
        coords = np.zeros((1, 3))
    r = np.linalg.norm(coords, axis=1)
    theta = np.arccos(np.clip(coords[:, 2] / np.clip(r, 1e-9, None), -1.0, 1.0))
    phi = np.arctan2(coords[:, 1], coords[:, 0])
    radial_nodes = np.linspace(r.min(initial=0.0), r.max(initial=1.0) + 1e-6, params.radial_points)
    radial_cov = _build_radial_covariance(radial_nodes, params.regularization)
    harmonics = _build_spherical_harmonics(params.max_l, theta, phi)
    K_covariant = harmonics.T @ harmonics
    K_covariant = K_covariant / (np.linalg.norm(K_covariant) + 1e-12)
    values = complex_grid.values
    if values is not None:
        values = np.asarray(values, dtype=float).ravel()
        energy_scale = float(np.mean(np.abs(values)) + 1e-12)
    else:
        energy_scale = 1.0
    einstein_residual = {
        "l2": float(np.linalg.norm(radial_cov) * 0.1 * energy_scale),
        "max": float(np.max(np.abs(radial_cov)) * 0.1 * energy_scale),
    }
    boundary_invariance = float(np.std(radial_nodes))
    covariance_residual = float(np.var(radial_cov))
    return Phase5Prior(
        K_covariant=K_covariant,
        einstein_residual=einstein_residual,
        boundary_invariance=boundary_invariance,
        covariance_residual=covariance_residual,
        metadata={"max_l": params.max_l, "radial_points": params.radial_points},
    )
