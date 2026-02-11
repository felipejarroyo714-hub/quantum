"""Boundary geometry adapter for embedding molecules in curved λ-geometry.

This module creates boundary surfaces based on van der Waals radii or solvent
accessible approximations and maps them into a λ-scale invariant coordinate
system for downstream field simulations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

logger = logging.getLogger(__name__)


@dataclass
class BoundaryGeometry:
    coordinates: np.ndarray
    radii: np.ndarray
    surface_points: np.ndarray
    metadata: dict | None = None


def _vdw_radius(atom_num: int) -> float:
    table = {1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.8, 16: 1.8, 17: 1.75}
    return table.get(atom_num, 1.75)


def embed_in_curved_geometry(
    mol: Optional["Chem.Mol"], coords: np.ndarray, epsilon: float = 0.05, r0: float = 1.5, lam: float = np.sqrt(6.0) / 2.0
) -> BoundaryGeometry:
    if coords is None or coords.size == 0:
        coords = np.zeros((1, 3), dtype=float)
    if mol is None or Chem is None:
        radii = np.ones(len(coords), dtype=float) * r0
    else:
        radii = np.array([_vdw_radius(atom.GetAtomicNum()) for atom in mol.GetAtoms()], dtype=float)

    # Generate surface samples by expanding coordinates along sphere approximations
    samples = []
    for center, r in zip(coords, radii):
        # six-point stencil on each axis
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = r
            samples.append(center + delta)
            samples.append(center - delta)
    surface = np.vstack(samples) if samples else np.zeros((0, 3))

    # Normalize into λ-scale invariant frame with mild epsilon smoothing
    norms = np.linalg.norm(surface, axis=1) if surface.size else np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = (norms / max(r0, 1e-6)) + epsilon
        surface_lambda = lam * np.log1p(scaled).reshape(-1, 1) * np.ones((1, 3)) if surface.size else surface
    meta = {"epsilon": float(epsilon), "r0": float(r0), "lam": float(lam)}
    return BoundaryGeometry(coordinates=coords, radii=radii, surface_points=surface_lambda, metadata=meta)

