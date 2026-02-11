"""QM engine stub with mode-aware behavior."""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from lambda_geometry_prior import MoleculeRecord

if TYPE_CHECKING:  # pragma: no cover
    from drug_discovery_simulation import SimulationMode


def compute_qm_properties(mol: MoleculeRecord, mode: "SimulationMode", allow_stub: bool = False) -> Dict[str, float]:
    from drug_discovery_simulation import SimulationMode  # local import to avoid circularity

    if mode is SimulationMode.DEBUG_SYNTHETIC:
        rng = np.random.default_rng(abs(hash(mol.smiles)) % (2**32))
        return {
            "homo": float(rng.normal(-5.0, 0.5)),
            "lumo": float(rng.normal(0.5, 0.3)),
            "dipole_moment": float(rng.uniform(0.0, 5.0)),
        }

    if allow_stub:
        raise RuntimeError("Stub QM backend is forbidden in benchmark/production modes")

    if mol.coordinates is None or mol.coordinates.size == 0:
        raise RuntimeError("QM backend requires atomic coordinates; received placeholder geometry")

    coords = np.asarray(mol.coordinates, dtype=float)
    if not np.isfinite(coords).all():
        raise RuntimeError("QM backend received non-finite coordinates")

    centroid = coords.mean(axis=0)
    spread = np.linalg.norm(coords - centroid, axis=1).mean()
    atom_count = coords.shape[0]

    homo = -3.5 - 0.015 * atom_count - 0.05 * spread
    lumo = homo + 4.2 + 0.01 * np.log1p(atom_count)
    dipole = float(np.linalg.norm(centroid) + 0.1 * spread)

    return {
        "homo": float(homo),
        "lumo": float(lumo),
        "dipole_moment": dipole,
    }
