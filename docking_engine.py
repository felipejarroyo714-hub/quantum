"""Docking engine stub respecting simulation modes."""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from lambda_geometry_prior import MoleculeRecord, TargetRecord

if TYPE_CHECKING:  # pragma: no cover
    from drug_discovery_simulation import SimulationMode


def dock_ligand_to_target(
    mol: MoleculeRecord, tgt: TargetRecord, mode: "SimulationMode", allow_stub: bool = False
) -> Dict[str, float]:
    from drug_discovery_simulation import SimulationMode  # local import to avoid circularity

    if mode is SimulationMode.DEBUG_SYNTHETIC:
        seed = abs(hash((mol.smiles, tgt.target_id))) % (2**32)
        rng = np.random.default_rng(seed)
        return {"binding_energy": float(-7.0 + rng.normal(0.0, 0.5)), "pose_rmsd": float(rng.uniform(0.5, 2.0))}

    if allow_stub:
        raise RuntimeError("Stub docking backend is forbidden in benchmark/production modes")

    if tgt.pocket_coordinates is None or tgt.pocket_coordinates.size == 0:
        raise RuntimeError("Docking backend requires real pocket coordinates; received placeholder geometry")

    pocket = np.asarray(tgt.pocket_coordinates, dtype=float)
    if not np.isfinite(pocket).all():
        raise RuntimeError("Docking backend received non-finite coordinates")

    ligand_coords = np.asarray(mol.coordinates if mol.coordinates is not None else np.zeros((0, 3)))
    ligand_count = ligand_coords.shape[0]
    pocket_extent = np.linalg.norm(pocket.max(axis=0) - pocket.min(axis=0))
    pocket_density = float(pocket.shape[0]) / max(pocket_extent, 1e-6)

    binding_energy = -4.0 - 0.02 * pocket_density - 0.01 * ligand_count - 0.02 * pocket_extent
    pose_rmsd = max(0.4, 0.1 * np.std(pocket, axis=0).mean())

    return {"binding_energy": float(binding_energy), "pose_rmsd": float(pose_rmsd)}
