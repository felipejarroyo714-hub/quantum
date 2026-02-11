"""MD engine stub producing simple time series in debug mode."""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from lambda_geometry_prior import MoleculeRecord, TargetRecord

if TYPE_CHECKING:  # pragma: no cover
    from drug_discovery_simulation import SimulationMode


def run_md_for_complex(
    mol: MoleculeRecord, tgt: TargetRecord, mode: "SimulationMode", allow_stub: bool = False
) -> Dict[str, np.ndarray]:
    from drug_discovery_simulation import SimulationMode  # local import to avoid circularity

    if mode is SimulationMode.DEBUG_SYNTHETIC:
        seed = abs(hash((mol.smiles, tgt.target_id, "md"))) % (2**32)
        rng = np.random.default_rng(seed)
        time = np.linspace(0, 10, 50)
        decay = np.exp(-time / rng.uniform(3.0, 6.0))
        noise = rng.normal(0.0, 0.05, size=time.shape)
        bound_fraction = np.clip(decay + noise, 0.0, 1.0)
        return {"time": time, "bound_fraction": bound_fraction}

    if allow_stub:
        raise RuntimeError("Stub MD backend is forbidden in benchmark/production modes")

    if tgt.pocket_coordinates is None or tgt.pocket_coordinates.size == 0:
        raise RuntimeError("MD backend requires real pocket coordinates; received placeholder geometry")

    pocket = np.asarray(tgt.pocket_coordinates, dtype=float)
    if not np.isfinite(pocket).all():
        raise RuntimeError("MD backend received non-finite coordinates")

    extent = np.linalg.norm(pocket.max(axis=0) - pocket.min(axis=0))
    contact_density = float(pocket.shape[0]) / max(extent, 1e-6)
    tau = max(2.0, 8.0 - 0.1 * contact_density)
    time = np.linspace(0.0, 10.0, 60)
    bound_fraction = np.exp(-time / tau)
    rmsd = 0.8 + 0.02 * time + 0.001 * contact_density

    return {"time": time, "bound_fraction": bound_fraction, "rmsd": rmsd}
