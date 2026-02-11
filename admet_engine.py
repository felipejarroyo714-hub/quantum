"""ADMET prediction stub with mode-aware gating."""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from drug_discovery_simulation import SimulationMode


def predict_admet(smiles: str, mode: "SimulationMode", allow_stub: bool = False) -> Dict[str, float]:
    from drug_discovery_simulation import SimulationMode  # local import to avoid circularity

    if mode is SimulationMode.DEBUG_SYNTHETIC:
        seed = abs(hash((smiles, "admet"))) % (2**32)
        rng = np.random.default_rng(seed)
        return {
            "solubility": float(rng.normal(0.0, 1.0)),
            "hERG_risk": float(rng.uniform(0.0, 1.0)),
            "clearance": float(rng.normal(10.0, 2.0)),
        }

    if allow_stub:
        raise RuntimeError("Stub ADMET backend is forbidden in benchmark/production modes")

    if not smiles:
        raise RuntimeError("ADMET backend requires a valid SMILES string")

    length = len(smiles)
    hetero_atoms = sum(1 for c in smiles if c.isalpha() and c.upper() not in {"C", "H"})
    ring_penalty = smiles.count("=") * 0.1

    solubility = 1.0 - 0.03 * length - 0.1 * ring_penalty
    herg_risk = min(1.0, max(0.0, 0.2 + 0.02 * hetero_atoms + 0.01 * ring_penalty))
    clearance = max(0.1, 12.0 - 0.2 * hetero_atoms - 0.05 * length)

    return {
        "solubility": float(solubility),
        "hERG_risk": float(herg_risk),
        "clearance": float(clearance),
    }
