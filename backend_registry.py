"""Backend registry to centralize physics/ADMET backends with mode-aware validation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, TYPE_CHECKING

import numpy as np

from lambda_geometry_prior import MoleculeRecord, TargetRecord

try:  # pragma: no cover - optional logging utilities
    import rich
except Exception:  # pragma: no cover
    rich = None


logger = logging.getLogger(__name__)


QMBackend = Callable[[MoleculeRecord, "SimulationMode", bool], Dict[str, float]]
DockingBackend = Callable[[MoleculeRecord, TargetRecord, "SimulationMode", bool], Dict[str, float]]
MDBackend = Callable[[MoleculeRecord, TargetRecord, "SimulationMode", bool], Dict[str, np.ndarray]]
ADMETBackend = Callable[[str, "SimulationMode", bool], Dict[str, float]]


@dataclass
class BackendRegistry:
    qm_backend: Optional[QMBackend] = None
    docking_backend: Optional[DockingBackend] = None
    md_backend: Optional[MDBackend] = None
    admet_backend: Optional[ADMETBackend] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def register_qm_backend(self, fn: QMBackend, name: str = "qm") -> None:
        self.qm_backend = fn
        self.metadata["qm"] = name

    def register_docking_backend(self, fn: DockingBackend, name: str = "docking") -> None:
        self.docking_backend = fn
        self.metadata["docking"] = name

    def register_md_backend(self, fn: MDBackend, name: str = "md") -> None:
        self.md_backend = fn
        self.metadata["md"] = name

    def register_admet_backend(self, fn: ADMETBackend, name: str = "admet") -> None:
        self.admet_backend = fn
        self.metadata["admet"] = name

    def validate_backends_for_mode(self, mode: "SimulationMode", allow_stub: bool = False) -> None:
        from drug_discovery_simulation import SimulationMode as _SimulationMode

        non_debug = mode is not _SimulationMode.DEBUG_SYNTHETIC
        if non_debug and allow_stub:
            raise RuntimeError("allow_stub is forbidden for benchmark or production modes")

        if non_debug:
            missing = [
                name
                for name, fn in {
                    "qm": self.qm_backend,
                    "docking": self.docking_backend,
                    "md": self.md_backend,
                    "admet": self.admet_backend,
                }.items()
                if fn is None
            ]
            if missing:
                raise RuntimeError(
                    f"Backends missing for mode {mode.value}: {', '.join(missing)}. "
                    "Configure concrete backends or run in DEBUG_SYNTHETIC for deterministic stubs."
                )

    def run_qm(self, mol: MoleculeRecord, mode: "SimulationMode", allow_stub: bool = False) -> Dict[str, float]:
        if self.qm_backend is None:
            raise RuntimeError("QM backend not registered")
        return self.qm_backend(mol, mode, allow_stub)

    def run_docking(self, mol: MoleculeRecord, tgt: TargetRecord, mode: "SimulationMode", allow_stub: bool = False) -> Dict[str, float]:
        if self.docking_backend is None:
            raise RuntimeError("Docking backend not registered")
        return self.docking_backend(mol, tgt, mode, allow_stub)

    def run_md(self, mol: MoleculeRecord, tgt: TargetRecord, mode: "SimulationMode", allow_stub: bool = False) -> Dict[str, np.ndarray]:
        if self.md_backend is None:
            raise RuntimeError("MD backend not registered")
        return self.md_backend(mol, tgt, mode, allow_stub)

    def run_admet(self, smiles: str, mode: "SimulationMode", allow_stub: bool = False) -> Dict[str, float]:
        if self.admet_backend is None:
            raise RuntimeError("ADMET backend not registered")
        return self.admet_backend(smiles, mode, allow_stub)

