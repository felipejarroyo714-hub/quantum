"""Shared experiment record enriched with λ-stack diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from drug_discovery_simulation import SimulationMode
from adiabatic_budget import AdiabaticBudget
from echo_validator import EchoCompareResult
from lambda_geometry_prior import LambdaPrior
from ligc_unified_potential import LigcResult
from phase5_unification_bridge import Phase5Prior
from stress_alignment import StressAlignmentResult


@dataclass
class ExperimentRecord:
    ligand_id: str
    target_id: str
    # High-level feature bundles
    features: Dict[str, Any] = field(default_factory=dict)

    # λ-stack diagnostics
    lambda_prior: Optional[LambdaPrior] = None
    phase5_prior: Optional[Phase5Prior] = None
    adiabatic_budget_snapshot: Optional[AdiabaticBudget] = None
    ligc_result: Optional[LigcResult] = None
    stress_alignment: Optional[StressAlignmentResult] = None
    echo_validation: Optional[EchoCompareResult] = None

    # Fields to be filled by docking/QM/MD/ADMET layers later
    qm_result: Optional[Dict[str, Any]] = None
    docking_result: Optional[Dict[str, Any]] = None
    md_result: Optional[Dict[str, Any]] = None
    admet_result: Optional[Dict[str, Any]] = None

    # Fields for LIGC & stress alignment
    ricci_field: Optional[np.ndarray] = None
    entropy_field: Optional[np.ndarray] = None
    energy_field: Optional[np.ndarray] = None
    energy_profile: Optional[np.ndarray] = None

    provenance: Dict[str, Any] = field(default_factory=dict)

    def set_mode(self, mode: "SimulationMode") -> None:
        self.provenance["mode"] = mode.value

    def composite_lambda_score(self) -> float:
        score = 0.0
        if self.lambda_prior is not None:
            score += float(np.tanh(np.mean(np.abs(self.lambda_prior.descriptors))))
        if self.phase5_prior is not None:
            norm_K = float(np.linalg.norm(self.phase5_prior.K_covariant))
            score += float(np.tanh(norm_K))
        if self.adiabatic_budget_snapshot is not None and self.adiabatic_budget_snapshot.J_adia_init > 0:
            used = 1.0 - float(
                self.adiabatic_budget_snapshot.J_adia_current / self.adiabatic_budget_snapshot.J_adia_init
            )
            score += max(0.0, min(1.0, used))
        if self.ligc_result is not None:
            score += float(np.tanh(1.0 / (1.0 + self.ligc_result.variance)))
            if getattr(self.ligc_result, "status", "") == "marginal":
                score -= 0.5
        if self.echo_validation is not None:
            mapping = {"echo-validated": 0.5, "echo-ambiguous": 0.0, "echo-failed": -0.5}
            score += mapping.get(self.echo_validation.classification, 0.0)
        return score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ligand_id": self.ligand_id,
            "target_id": self.target_id,
            "features": self.features,
            "binding_energy": (self.docking_result or {}).get("binding_energy"),
            "pose_rmsd": (self.docking_result or {}).get("pose_rmsd"),
            "qm": self.qm_result,
            "md": self.md_result,
            "admet": self.admet_result,
            "lambda_prior_status": getattr(self.lambda_prior, "status", None),
            "phase5_boundary_invariance": getattr(self.phase5_prior, "boundary_invariance", None),
            "adiabatic_budget_utilization": None
            if self.adiabatic_budget_snapshot is None
            else float(
                self.adiabatic_budget_snapshot.adiabatic_utilization_ratio
            ),
            "adiabatic_budget_remaining": None
            if self.adiabatic_budget_snapshot is None
            else float(self.adiabatic_budget_snapshot.J_adia_current),
            "ligc_variance": getattr(self.ligc_result, "variance", None),
            "ligc_status": getattr(self.ligc_result, "status", None),
            "einstein_residual_l2": getattr(self.stress_alignment, "final_l2", None),
            "echo_classification": getattr(self.echo_validation, "classification", None),
            "provenance": self.provenance,
            "has_qm": self.qm_result is not None,
            "has_docking": self.docking_result is not None,
            "has_md": self.md_result is not None,
            "has_admet": self.admet_result is not None,
        }

    def attach_echo_result(self, sim: "EchoProfile", ref: "EchoProfile", cfg: "EchoCompareConfig") -> None:
        from echo_validator import compare_echo_profiles

        self.echo_validation = compare_echo_profiles(sim, ref, cfg)


def build_energy_profile_from_results(exp: ExperimentRecord, num_points: int) -> None:
    z = exp.lambda_prior.z if exp.lambda_prior is not None else np.linspace(-1.0, 1.0, num_points)
    if exp.docking_result:
        base_energy = float(exp.docking_result.get("binding_energy", -5.0))
    else:
        base_energy = -5.0
    profile = np.full_like(z, base_energy, dtype=float)
    exp.energy_profile = profile
