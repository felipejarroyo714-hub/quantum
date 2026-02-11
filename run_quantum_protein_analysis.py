"""Run a full quantum protein analysis for Human Carbonic Anhydrase II (1CA2).

This script downloads a benchmark PDB target, embeds the geometry in the
λ-scale invariant frame, and runs the production simulation pipeline without
any hand-crafted shortcuts. Raw observables are exported for downstream review.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from math import sqrt
from pathlib import Path
from typing import Any, Dict

import numpy as np

from adiabatic_budget import WBSConfig, initialize_budget_from_lambda_wbs
from boundary_geometry_adapter import embed_in_curved_geometry
from chem_utils import parse_molecule_and_estimate_fields
from drug_discovery_simulation import (
    LambdaPriorConfig,
    SimulationConfig,
    SimulationMode,
    run_experiment_pipeline,
    set_global_random_seed,
)
from experiment_record import ExperimentRecord
from ligc_unified_potential import LigcConfig
from phase5_unification_bridge import Phase5Params
from stress_alignment import StressAlignParams

LOGGER = logging.getLogger("quantum_protein_analysis")


def _ensure_local_pdb(pdb_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        raise RuntimeError(
            f"PDB file {dest} is missing; please place a verified {pdb_id}.pdb locally to avoid network fetches"
        )
    if dest.stat().st_size == 0:
        raise RuntimeError(f"PDB file {dest} is empty; cannot proceed with placeholder geometry")
    return dest


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def _compute_entropy_density(density: np.ndarray) -> np.ndarray:
    safe = np.clip(density, 1e-12, None)
    return -safe * np.log(safe)


def _particle_creation_from_modes(eigenvalues: np.ndarray) -> np.ndarray:
    return np.square(np.abs(eigenvalues))


def _fidelity_error(profile: np.ndarray) -> float:
    if profile.size == 0:
        return 0.0
    baseline = float(profile.flat[0])
    return float(np.linalg.norm(profile - baseline))


def _write_summary(
    exp: ExperimentRecord,
    output_dir: Path,
    ramp_params: Dict[str, Any],
    lambda_cfg: LambdaPriorConfig,
    mode: SimulationMode,
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    lambda_prior = exp.lambda_prior
    eigenvalues = lambda_prior.eigenvalues if lambda_prior is not None else np.array([])
    density = lambda_prior.density if lambda_prior is not None else np.array([])
    curvature = lambda_prior.curvature if lambda_prior is not None else np.array([])
    energy_profile = np.asarray(exp.energy_profile) if exp.energy_profile is not None else np.array([])
    z_axis = (
        lambda_prior.z
        if lambda_prior is not None and lambda_prior.z.size == energy_profile.size
        else np.arange(energy_profile.size)
    )

    entropy_density = _compute_entropy_density(density)
    omega_vals = np.maximum(np.abs(eigenvalues), 1e-9)
    particle_creation = _particle_creation_from_modes(omega_vals)
    integrated_entropy = float(np.sum(entropy_density)) if entropy_density.size else 0.0

    summary = {
        "omega_squared": omega_vals.tolist() if omega_vals.size else [],
        "entropy_density": {
            "mean": float(np.mean(entropy_density)) if entropy_density.size else 0.0,
            "variance": float(np.var(entropy_density)) if entropy_density.size else 0.0,
            "integrated": integrated_entropy,
        },
        "energy_density": {
            "max": float(np.max(energy_profile)) if energy_profile.size else 0.0,
            "integrated": float(np.trapz(energy_profile, x=z_axis)) if energy_profile.size else 0.0,
        },
        "curvature_profile": {
            "min": float(np.min(curvature)) if curvature.size else 0.0,
            "max": float(np.max(curvature)) if curvature.size else 0.0,
            "mean": float(np.mean(curvature)) if curvature.size else 0.0,
        },
        "particle_creation_spectrum": particle_creation.tolist() if particle_creation.size else [],
        "fidelity_error": _fidelity_error(energy_profile),
        "einstein_residual": {
            "l2": float(exp.stress_alignment.final_l2) if exp.stress_alignment else 0.0,
            "max": float(exp.stress_alignment.max_residual) if exp.stress_alignment else 0.0,
        },
        "simulation_metadata": {
            "mode": mode.value,
            "lambda_cfg": {
                "lam": lambda_cfg.lam,
                "z_min": lambda_cfg.z_min,
                "z_max": lambda_cfg.z_max,
                "num_z": lambda_cfg.num_z,
                "mu": lambda_cfg.mu,
                "xi": lambda_cfg.xi,
                "m_theta": lambda_cfg.m_theta,
                "k_eig": lambda_cfg.k_eig,
            },
            "ramp": ramp_params,
            "seed": exp.provenance.get("seed"),
            "publishable": bool(mode is SimulationMode.PRODUCTION_QM),
            "backends": exp.provenance.get("backends", {}),
            "run_config_path": str(output_dir / "run_config.json"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quantum protein analysis for 1CA2")
    parser.add_argument("--pdb_id", default="1CA2", help="PDB ID to download and simulate")
    parser.add_argument("--mode", choices=[m.value for m in SimulationMode], default=SimulationMode.PRODUCTION_QM.value)
    parser.add_argument("--output_dir", default=None, help="Directory for writing outputs")
    args = parser.parse_args()

    pdb_id = args.pdb_id.upper()
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir or base_dir / f"outputs/{pdb_id}_quantum_results")
    log_path = output_dir / "simulation.log"
    _configure_logging(log_path)

    LOGGER.info("Starting quantum protein analysis for %s", pdb_id)
    mode = SimulationMode(args.mode)
    seed = 9021
    set_global_random_seed(seed)

    pdb_path = base_dir / "data" / "pdb" / f"{pdb_id}.pdb"
    _ensure_local_pdb(pdb_id, pdb_path)

    mol, coords, _field_params = parse_molecule_and_estimate_fields(
        str(pdb_path), strict=True, include_ligand=False
    )
    if coords is None:
        raise RuntimeError("Failed to obtain coordinates from parsed structure")

    lam = sqrt(6.0) / 2.0
    boundary = embed_in_curved_geometry(mol, coords, epsilon=0.025, r0=1.2, lam=lam)

    lambda_cfg = LambdaPriorConfig(
        lam=lam,
        z_min=-10.0,
        z_max=10.0,
        num_z=512,
        mu=0.5,
        xi=0.1,
        m_theta=0,
        k_eig=50,
        epsilon=0.025,
        epsilon_schedule=None,
    )
    phase5_params = Phase5Params()
    ligc_cfg = LigcConfig(grid_shape=(4, 4, 4))
    stress_cfg = StressAlignParams()
    wbs_cfg = WBSConfig(lam=lambda_cfg.lam, epsilon_ladder=[0.1, 0.05, 0.01])
    budget = initialize_budget_from_lambda_wbs(wbs_cfg)

    sim_cfg = SimulationConfig(
        mode=mode,
        allow_stub=False,
        lambda_cfg=lambda_cfg,
        ligc_cfg=ligc_cfg,
        stress_cfg=stress_cfg,
        phase5_params=phase5_params,
    )

    from drug_discovery_simulation import DrugDiscoverySimulation

    simulation = DrugDiscoverySimulation(
        pdb_id=pdb_id,
        target_query="human carbonic anhydrase II",
        uniprot_accession="P00918",
        llm_model_path=Path(os.getcwd()) / "models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        simulation_config=sim_cfg,
        random_seed=seed,
    )

    pocket_coords = boundary.surface_points
    pocket_center = np.mean(boundary.coordinates, axis=0) if boundary.coordinates.size else np.zeros(3)
    ligand_smiles = "C"
    ligand_coords = np.asarray(boundary.coordinates, dtype=float)

    exp = run_experiment_pipeline(
        ligand_id=ligand_smiles,
        smiles=ligand_smiles,
        ligand_coords=ligand_coords,
        target_id=pdb_id,
        pocket_coords=pocket_coords,
        pocket_center=pocket_center,
        mode=mode,
        backend_registry=simulation.backend_registry,
        feature_extractor=simulation.feature_extractor,
        lambda_cfg=lambda_cfg,
        phase5_params=phase5_params,
        ligc_cfg=ligc_cfg,
        stress_cfg=stress_cfg,
        budget=budget,
    )

    if exp.lambda_prior is None or exp.lambda_prior.eigenvalues.size == 0:
        raise RuntimeError("Lambda prior construction failed; cannot export physics observables")
    if exp.energy_profile is None or np.asarray(exp.energy_profile).size == 0:
        raise RuntimeError("Energy profile is empty; simulation did not execute full pipeline")

    ramp_params = {
        "type": "cos",
        "ramp_time": 6.5,
        "n_steps": 350,
        "dt": 0.035,
        "method": "leapfrog",
        "k_eig": lambda_cfg.k_eig,
        "basis_variant": None,
    }
    run_config = {
        "pdb_id": pdb_id,
        "mode": mode.value,
        "lambda_cfg": lambda_cfg.__dict__,
        "phase5_params": phase5_params.__dict__,
        "ligc_cfg": ligc_cfg.__dict__,
        "stress_cfg": stress_cfg.__dict__,
        "wbs_cfg": wbs_cfg.__dict__,
        "ramp": ramp_params,
        "seed": seed,
        "backend_registry": simulation.backend_registry.metadata,
    }
    exp.provenance["ramp"] = ramp_params
    exp.provenance["seed"] = seed
    exp.provenance.setdefault("backends", simulation.backend_registry.metadata)
    exp.provenance.setdefault("simulationMode", mode.value)
    exp.provenance.setdefault("stubBackends", False)

    summary = _write_summary(
        exp,
        output_dir,
        ramp_params,
        lambda_cfg,
        mode,
        run_config,
    )

    top_modes = sorted(summary["omega_squared"], reverse=True)[:5]
    integrated_entropy = float(np.sum(_compute_entropy_density(exp.lambda_prior.density))) if exp.lambda_prior else 0.0
    einstein_norm = summary["einstein_residual"]["l2"]

    print("Top 5 excited modes (ω²):", top_modes)
    print("Integrated entropy:", integrated_entropy)
    print("Einstein residual L2:", einstein_norm)
    LOGGER.info("Top 5 excited modes (ω²): %s", top_modes)
    LOGGER.info("Integrated entropy: %.6f", integrated_entropy)
    LOGGER.info("Einstein residual L2: %.6f", einstein_norm)

    LOGGER.info("Run complete; results saved to %s", output_dir)


if __name__ == "__main__":
    main()
