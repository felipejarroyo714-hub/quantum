"""Run ligand-bound HCA II (3HS4) quantum simulation with λ-stack outputs.

This harness reuses the core DrugDiscoverySimulation pipeline, forbids stub
backends, and exports physics observables plus a delta profile versus the apo
1CA2 run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adiabatic_budget import WBSConfig, initialize_budget_from_lambda_wbs
from boundary_geometry_adapter import embed_in_curved_geometry
from chem_utils import parse_molecule_and_estimate_fields, sanitize_smiles
from drug_discovery_simulation import (
    FeatureExtractor,
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

LOGGER = logging.getLogger("hca2_azm_quantum")

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def _ensure_local_pdb(pdb_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        raise RuntimeError(
            f"PDB file {dest} is missing; please place a verified {pdb_id}.pdb locally to avoid network fetches"
        )
    if dest.stat().st_size == 0:
        raise RuntimeError(f"PDB file {dest} is empty; cannot proceed with placeholder geometry")
    return dest


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


def _extract_ligand_from_pdb(pdb_path: Path, resname: str = "AZM") -> Tuple[str, np.ndarray]:
    if Chem is None:
        raise RuntimeError("RDKit is required to extract ligand coordinates from PDB")
    mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=False, removeHs=False)
    if mol is None:
        raise RuntimeError(f"Failed to parse PDB file {pdb_path}")

    conformer = mol.GetConformer() if mol.GetNumConformers() else None
    if conformer is None:
        raise RuntimeError("No conformer available for ligand extraction")

    editable = Chem.RWMol()
    old_to_new: Dict[int, int] = {}
    coords: List[np.ndarray] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        if info.GetResName().strip().upper() != resname.upper():
            continue
        new_idx = editable.AddAtom(atom)
        old_to_new[atom.GetIdx()] = new_idx
        coords.append(np.array(conformer.GetAtomPosition(atom.GetIdx()), dtype=float))
    for bond in mol.GetBonds():
        bgn, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bgn in old_to_new and end in old_to_new:
            editable.AddBond(old_to_new[bgn], old_to_new[end], bond.GetBondType())
    ligand = editable.GetMol()
    try:
        Chem.SanitizeMol(ligand)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Ligand sanitization warning: %s", exc)
    ligand_smiles = Chem.MolToSmiles(ligand, canonical=True) if ligand.GetNumAtoms() else ""
    ligand_coords = np.vstack(coords) if coords else np.zeros((0, 3), dtype=float)
    if ligand_coords.size == 0 or not ligand_smiles:
        raise RuntimeError("Ligand extraction failed; no AZM atoms found")
    canonical = sanitize_smiles(ligand_smiles, strict=True)
    if canonical is None:
        raise RuntimeError("Extracted ligand SMILES failed sanitization")
    return canonical, ligand_coords


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
            "publishable": bool(mode is SimulationMode.PRODUCTION_QM and not exp.provenance.get("stubBackends", False)),
            "backends": exp.provenance.get("backends", {}),
            "run_config_path": str(output_dir / "run_config.json"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    return summary


def _compute_delta_profile(bound_summary: Dict[str, Any], apo_summary: Dict[str, Any]) -> Dict[str, Any]:
    def _arr(path: str, default: List[float]) -> np.ndarray:
        return np.asarray(bound_summary.get(path, default))

    def _arr_other(key: str, default: List[float]) -> np.ndarray:
        return np.asarray(apo_summary.get(key, default))

    omega_bound = _arr("omega_squared", [])
    omega_apo = _arr_other("omega_squared", [])
    max_len = max(omega_bound.size, omega_apo.size)
    omega_bound = np.pad(omega_bound, (0, max_len - omega_bound.size))
    omega_apo = np.pad(omega_apo, (0, max_len - omega_apo.size))

    ent_bound = bound_summary.get("entropy_density", {})
    ent_apo = apo_summary.get("entropy_density", {})
    eng_bound = bound_summary.get("energy_density", {})
    eng_apo = apo_summary.get("energy_density", {})
    curv_bound = bound_summary.get("curvature_profile", {})
    curv_apo = apo_summary.get("curvature_profile", {})

    part_bound = np.asarray(bound_summary.get("particle_creation_spectrum", []))
    part_apo = np.asarray(apo_summary.get("particle_creation_spectrum", []))
    max_p = max(part_bound.size, part_apo.size)
    part_bound = np.pad(part_bound, (0, max_p - part_bound.size))
    part_apo = np.pad(part_apo, (0, max_p - part_apo.size))

    delta_entropy = (ent_bound.get("integrated", 0.0) - ent_apo.get("integrated", 0.0))
    delta_energy = (eng_bound.get("integrated", 0.0) - eng_apo.get("integrated", 0.0))
    delta_curv = (curv_bound.get("mean", 0.0) - curv_apo.get("mean", 0.0))

    return {
        "omega_delta": (omega_bound - omega_apo).tolist(),
        "entropy_integrated_delta": float(delta_entropy),
        "energy_integrated_delta": float(delta_energy),
        "curvature_mean_delta": float(delta_curv),
        "particle_creation_delta": (part_bound - part_apo).tolist(),
        "particle_creation_total_change": float(np.sum(part_bound) - np.sum(part_apo)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ligand-bound HCA II (3HS4) quantum simulation")
    parser.add_argument("--pdb_id", default="3HS4", help="PDB ID to load")
    parser.add_argument("--mode", choices=[m.value for m in SimulationMode], default=SimulationMode.PRODUCTION_QM.value)
    parser.add_argument("--output_dir", default=None, help="Directory for outputs")
    parser.add_argument("--apo_summary", default=None, help="Path to apo summary JSON for delta comparison")
    args = parser.parse_args()

    mode = SimulationMode(args.mode)
    if mode is not SimulationMode.PRODUCTION_QM:
        raise RuntimeError("Ligand-bound benchmark must run in production_qm mode")

    base_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir or base_dir / "outputs/hca2_azm_quantum_results")
    log_path = output_dir / "simulation.log"
    _configure_logging(log_path)

    seed = 9021
    set_global_random_seed(seed)
    pdb_id = args.pdb_id.upper()

    pdb_path = _ensure_local_pdb(pdb_id, base_dir / "data" / "pdb" / f"{pdb_id}.pdb")

    protein_mol, protein_coords, _ = parse_molecule_and_estimate_fields(str(pdb_path), strict=True, include_ligand=False)
    if protein_coords is None:
        raise RuntimeError("Failed to extract protein coordinates for HCA II")
    ligand_smiles, ligand_coords = _extract_ligand_from_pdb(pdb_path, resname="AZM")

    lam = sqrt(6.0) / 2.0
    boundary = embed_in_curved_geometry(protein_mol, protein_coords, epsilon=0.025, r0=1.2, lam=lam)

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
        benchmark_datasets=[],
    )

    from drug_discovery_simulation import DrugDiscoverySimulation

    simulation = DrugDiscoverySimulation(
        pdb_id=pdb_id,
        target_query="human carbonic anhydrase II",
        uniprot_accession="P00918",
        llm_model_path=Path.cwd() / "models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        simulation_config=sim_cfg,
        random_seed=seed,
    )

    pocket_coords = boundary.surface_points
    pocket_center = np.mean(boundary.coordinates, axis=0) if boundary.coordinates.size else np.zeros(3)
    feature_extractor: FeatureExtractor = simulation.feature_extractor

    exp = run_experiment_pipeline(
        ligand_id="AZM",
        smiles=ligand_smiles,
        ligand_coords=ligand_coords,
        target_id=pdb_id,
        pocket_coords=pocket_coords,
        pocket_center=pocket_center,
        mode=mode,
        backend_registry=simulation.backend_registry,
        feature_extractor=feature_extractor,
        lambda_cfg=lambda_cfg,
        phase5_params=phase5_params,
        ligc_cfg=ligc_cfg,
        stress_cfg=stress_cfg,
        budget=budget,
        allow_stub=False,
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
        "publishable": True,
    }
    exp.provenance["ramp"] = ramp_params
    exp.provenance["seed"] = seed
    exp.provenance.setdefault("backends", simulation.backend_registry.metadata)
    exp.provenance.setdefault("simulationMode", mode.value)
    exp.provenance.setdefault("stubBackends", False)

    summary = _write_summary(exp, output_dir, ramp_params, lambda_cfg, mode, run_config)

    apo_path = Path(args.apo_summary or (base_dir / "outputs/1CA2_quantum_results/summary.json"))
    if not apo_path.exists():
        raise RuntimeError(f"Apo summary not found at {apo_path}; run apo pipeline first")
    apo_summary = json.loads(apo_path.read_text())
    delta_profile = _compute_delta_profile(summary, apo_summary)
    (output_dir / "delta_profile.json").write_text(json.dumps(delta_profile, indent=2))

    top_modes = sorted(summary["omega_squared"], reverse=True)[:5]
    apo_entropy = apo_summary.get("entropy_density", {}).get("integrated", 0.0)
    bound_entropy = summary.get("entropy_density", {}).get("integrated", 0.0)
    entropy_pct = ((bound_entropy - apo_entropy) / apo_entropy * 100.0) if apo_entropy else float("nan")
    energy_delta = delta_profile["energy_integrated_delta"]
    curvature_delta = delta_profile["curvature_mean_delta"]
    particle_delta = delta_profile["particle_creation_total_change"]
    particle_trend = "increased" if particle_delta > 0 else "decreased" if particle_delta < 0 else "unchanged"

    print("Top 5 excited modes (ω²):", top_modes)
    print(f"Integrated entropy change (%): {entropy_pct:.3f}")
    print(f"Integrated energy change: {energy_delta:.6f}")
    print(f"Curvature mean change: {curvature_delta:.6f}")
    print(f"Particle creation overall {particle_trend} (Δ={particle_delta:.6f})")
    LOGGER.info("Top 5 excited modes (ω²): %s", top_modes)
    LOGGER.info("Integrated entropy change (%%): %.6f", entropy_pct)
    LOGGER.info("Integrated energy change: %.6f", energy_delta)
    LOGGER.info("Curvature mean change: %.6f", curvature_delta)
    LOGGER.info("Particle creation overall %s (Δ=%.6f)", particle_trend, particle_delta)
    LOGGER.info("Run complete; results saved to %s", output_dir)


if __name__ == "__main__":
    main()
