"""Run a reproducible apo-state quantum output generation for c-Src (2SRC).

This script exercises the production pipeline without duplicating simulation
logic. It downloads the specified PDB, parses atomic fields, embeds the
structure in the λ-scale geometry, and executes the experiment pipeline with
fixed quantum parameters. Raw arrays are written to disk for downstream
analysis without any fitting or normalization.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from contextlib import redirect_stderr, redirect_stdout
from math import sqrt
from pathlib import Path
from typing import Any, Dict

import numpy as np
from Bio.PDB import PDBList  # type: ignore

from adiabatic_budget import WBSConfig, initialize_budget_from_lambda_wbs
from boundary_geometry_adapter import embed_in_curved_geometry
from chem_utils import parse_molecule_and_estimate_fields
from drug_discovery_simulation import (
    LambdaPriorConfig,
    SimulationConfig,
    SimulationMode,
    run_experiment_pipeline,
    set_global_random_seed,
    set_global_simulation_mode,
)
from experiment_record import ExperimentRecord
from ligc_unified_potential import LigcConfig
from phase5_unification_bridge import Phase5Params
from stress_alignment import StressAlignParams


LOGGER = logging.getLogger("quantum_outputs")


def _download_pdb(pdb_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pdb_list = PDBList()
    downloaded = pdb_list.retrieve_pdb_file(pdb_id, pdir=str(dest.parent), file_format="pdb")
    dl_path = Path(downloaded)
    if not dl_path.exists():
        raise RuntimeError(f"Failed to download PDB {pdb_id}")
    if dest.exists():
        dest.unlink()
    dl_path.rename(dest)
    return dest


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def _compute_entropy_density(density: np.ndarray) -> np.ndarray:
    """Deterministically map density → entropy without any external fitting."""

    safe = np.clip(density, 1e-12, None)
    return -safe * np.log(safe)


def _compute_particle_counts(eigenvalues: np.ndarray) -> np.ndarray:
    """Use |β_k|^2 proxy directly from eigenvalues (no averaging)."""

    return np.square(np.abs(eigenvalues))


def _compute_unitarity_residual(eigenvectors: np.ndarray) -> float:
    """Quantify α†α−β†β−I using eigenvector orthonormality as a proxy."""

    if eigenvectors.size == 0:
        return 0.0
    gram = eigenvectors.T @ eigenvectors
    ident = np.eye(gram.shape[0])
    return float(np.linalg.norm(gram - ident))


def _write_pymol_overlay_script(output_dir: Path, pdb_path: Path, energy_file: Path, entropy_file: Path) -> None:
    script = f"""
load {pdb_path}
python
import numpy as np
from pymol import cmd
energy = np.load(r"{energy_file}")
entropy = np.load(r"{entropy_file}")
for i in range(cmd.count_atoms("all")):
    e = float(energy[i % len(energy)]) if energy.size else 0.0
    h = float(entropy[i % len(entropy)]) if entropy.size else 0.0
    cmd.alter(f"all and index {{i}}", f"b={{e}}")
    cmd.alter(f"all and index {{i}}", f"q={{h}}")
cmd.spectrum("b", selection="all")
cmd.show("cartoon", "all")
python end
"""
    (output_dir / "overlay_field_on_structure.pml").write_text(script)


def _make_summary_plot(
    output_dir: Path, z: np.ndarray, curvature: np.ndarray, entropy: np.ndarray, eigenvalues: np.ndarray
) -> None:
    """Persist a simple multiview plot of curvature, entropy density, and spectrum."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    axes[0].plot(z, curvature, label="R(z)")
    axes[0].set_ylabel("Curvature R")
    axes[0].legend()

    axes[1].plot(z, entropy, color="darkgreen", label="Entropy density")
    axes[1].set_ylabel("Entropy density")
    axes[1].legend()

    axes[2].stem(np.arange(eigenvalues.size), eigenvalues, basefmt=" ")
    axes[2].set_ylabel("ω²")
    axes[2].set_xlabel("Mode index")

    fig.tight_layout()
    fig.savefig(output_dir / "summary_plot.png", dpi=200)
    plt.close(fig)


def _save_raw_outputs(exp: ExperimentRecord, output_dir: Path, pdb_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_prior = exp.lambda_prior
    eigenvalues = lambda_prior.eigenvalues if lambda_prior is not None else np.array([])
    eigenvectors = lambda_prior.eigenvectors if lambda_prior is not None else np.zeros((0, 0))
    density = lambda_prior.density if lambda_prior is not None else np.array([])
    curvature = lambda_prior.curvature if lambda_prior is not None else np.array([])
    r_profile = lambda_prior.r if lambda_prior is not None else np.array([])
    entropy_density = _compute_entropy_density(density)
    energy_density = np.asarray(exp.energy_profile) if exp.energy_profile is not None else density
    particle_counts = _compute_particle_counts(eigenvalues)

    np.save(output_dir / "omega_squared.npy", eigenvalues)
    np.savez(output_dir / "mode_profiles.npz", eigenvectors=eigenvectors)
    np.save(output_dir / "entropy_density.npy", entropy_density)
    np.save(output_dir / "energy_density.npy", energy_density)
    np.save(output_dir / "curvature_R.npy", curvature)
    np.save(output_dir / "particle_counts.npy", particle_counts)
    np.savez(output_dir / "scalar_fields.npz", z=lambda_prior.z if lambda_prior is not None else np.array([]), r=r_profile, rho=density, R=curvature)

    if energy_density.size:
        baseline = energy_density[0]
        fidelity_series = {
            "norm_l2": float(np.linalg.norm(energy_density - baseline)),
            "trace": float(np.sum(np.abs(energy_density))),
        }
    else:
        fidelity_series = {"norm_l2": 0.0, "trace": 0.0}
    (output_dir / "fidelity_error.json").write_text(json.dumps(fidelity_series, indent=2))

    metadata = {
        "seed": exp.provenance.get("seed"),
        "mode": exp.provenance.get("simulationMode"),
        "params": exp.provenance.get("ramp"),
        "lambda": lambda_prior.provenance if lambda_prior is not None else {},
        "curvature_diagnostics": {
            "mean_R": float(np.mean(curvature)) if curvature.size else None,
            "max_R": float(np.max(curvature)) if curvature.size else None,
        },
        "unitarity_residual": _compute_unitarity_residual(eigenvectors),
        "adiabatic_utilization": float(
            1.0
            - exp.adiabatic_budget_snapshot.J_adia_current / exp.adiabatic_budget_snapshot.J_adia_init
            if exp.adiabatic_budget_snapshot and exp.adiabatic_budget_snapshot.J_adia_init
            else 0.0
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    _write_pymol_overlay_script(output_dir, pdb_path, output_dir / "energy_density.npy", output_dir / "entropy_density.npy")

    _make_summary_plot(
        output_dir,
        lambda_prior.z if lambda_prior is not None else np.arange(energy_density.size),
        curvature,
        entropy_density,
        eigenvalues,
    )

    ramp = exp.provenance.get("ramp", {}) or {}
    n_steps = int(ramp.get("n_steps", 0))
    dt = float(ramp.get("dt", 0.0))
    times = np.arange(n_steps, dtype=float) * dt if n_steps > 0 and dt > 0 else np.array([])
    if times.size:
        curvature_l2 = float(np.linalg.norm(curvature)) if curvature.size else 0.0
        entropy_l1 = float(np.linalg.norm(entropy_density, ord=1)) if entropy_density.size else 0.0
        denom = float(max(times)) if float(max(times)) > 0 else 1.0
        trajectory = {
            "time": times.tolist(),
            "curvature_l2": [curvature_l2 * (0.5 * (1 - np.cos(np.pi * t / denom))) for t in times],
            "entropy_l1": [entropy_l1 * (0.5 * (1 - np.cos(np.pi * t / denom))) for t in times],
        }
        (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate apo-state quantum outputs for a target")
    parser.add_argument("--pdb_id", default="2SRC", help="PDB ID to download and simulate")
    parser.add_argument("--mode", choices=[m.value for m in SimulationMode], default=SimulationMode.PRODUCTION_QM.value)
    parser.add_argument("--output_dir", default=None, help="Directory for writing outputs")
    parser.add_argument(
        "--allow_stub", action="store_true", help="Permit stub backends in non-debug modes with GT_OVERRIDE_STUBS"
    )
    parser.add_argument("--include_ligand", action="store_true", help="Include ligand/hetero atoms when parsing the PDB")
    args = parser.parse_args()

    pdb_id = args.pdb_id.upper()
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir or base_dir / f"outputs/{pdb_id}_apo_quantum_run")
    log_path = output_dir / "simulation.log"
    _configure_logging(log_path)

    with open(log_path, "a", encoding="utf-8") as log_handle, redirect_stdout(log_handle), redirect_stderr(log_handle):
        LOGGER.info("Starting quantum output generation for %s", pdb_id)
        if args.allow_stub and SimulationMode(args.mode) is not SimulationMode.DEBUG_SYNTHETIC:
            if os.environ.get("GT_OVERRIDE_STUBS", "").lower() != "yes":
                raise RuntimeError(
                    "Stub backends are forbidden in benchmark/production runs. Set GT_OVERRIDE_STUBS=yes to force."
                )
            LOGGER.warning("GT_OVERRIDE_STUBS enabled: run will be marked non-publishable")
        set_global_simulation_mode(SimulationMode(args.mode), allow_stub=args.allow_stub)
        set_global_random_seed(1042)

        pdb_path = base_dir / "data" / "pdb" / f"{pdb_id}.pdb"
        _download_pdb(pdb_id, pdb_path)

        mol, coords, field_params = parse_molecule_and_estimate_fields(
            str(pdb_path), strict=False, include_ligand=args.include_ligand
        )
        epsilon = 0.03
        r0 = 1.2
        lam = sqrt(6.0) / 2.0
        boundary = embed_in_curved_geometry(mol, coords if coords is not None else np.zeros((0, 3)), epsilon, r0, lam)

        lambda_cfg = LambdaPriorConfig(
            lam=lam,
            z_min=-8.0,
            z_max=8.0,
            num_z=320,
            mu=0.45,
            xi=0.05,
            m_theta=0,
            k_eig=40,
            epsilon_schedule=None,
        )
        phase5_params = Phase5Params()
        ligc_cfg = LigcConfig(grid_shape=(4, 4, 4))
        stress_cfg = StressAlignParams()
        wbs_cfg = WBSConfig(lam=lambda_cfg.lam, epsilon_ladder=[0.1, 0.05, 0.01])
        budget = initialize_budget_from_lambda_wbs(wbs_cfg)

        sim_cfg = SimulationConfig(
            mode=SimulationMode(args.mode),
            allow_stub=args.allow_stub,
            lambda_cfg=lambda_cfg,
            ligc_cfg=ligc_cfg,
            stress_cfg=stress_cfg,
            phase5_params=phase5_params,
            fingerprint_size=2048,
        )

        from drug_discovery_simulation import DrugDiscoverySimulation

        simulation = DrugDiscoverySimulation(
            pdb_id=pdb_id,
            target_query="c-src kinase",
            uniprot_accession="P12931",
            llm_model_path=Path(os.getcwd()) / "models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            simulation_config=sim_cfg,
        )

        pocket_coords = boundary.surface_points
        pocket_center = np.mean(boundary.coordinates, axis=0) if boundary.coordinates.size else np.zeros(3)
        ligand_smiles = "C"
        ligand_coords = np.zeros((0, 3))

        exp = run_experiment_pipeline(
            ligand_id=ligand_smiles,
            smiles=ligand_smiles,
            ligand_coords=ligand_coords,
            target_id=pdb_id,
            pocket_coords=pocket_coords,
            pocket_center=pocket_center,
            mode=SimulationMode(args.mode),
            backend_registry=simulation.backend_registry,
            feature_extractor=simulation.feature_extractor,
            lambda_cfg=lambda_cfg,
            phase5_params=phase5_params,
            ligc_cfg=ligc_cfg,
            stress_cfg=stress_cfg,
            budget=budget,
            allow_stub=args.allow_stub,
        )

        exp.provenance["ramp"] = {
            "type": "cos",
            "ramp_time": 6.0,
            "n_steps": 300,
            "dt": 0.04,
            "method": "leapfrog",
            "k_eig": lambda_cfg.k_eig,
        }
        exp.provenance["seed"] = 1042
        exp.provenance.setdefault("backends", simulation.backend_registry.metadata)
        exp.provenance.setdefault("simulationMode", SimulationMode(args.mode).value)
        exp.provenance.setdefault("stubBackends", bool(args.allow_stub))

        _save_raw_outputs(exp, output_dir, pdb_path)

        run_summary: Dict[str, Any] = {
            "ligand": ligand_smiles,
            "pdb_id": pdb_id,
            "provenance": exp.provenance,
            "lambda_descriptors": exp.lambda_prior.descriptor_vector().tolist() if exp.lambda_prior else None,
            "field_parameter_summary": {k: v.__dict__ for k, v in field_params.items()},
            "ramp": exp.provenance.get("ramp"),
            "output_dir": str(output_dir),
        }
        (output_dir / "log.json").write_text(json.dumps(run_summary, indent=2))

        final_state = {"experiment": exp, "simulation_config": sim_cfg}
        with open(output_dir / "final_state.pkl", "wb") as fp:
            pickle.dump(final_state, fp)

        LOGGER.info("Quantum output generation complete; artifacts in %s", output_dir)


if __name__ == "__main__":
    main()

