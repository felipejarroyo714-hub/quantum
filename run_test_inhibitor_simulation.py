"""Test harness for simulating a c-Src inhibitor scenario.

This script instantiates :class:`DrugDiscoverySimulation` and drives a single
ligand–target experiment using the existing orchestration logic. It downloads
the 2SRC PDB structure, parses geometry and atomic field parameters, embeds the
geometry into the λ-scale frame, and runs the core experiment pipeline. Results
and diagnostics are written to ``outputs/<pdb_id>_test_run``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict

import numpy as np

from Bio.PDB import PDBList  # type: ignore

from adiabatic_budget import WBSConfig, initialize_budget_from_lambda_wbs
from boundary_geometry_adapter import embed_in_curved_geometry
from chem_utils import parse_molecule_and_estimate_fields, parse_molecule_source
from drug_discovery_simulation import (
    LAMBDA_DILATION,
    SimulationConfig,
    SimulationMode,
    run_experiment_pipeline,
    set_global_simulation_mode,
    LambdaPriorConfig,
)
from experiment_record import ExperimentRecord
from ligc_unified_potential import LigcConfig
from phase5_unification_bridge import Phase5Params
from stress_alignment import StressAlignParams


def _download_pdb(pdb_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pdb_list = PDBList()
    downloaded = pdb_list.retrieve_pdb_file(pdb_id, pdir=str(dest.parent), file_format="pdb")
    dl_path = Path(downloaded)
    if not dl_path.exists():
        raise RuntimeError(f"Failed to download PDB file for {pdb_id}")
    if dest.exists():
        dest.unlink()
    shutil.move(str(dl_path), dest)
    return dest


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def _build_output_payload(exp: ExperimentRecord, boundary_meta: Dict[str, Any]) -> Dict[str, Any]:
    lambda_prior = exp.lambda_prior
    phase5 = exp.phase5_prior
    payload = {
        "provenance": exp.provenance,
        "binding_energy": (exp.docking_result or {}).get("binding_energy"),
        "adiabatic_budget": {
            "J_init": exp.adiabatic_budget_snapshot.J_adia_init if exp.adiabatic_budget_snapshot else None,
            "J_current": exp.adiabatic_budget_snapshot.J_adia_current if exp.adiabatic_budget_snapshot else None,
            "utilization": exp.adiabatic_budget_snapshot.adiabatic_utilization_ratio
            if exp.adiabatic_budget_snapshot
            else None,
        },
        "lambda_descriptors": lambda_prior.descriptor_vector().tolist() if lambda_prior else None,
        "lambda_eigenvalues": lambda_prior.eigenvalues.tolist() if lambda_prior is not None else None,
        "phase5_einstein_residual": getattr(phase5, "einstein_residual", None),
        "ligc": {
            "gamma": getattr(exp.ligc_result, "gamma", None),
            "delta": getattr(exp.ligc_result, "delta", None),
            "variance": getattr(exp.ligc_result, "variance", None),
            "status": getattr(exp.ligc_result, "status", None),
        },
        "stress_alignment": {
            "initial_l2": getattr(exp.stress_alignment, "initial_l2", None),
            "final_l2": getattr(exp.stress_alignment, "final_l2", None),
            "residual_profile": getattr(exp.stress_alignment, "residual_profile", None).tolist()
            if getattr(exp.stress_alignment, "residual_profile", None) is not None
            else None,
        },
        "echo": exp.echo_validation.to_dict() if exp.echo_validation else None,
        "boundary": boundary_meta,
    }
    return payload


def _save_scalar_fields(exp: ExperimentRecord, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_prior = exp.lambda_prior
    np.savez(
        output_dir / "scalar_fields.npz",
        z=lambda_prior.z if lambda_prior is not None else np.array([]),
        density=lambda_prior.density if lambda_prior is not None else np.array([]),
        curvature=lambda_prior.curvature if lambda_prior is not None else np.array([]),
        eigenvalues=lambda_prior.eigenvalues if lambda_prior is not None else np.array([]),
        eigenvectors=lambda_prior.eigenvectors if lambda_prior is not None else np.array([]),
        energy_profile=np.asarray(exp.energy_profile) if exp.energy_profile is not None else np.array([]),
    )


def _save_summary_plot(exp: ExperimentRecord, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore

        lambda_prior = exp.lambda_prior
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        if lambda_prior is not None:
            axes[0].plot(lambda_prior.z, lambda_prior.curvature, label="curvature")
            axes[0].set_title("Curvature vs z")
            axes[1].plot(lambda_prior.z, lambda_prior.density, label="density", color="orange")
            axes[1].set_title("Density vs z")
            axes[2].bar(range(len(lambda_prior.eigenvalues)), lambda_prior.eigenvalues)
            axes[2].set_title("Mode spectrum (ω²)")
        for ax in axes:
            ax.grid(True)
        plt.tight_layout()
        fig.savefig(output_dir / "summary_plot.png", dpi=200)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting optional
        logging.warning("Failed to create summary plot: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a c-Src inhibitor simulation test")
    parser.add_argument("--pdb_id", default="2SRC", help="PDB identifier to download")
    parser.add_argument("--ligand_smiles", default="CC(C)NC1=NC=NC2=C1N=CN2", help="Ligand SMILES for the experiment")
    parser.add_argument("--mode", choices=[m.value for m in SimulationMode], default=SimulationMode.DEBUG_SYNTHETIC.value)
    parser.add_argument("--allow_stub", action="store_true", help="Allow stub backends (debug/integration only)")
    parser.add_argument("--output_dir", default=None, help="Output directory for artifacts")
    parser.add_argument(
        "--llm_model_path",
        default=os.path.join(os.getcwd(), "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        help="Path to TinyLlama GGUF model",
    )
    parser.add_argument("--epsilon", type=float, default=0.05, help="Embedding epsilon for curved geometry")
    parser.add_argument("--r0", type=float, default=1.5, help="Reference radius for geometry embedding")
    parser.add_argument("--ramp", default="linear", help="Ramp type for evolution metadata")
    parser.add_argument("--ramp_time", type=float, default=8.0, help="Ramp time metadata")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of evolution steps metadata")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step metadata")
    args = parser.parse_args()

    pdb_id = args.pdb_id.upper()
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir or base_dir / f"outputs/{pdb_id}_test_run")
    log_path = output_dir / "simulation.log"
    _configure_logging(log_path)

    with open(log_path, "a", encoding="utf-8") as log_handle, redirect_stdout(log_handle), redirect_stderr(log_handle):
        logging.info("Starting test run for %s", pdb_id)
        set_global_simulation_mode(SimulationMode(args.mode), allow_stub=args.allow_stub)

        pdb_path = (base_dir / "data" / "pdb" / f"{pdb_id}.pdb").resolve()
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        _download_pdb(pdb_id, pdb_path)

        mol, coords, field_params = parse_molecule_and_estimate_fields(str(pdb_path), strict=False)
        boundary = embed_in_curved_geometry(mol, coords if coords is not None else np.zeros((0, 3)), args.epsilon, args.r0, LAMBDA_DILATION)

        lambda_cfg = LambdaPriorConfig(lam=LAMBDA_DILATION, z_min=-5.0, z_max=5.0, num_z=96)
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
        )

        from drug_discovery_simulation import DrugDiscoverySimulation

        simulation = DrugDiscoverySimulation(
            pdb_id=pdb_id,
            target_query="c-src kinase",
            uniprot_accession="P12931",
            llm_model_path=Path(args.llm_model_path),
            simulation_config=sim_cfg,
        )

        ligand_smiles, ligand_coords = parse_molecule_source(args.ligand_smiles, strict=False)
        if ligand_smiles is None:
            raise RuntimeError("Ligand SMILES could not be sanitized for test run")

        pocket_coords = boundary.surface_points
        pocket_center = np.mean(boundary.coordinates, axis=0) if boundary.coordinates.size else np.zeros(3)

        exp = run_experiment_pipeline(
            ligand_id=ligand_smiles,
            smiles=ligand_smiles,
            ligand_coords=ligand_coords if ligand_coords is not None else np.zeros((0, 3)),
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

        exp.provenance["benchmarkDataset"] = None
        exp.provenance["ramp"] = {
            "type": args.ramp,
            "ramp_time": args.ramp_time,
            "n_steps": args.n_steps,
            "dt": args.dt,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        payload = _build_output_payload(exp, boundary.metadata or {})
        payload["field_parameter_summary"] = {k: v.__dict__ for k, v in field_params.items()}
        payload["energy_profile"] = exp.energy_profile.tolist() if exp.energy_profile is not None else None
        (output_dir / "log.json").write_text(json.dumps(payload, indent=2))

        _save_scalar_fields(exp, output_dir)
        _save_summary_plot(exp, output_dir)

        trajectory = {
            "md_time": exp.md_result.get("time") if exp.md_result else None,
            "md_bound_fraction": exp.md_result.get("bound_fraction") if exp.md_result else None,
            "stress_residual_profile": payload.get("stress_alignment", {}).get("residual_profile"),
        }
        (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

        with open(output_dir / "experiment_record.json", "w", encoding="utf-8") as handle:
            handle.write(json.dumps(exp.to_dict(), indent=2, default=str))

        with open(output_dir / "final_state.pkl", "wb") as handle:
            pickle.dump(simulation, handle)

        logging.info("Test run complete; artifacts stored in %s", output_dir)


if __name__ == "__main__":
    main()

