"""Lambda-scale geometry priors for ligand/target complexes."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from kg_scale_invariant_metric import (
    FieldParams,
    GeometryParams,
    build_kg_operator,
    compute_modes,
    integrate_profile,
    normalize_on_z,
)
from atomic_properties import compute_atomic_properties, summarize_field_parameters
from boundary_geometry_adapter import embed_in_curved_geometry

logger = logging.getLogger(__name__)


@dataclass
class LambdaPriorConfig:
    """Configuration for building λ-geometry priors."""

    lam: float
    z_min: float
    z_max: float
    num_z: int
    mu: float = 0.5
    xi: float = 0.0
    m_theta: int = 0
    k_eig: int = 8
    epsilon: float = 0.0
    epsilon_schedule: Optional[Sequence[float]] = None

    def geometry_params(self) -> GeometryParams:
        return GeometryParams(
            lam=self.lam, z_min=self.z_min, z_max=self.z_max, num_z=self.num_z, epsilon=self.epsilon
        )

    def field_params(self) -> FieldParams:
        return FieldParams(mu=self.mu, xi=self.xi, m_theta=self.m_theta, k_eig=self.k_eig)


@dataclass
class MoleculeRecord:
    """Lightweight representation of a ligand with optional coordinates."""

    ligand_id: str
    smiles: str
    coordinates: Optional[np.ndarray] = None
    lambda_prior: Optional["LambdaPrior"] = None


@dataclass
class TargetRecord:
    """Representation of a protein target or binding pocket."""

    target_id: str
    pocket_center: Optional[np.ndarray] = None
    pocket_coordinates: Optional[np.ndarray] = None


@dataclass
class LambdaPrior:
    """Container for λ-shell derived descriptors and diagnostics."""

    z: np.ndarray
    r: np.ndarray
    density: np.ndarray
    curvature: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    descriptors: np.ndarray
    status: str = "ok"
    particle_creation: float | None = None
    provenance: dict = field(default_factory=dict)

    def descriptor_vector(self) -> np.ndarray:
        return np.asarray(self.descriptors, dtype=float)


def _build_reaction_coordinate(
    ligand_record: MoleculeRecord, target_record: TargetRecord, num_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    if ligand_record.coordinates is None and target_record.pocket_coordinates is None:
        z = np.linspace(-1.0, 1.0, num_bins)
        density = np.zeros_like(z)
        return z, density

    ligand_coords = (
        ligand_record.coordinates
        if ligand_record.coordinates is not None and ligand_record.coordinates.size
        else np.zeros((0, 3))
    )
    pocket_coords = (
        target_record.pocket_coordinates
        if target_record.pocket_coordinates is not None and target_record.pocket_coordinates.size
        else np.zeros((0, 3))
    )
    origin = target_record.pocket_center
    if origin is None:
        combined = np.vstack([c for c in (ligand_coords, pocket_coords) if c.size]) if (
            ligand_coords.size or pocket_coords.size
        ) else np.zeros((1, 3))
        origin = np.mean(combined, axis=0)

    all_points = np.vstack([p for p in (ligand_coords, pocket_coords) if p.size]) if (
        ligand_coords.size or pocket_coords.size
    ) else np.zeros((1, 3))
    distances = np.linalg.norm(all_points - origin, axis=1)
    bins = np.linspace(distances.min(initial=0.0), distances.max(initial=1.0) + 1e-6, num_bins + 1)
    hist, edges = np.histogram(distances, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return centers, hist


def _build_lambda_descriptors(z: np.ndarray, rho: np.ndarray, R: np.ndarray, evals: np.ndarray) -> np.ndarray:
    curvature_stats = np.array([np.min(R), np.mean(R), np.max(R)], dtype=float)
    density_dev = float(np.linalg.norm(rho - 1.0)) if rho.size else 0.0
    spectral_signature = np.sort(np.sqrt(np.abs(evals) + 1e-12))[:8]
    descriptor = np.concatenate([spectral_signature, curvature_stats, np.array([density_dev])])
    assert descriptor.ndim == 1
    return descriptor


def build_lambda_prior_for_complex(
    ligand_record: MoleculeRecord, target_record: TargetRecord, cfg: LambdaPriorConfig
) -> LambdaPrior:
    try:
        # Enrich geometry from raw coordinates into λ-aligned boundary surfaces
        rdkit_mol = None
        try:  # pragma: no cover - optional
            from rdkit import Chem

            rdkit_mol = Chem.MolFromSmiles(ligand_record.smiles) if ligand_record.smiles else None
        except Exception:
            rdkit_mol = None

        coords = ligand_record.coordinates if (ligand_record.coordinates is not None and ligand_record.coordinates.size) else np.zeros((0, 3))
        boundary = embed_in_curved_geometry(rdkit_mol, coords)

        z_embed, _ = _build_reaction_coordinate(ligand_record, target_record, cfg.num_z)
        geo = cfg.geometry_params()
        geo.z_min = float(z_embed.min(initial=geo.z_min)) if z_embed.size else geo.z_min
        geo.z_max = float(z_embed.max(initial=geo.z_max)) if z_embed.size else geo.z_max
        z_grid, r_profile, rho_profile, curvature = integrate_profile(geo)
        field_params = cfg.field_params()

        # If atomic properties are available, adjust μ/ξ and anisotropy
        anisotropy = None
        try:
            if rdkit_mol is not None:
                atomic_props = compute_atomic_properties(rdkit_mol)
                stats = summarize_field_parameters(atomic_props)
                field_params.mu = max(field_params.mu, stats.get("mu_mean", field_params.mu))
                field_params.xi = max(field_params.xi, stats.get("xi_mean", field_params.xi))
                anisotropy = np.interp(
                    z_grid,
                    np.linspace(z_grid.min(), z_grid.max(), len(boundary.radii)),
                    1.0 + 0.1 * (boundary.radii / np.clip(boundary.radii.max(initial=1.0), 1.0, None)),
                )
        except Exception as exc:  # pragma: no cover
            logger.debug("Atomic property adjustment skipped: %s", exc)

        operator, _ = build_kg_operator(
            z_grid, r_profile, curvature, field_params, anisotropy=anisotropy
        )
        eigenvalues, eigenvectors = compute_modes(operator, k=min(field_params.k_eig, len(z_grid) - 2))
        eigenvectors = np.column_stack([normalize_on_z(z_grid, eigenvectors[:, i]) for i in range(eigenvectors.shape[1])])
        descriptors = _build_lambda_descriptors(z_grid, rho_profile, curvature, eigenvalues)
        particle_creation = None
        if cfg.epsilon_schedule:
            particle_creation = float(np.var(cfg.epsilon_schedule) * np.mean(np.abs(eigenvalues)))
        prior = LambdaPrior(
            z=z_grid,
            r=r_profile,
            density=rho_profile,
            curvature=curvature,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            descriptors=descriptors,
            particle_creation=particle_creation,
            provenance={"lambda": cfg.lam, "k_eig": field_params.k_eig, "boundary_samples": len(boundary.surface_points)},
        )
        ligand_record.lambda_prior = prior
        return prior
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to build lambda prior: %s", exc)
        descriptors = np.zeros(12, dtype=float)
        prior = LambdaPrior(
            z=np.zeros(cfg.num_z, dtype=float),
            r=np.zeros(cfg.num_z, dtype=float),
            density=np.zeros(cfg.num_z, dtype=float),
            curvature=np.zeros(cfg.num_z, dtype=float),
            eigenvalues=np.zeros(cfg.k_eig, dtype=float),
            eigenvectors=np.zeros((cfg.num_z, cfg.k_eig), dtype=float),
            descriptors=descriptors,
            status="failed",
            provenance={"error": str(exc)},
        )
        ligand_record.lambda_prior = prior
        return prior


def get_lambda_descriptor_vector(record: Union[MoleculeRecord, LambdaPrior, None]) -> np.ndarray:
    if record is None:
        return np.zeros(12, dtype=float)
    if isinstance(record, LambdaPrior):
        return record.descriptor_vector()
    if isinstance(record, MoleculeRecord):
        if record.lambda_prior is None:
            return np.zeros(12, dtype=float)
        return record.lambda_prior.descriptor_vector()
    return np.zeros(12, dtype=float)
