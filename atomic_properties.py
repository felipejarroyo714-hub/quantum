"""Atomic property estimation utilities.

This module computes partial charges, polarizabilities, and hybridization
states using RDKit-based heuristics or semi-empirical surrogates when
available. Outputs map into field parameters that can be consumed by the
λ-scale geometry and Klein-Gordon field builders.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None

logger = logging.getLogger(__name__)


@dataclass
class AtomicFieldParameters:
    mu: float
    xi: float
    curvature_coupling: float
    partial_charge: float
    polarizability: float
    hybridization: str


def _estimate_polarizability(atom: "Chem.Atom") -> float:
    # Simple electronegativity-based heuristic; replace with ML/SEQM as needed
    en = Chem.GetPeriodicTable().GetElectronegativity(atom.GetAtomicNum()) if Chem else 0.0
    return max(0.1, 5.0 - 0.5 * en)


def compute_atomic_properties(mol: "Chem.Mol") -> Dict[int, AtomicFieldParameters]:
    """Compute per-atom field parameters from an RDKit molecule.

    Returns a mapping from atom index to :class:`AtomicFieldParameters`. Raises
    RuntimeError if RDKit is unavailable.
    """

    if Chem is None or AllChem is None:
        raise RuntimeError("RDKit is required for atomic property estimation")

    props: Dict[int, AtomicFieldParameters] = {}
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as exc:  # pragma: no cover
        logger.warning("Gasteiger charges failed: %s", exc)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        charge = float(atom.GetDoubleProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0
        polar = _estimate_polarizability(atom)
        hybrid = str(atom.GetHybridization())
        # Map to field couplings; simple proportional models for now
        mu = 0.5 + 0.1 * abs(charge)
        xi = 0.05 * (atom.GetDegree() + 1)
        curvature = 0.1 * polar
        props[idx] = AtomicFieldParameters(
            mu=mu,
            xi=xi,
            curvature_coupling=curvature,
            partial_charge=charge,
            polarizability=polar,
            hybridization=hybrid,
        )
    return props


def summarize_field_parameters(params: Dict[int, AtomicFieldParameters]) -> Dict[str, float]:
    if not params:
        return {}
    mu_vals = np.array([p.mu for p in params.values()], dtype=float)
    xi_vals = np.array([p.xi for p in params.values()], dtype=float)
    pol_vals = np.array([p.polarizability for p in params.values()], dtype=float)
    return {
        "mu_mean": float(mu_vals.mean()),
        "xi_mean": float(xi_vals.mean()),
        "polarizability_mean": float(pol_vals.mean()),
        "mu_std": float(mu_vals.std()),
        "xi_std": float(xi_vals.std()),
    }

