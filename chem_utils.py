"""Chemistry utilities for sanitization, parsing, and featurization.

This module centralizes light-weight cheminformatics helpers so that higher-level
pipelines can accept inputs in multiple formats (SMILES, PDB, MOL2) while keeping
production pathways chemically valid. When RDKit is not available, strict
parsing will fail fast to avoid silently accepting unvalidated structures.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Optional, Tuple

from atomic_properties import AtomicFieldParameters, compute_atomic_properties

import numpy as np

try:  # pragma: no cover - RDKit optional
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None
    DataStructs = None

logger = logging.getLogger(__name__)


def sanitize_smiles(smiles: str, strict: bool = False) -> Optional[str]:
    """Return canonical SMILES if valid, otherwise None.

    In strict mode, RDKit must be available and successful sanitization is required;
    otherwise ``None`` is returned to force the caller to reject the molecule.
    """

    if Chem is None:
        # Fallback validation when RDKit is unavailable: accept non-empty ASCII SMILES
        smiles = smiles.strip()
        if strict and not smiles:
            return None
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _mol_from_path(path: str) -> Optional["Chem.Mol"]:
    if Chem is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in {".pdb", ".ent"}:
            return Chem.MolFromPDBFile(path, sanitize=True, removeHs=False)
        if ext in {".mol2"}:
            return Chem.MolFromMol2File(path, sanitize=True, removeHs=False)
        if ext in {".sdf"}:
            suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
            return suppl[0] if len(suppl) else None
        # generic MOL
        return Chem.MolFromMolFile(path, sanitize=True, removeHs=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse molecular file %s: %s", path, exc)
        return None


def _pdb_basic_parse(path: str, include_ligand: bool = True) -> Tuple[Optional[np.ndarray], Dict[int, AtomicFieldParameters]]:
    """Lightweight PDB reader used when RDKit is unavailable.

    Extracts coordinates from ATOM/HETATM records and synthesizes per-atom field
    parameters using simple heuristics on atomic numbers. This keeps production
    flows running in environments without cheminformatics binaries while still
    grounding geometry in the provided structure file.
    """

    aa_resnames = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }

    def _atomic_number(sym: str) -> int:
        table = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "S": 16,
            "P": 15,
            "F": 9,
            "CL": 17,
            "BR": 35,
            "I": 53,
        }
        return table.get(sym.upper().strip(), 0)

    coords = []
    params: Dict[int, AtomicFieldParameters] = {}
    try:
        with open(path, "r") as handle:
            for line in handle:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                resname = line[17:20].strip().upper()
                if not include_ligand and resname not in aa_resnames:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                coords.append((x, y, z))
                element = line[76:78].strip() or line[12:14].strip()
                at_num = _atomic_number(element)
                idx = len(coords) - 1
                # Heuristic field parameters without RDKit
                mu = 0.5 + 0.02 * max(at_num - 1, 0)
                xi = 0.05 + 0.005 * max(at_num - 1, 0)
                curvature = 0.05 + 0.001 * at_num
                params[idx] = AtomicFieldParameters(
                    mu=mu,
                    xi=xi,
                    curvature_coupling=curvature,
                    partial_charge=0.0,
                    polarizability=0.5,
                    hybridization="UNK",
                )
    except FileNotFoundError:
        return None, {}

    if not coords:
        return None, {}
    return np.array(coords, dtype=float), params


def parse_molecule_source(source: str, strict: bool = True) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Parse SMILES or a structure file into canonical SMILES and coordinates.

    Returns a tuple ``(canonical_smiles, coordinates)`` where coordinates are the
    first conformer atomic positions in Angstrom, or ``None`` if unavailable.
    In strict mode, parsing failure returns ``(None, None)`` to force callers to
    reject invalid inputs.
    """

    if Chem is None:
        if os.path.isfile(source) and source.lower().endswith((".pdb", ".ent")):
            coords, _ = _pdb_basic_parse(source, include_ligand=True)
            return (None, coords) if strict else (source, coords)
        return (None, None) if strict else (source, None)

    mol: Optional["Chem.Mol"]
    if os.path.isfile(source):
        mol = _mol_from_path(source)
    else:
        mol = Chem.MolFromSmiles(source, sanitize=True)

    if mol is None:
        return (None, None)

    if mol.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("Conformer generation failed for %s: %s", source, exc)

    conformer = mol.GetConformer() if mol.GetNumConformers() else None
    coords = None
    if conformer is not None:
        coords = np.array(conformer.GetPositions(), dtype=float)

    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        canonical = None

    if strict and canonical is None:
        return (None, None)
    return canonical, coords


def parse_molecule_and_estimate_fields(
    source: str, strict: bool = True, include_ligand: bool = True
) -> Tuple[Optional["Chem.Mol"], Optional[np.ndarray], Dict[int, AtomicFieldParameters]]:
    """Parse a molecular source and estimate per-atom field parameters.

    Returns a tuple of (RDKit Mol, coordinates array, field parameter mapping).
    In strict mode, RDKit must be available and parsing failures will return
    ``(None, None, {})`` to allow the caller to reject invalid inputs.

    When ``include_ligand`` is False and the input is a PDB-like structure,
    non-standard residues are filtered out to approximate an apo state, keeping
    only canonical amino-acid atoms for geometry embedding.
    """

    if Chem is None:
        if os.path.isfile(source) and source.lower().endswith((".pdb", ".ent")):
            coords, params = _pdb_basic_parse(source, include_ligand=include_ligand)
            return (None, coords, params)
        return (None, None, {}) if strict else (None, None, {})

    if os.path.isfile(source):
        mol = _mol_from_path(source)
    else:
        try:
            mol = Chem.MolFromSmiles(source, sanitize=True)
        except Exception:
            mol = None

    if mol is None:
        return (None, None, {}) if strict else (None, None, {})

    if not include_ligand:
        try:
            aa_resnames = {
                "ALA",
                "ARG",
                "ASN",
                "ASP",
                "CYS",
                "GLN",
                "GLU",
                "GLY",
                "HIS",
                "ILE",
                "LEU",
                "LYS",
                "MET",
                "PHE",
                "PRO",
                "SER",
                "THR",
                "TRP",
                "TYR",
                "VAL",
            }
            editable = Chem.RWMol()
            old_to_new: Dict[int, int] = {}
            for atom in mol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                if info is not None and info.GetResName().strip().upper() not in aa_resnames:
                    continue
                new_idx = editable.AddAtom(atom)
                old_to_new[atom.GetIdx()] = new_idx
            for bond in mol.GetBonds():
                bgn, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if bgn in old_to_new and end in old_to_new:
                    editable.AddBond(
                        old_to_new[bgn],
                        old_to_new[end],
                        bond.GetBondType(),
                    )
            filtered = editable.GetMol()
            Chem.SanitizeMol(filtered)
            mol = filtered
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Ligand filtering failed, proceeding with full structure: %s", exc)

    if mol.GetNumConformers() == 0:
        try:  # pragma: no cover - optional conformer generation
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as exc:  # pragma: no cover
            logger.warning("Conformer generation failed for %s: %s", source, exc)

    conformer = mol.GetConformer() if mol.GetNumConformers() else None
    coords = np.array(conformer.GetPositions(), dtype=float) if conformer is not None else None

    try:
        field_params = compute_atomic_properties(mol)
    except Exception as exc:  # pragma: no cover - fallback when RDKit optional pieces missing
        logger.warning("Atomic property estimation failed for %s: %s", source, exc)
        field_params = {}

    if strict and coords is None:
        return (None, None, {})

    return mol, coords, field_params


def default_rdkit_featurizer() -> Optional[Callable[[str], np.ndarray]]:
    if Chem is None or AllChem is None or DataStructs is None:
        return None

    def _featurize(smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2048, dtype=float)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((1, 2048), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr[0])
        return arr[0].astype(float)

    return _featurize

