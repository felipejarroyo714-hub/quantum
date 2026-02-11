"""Benchmark simulated energies against reference DFT calculations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def compare_to_dft(simulated: List[float], reference: List[float]) -> Dict[str, float]:
    sim = np.asarray(simulated, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if sim.size != ref.size:
        raise ValueError("Simulated and reference arrays must be the same length")
    diff = sim - ref
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def write_report(simulated: List[float], reference: List[float], path: Path) -> Path:
    metrics = compare_to_dft(simulated, reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


if __name__ == "__main__":  # pragma: no cover
    # Example usage with dummy values
    sim = [-10.1, -5.2, -7.3]
    ref = [-10.0, -5.0, -7.0]
    report = write_report(sim, ref, Path("outputs/dft_comparison.json"))
    print(f"Wrote {report}")

