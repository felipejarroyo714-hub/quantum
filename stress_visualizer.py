"""Utilities to export stress/curvature overlays for visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def write_pymol_script(points: np.ndarray, stresses: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("from pymol import cmd\n")
        for idx, (pt, s) in enumerate(zip(points, stresses)):
            color = min(1.0, max(0.0, float(abs(s) / (np.max(np.abs(stresses)) + 1e-12))))
            handle.write(
                f"cmd.pseudoatom('stress', pos=[{pt[0]:.3f},{pt[1]:.3f},{pt[2]:.3f}], b={s:.3f}, q={color:.3f})\n"
            )
        handle.write("cmd.show('spheres','stress')\n")
    return path


def write_vmd_points(points: np.ndarray, stresses: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for pt, s in zip(points, stresses):
            handle.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {s:.4f}\n")
    return path

