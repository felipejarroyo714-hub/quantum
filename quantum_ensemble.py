"""Monte Carlo / Langevin wrappers around quantum mode evolution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class EnsembleConfig:
    temperature: float = 300.0
    steps: int = 64
    step_size: float = 0.01


def langevin_sample(modes: np.ndarray, energy_fn: Callable[[np.ndarray], float], cfg: EnsembleConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    samples = []
    energies = []
    state = np.array(modes, dtype=float)
    beta = 1.0 / max(cfg.temperature * 1.380649e-23, 1e-9)
    for _ in range(cfg.steps):
        grad = np.gradient(state, axis=0)
        noise = rng.normal(scale=np.sqrt(2.0 * cfg.step_size / beta), size=state.shape)
        state = state - cfg.step_size * grad + noise
        e = energy_fn(state)
        samples.append(state.copy())
        energies.append(e)
    weights = np.exp(-beta * (np.array(energies) - np.min(energies)))
    weights /= weights.sum() if weights.sum() else 1.0
    return np.array(samples), weights

