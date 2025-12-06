"""Plugin system for ML potentials used as initial guesses for quantum modes."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


class MLPotentialRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    def register(self, name: str, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._plugins[name] = fn

    def has_plugin(self, name: str) -> bool:
        return name in self._plugins

    def evaluate(self, name: str, coordinates: np.ndarray) -> Optional[np.ndarray]:
        if name not in self._plugins:
            return None
        return self._plugins[name](coordinates)


GLOBAL_ML_POTENTIALS = MLPotentialRegistry()

