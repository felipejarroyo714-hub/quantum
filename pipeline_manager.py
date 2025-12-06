"""Pipeline manager orchestrating parsing → geometry → simulation → validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from chem_utils import parse_molecule_source
from drug_discovery_simulation import DrugDiscoverySimulation, SimulationMode, SimulationConfig
from experiment_record import ExperimentRecord


class PipelineManager:
    def __init__(self, simulation: DrugDiscoverySimulation) -> None:
        self.simulation = simulation

    def run_from_source(self, source: str, ligand_id: Optional[str] = None) -> ExperimentRecord:
        smiles, coords = parse_molecule_source(source, strict=True)
        if smiles is None:
            raise RuntimeError(f"Invalid molecular source: {source}")
        return self.simulation.run_single_experiment(smiles, ligand_id=ligand_id, coordinates=coords)

    def run_from_config(self, config_path: Path) -> Dict[str, Any]:
        cfg = json.loads(Path(config_path).read_text())
        sources = cfg.get("molecules", [])
        results = []
        for idx, src in enumerate(sources):
            rec = self.run_from_source(src, ligand_id=str(idx))
            results.append(rec.to_dict())
        return {"mode": self.simulation.simulation_config.mode.value, "results": results}


def build_simulation_from_yaml(yaml_path: Path) -> DrugDiscoverySimulation:
    import yaml  # pragma: no cover - optional dependency

    cfg_dict = yaml.safe_load(Path(yaml_path).read_text())
    sim_cfg = SimulationConfig(**cfg_dict)
    return DrugDiscoverySimulation(sim_cfg)

