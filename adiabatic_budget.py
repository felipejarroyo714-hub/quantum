"""Adiabatic invariant budgeting utilities for λ-WBS campaigns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import copy

import numpy as np


@dataclass
class WBSConfig:
    lam: float
    epsilon_ladder: List[float]
    z_extent: float = 10.0


@dataclass
class AdiabaticBudget:
    J_adia_init: float
    J_adia_current: float
    eps_schedule: List[float]
    exotic_energy_trace: List[float] = field(default_factory=list)
    rg_stability_flag: bool = True
    adiabatic_utilization_ratio: float = 0.0


def _compute_initial_budget(cfg: WBSConfig) -> float:
    ladder_energy = np.sum(np.abs(cfg.epsilon_ladder))
    return float(cfg.lam * (1.0 + ladder_energy))


def initialize_budget_from_lambda_wbs(cfg: WBSConfig) -> AdiabaticBudget:
    J0 = _compute_initial_budget(cfg)
    budget = AdiabaticBudget(J_adia_init=J0, J_adia_current=J0, eps_schedule=list(cfg.epsilon_ladder))
    return budget


def update_budget_with_experiment(budget: AdiabaticBudget, experiment: "ExperimentRecord") -> None:
    exotic_increment = 0.0

    if getattr(experiment, "stress_alignment", None):
        exotic_increment += float(max(0.0, experiment.stress_alignment.final_l2))

    if getattr(experiment, "ligc_result", None):
        exotic_increment += float(max(0.0, experiment.ligc_result.variance))
        if experiment.ligc_result.gamma_deviation is not None:
            exotic_increment += float(abs(experiment.ligc_result.gamma_deviation)) * 0.1
        if experiment.ligc_result.delta_deviation is not None:
            exotic_increment += float(abs(experiment.ligc_result.delta_deviation)) * 0.1

    phase5_prior = getattr(experiment, "phase5_prior", None)
    if phase5_prior is not None:
        exotic_increment += float(max(0.0, phase5_prior.einstein_residual.get("l2", 0.0)))

    echo_val = getattr(experiment, "echo_validation", None)
    if echo_val is not None and echo_val.classification == "echo-failed":
        exotic_increment *= 1.5

    budget.exotic_energy_trace.append(exotic_increment)
    budget.J_adia_current = max(0.0, budget.J_adia_current - exotic_increment)
    budget.adiabatic_utilization_ratio = float(
        1.0 - budget.J_adia_current / max(budget.J_adia_init, 1e-12)
    )

    if budget.J_adia_current < 0.1 * budget.J_adia_init:
        budget.rg_stability_flag = False
    experiment.adiabatic_budget_snapshot = copy.deepcopy(budget)
