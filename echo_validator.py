"""Echo-style validation utilities for simulated vs empirical profiles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EchoProfile:
    time: np.ndarray
    amplitude: np.ndarray
    uncertainty: Optional[np.ndarray] = None


@dataclass
class EchoCompareConfig:
    tolerance_l2: float = 0.2
    tolerance_phase: float = 0.1


@dataclass
class EchoCompareResult:
    l2_distance: float
    cross_correlation_peak: float
    phase_shift: float
    classification: str


def _phase_difference(sim: np.ndarray, ref: np.ndarray) -> float:
    sim_fft = np.fft.rfft(sim)
    ref_fft = np.fft.rfft(ref)
    phase_sim = np.angle(sim_fft[1:])
    phase_ref = np.angle(ref_fft[1:])
    if phase_sim.size == 0:
        return 0.0
    return float(np.mean(np.abs(phase_sim - phase_ref)))


def compare_echo_profiles(sim_profile: EchoProfile, ref_profile: EchoProfile, cfg: EchoCompareConfig) -> EchoCompareResult:
    sim_amp = np.asarray(sim_profile.amplitude, dtype=float)
    ref_amp = np.asarray(ref_profile.amplitude, dtype=float)
    min_len = min(sim_amp.size, ref_amp.size)
    if min_len == 0:
        return EchoCompareResult(0.0, 0.0, 0.0, classification="echo-ambiguous")
    sim_amp = sim_amp[:min_len]
    ref_amp = ref_amp[:min_len]
    l2 = float(np.linalg.norm(sim_amp - ref_amp) / np.sqrt(min_len))
    correlation = np.correlate(sim_amp, ref_amp, mode="valid")
    phase_delta = _phase_difference(sim_amp, ref_amp)
    peak = float(np.max(correlation)) if correlation.size else 0.0
    if l2 <= cfg.tolerance_l2 and phase_delta <= cfg.tolerance_phase:
        classification = "echo-validated"
    elif l2 <= 2 * cfg.tolerance_l2:
        classification = "echo-ambiguous"
    else:
        classification = "echo-failed"
    return EchoCompareResult(l2_distance=l2, cross_correlation_peak=peak, phase_shift=phase_delta, classification=classification)
