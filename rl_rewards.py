"""Reward primitives for the lambda-aware drug discovery RL system."""

from __future__ import annotations

import numpy as np


class RewardPrimitives:
    """Collection of bounded reward primitives shared across agents."""

    @staticmethod
    def R_bind(E_bind: float, E_ref_mean: float, dE_bind: float = 1.5) -> float:
        return float(np.tanh((E_ref_mean - E_bind) / max(dE_bind, 1e-6)))

    @staticmethod
    def R_select(E_on: float, E_off_mean: float, dE_sel: float = 2.0) -> float:
        return float(np.tanh((E_off_mean - E_on) / max(dE_sel, 1e-6)))

    @staticmethod
    def R_lambda(
        entropy_per_shell: np.ndarray,
        curvature_gradient: float,
        leakage: np.ndarray,
    ) -> float:
        entropy_arr = np.asarray(entropy_per_shell, dtype=float)
        leakage_arr = np.asarray(leakage, dtype=float)
        sigma_H = float(np.std(entropy_arr)) if entropy_arr.size else 0.0
        G = abs(float(curvature_gradient))
        L = float(np.mean(leakage_arr)) if leakage_arr.size else 0.0
        w_H = w_G = w_L = 1.0 / 3.0
        return float(
            w_H * np.exp(-sigma_H)
            + w_G * np.exp(-G)
            + w_L * np.exp(-L)
        )

    @staticmethod
    def R_safety(p_tox: float, u_tox: float, beta_u: float = 0.5) -> float:
        return float(-(p_tox + beta_u * u_tox))

    @staticmethod
    def R_synth(Y: float, C: float, gamma_Y: float = 2.0, gamma_C: float = 1.0) -> float:
        return float(np.tanh(gamma_Y * Y - gamma_C * C))

    @staticmethod
    def R_IP(d_IP: float, d0: float) -> float:
        return float(np.tanh(d_IP / max(d0, 1e-6)))

    @staticmethod
    def R_dd(Q_t: float, max_Q_so_far: float, lambda_dd: float) -> float:
        DD_t = max(0.0, max_Q_so_far - Q_t)
        return float(-lambda_dd * DD_t)

    @staticmethod
    def R_vol(Q_window: np.ndarray, lambda_vol: float) -> float:
        window = np.asarray(Q_window, dtype=float)
        if window.size == 0:
            return 0.0
        return float(-lambda_vol * np.std(window))

    @staticmethod
    def R_compress(
        is_compressed: bool,
        compression_ratio: float,
        basis_mismatch: bool,
        k_c: float,
        k_b: float,
    ) -> float:
        if not is_compressed:
            return 0.0
        mismatch_penalty = k_b if basis_mismatch else 0.0
        return float(k_c * np.log1p(max(compression_ratio, 0.0)) - mismatch_penalty)

    @staticmethod
    def R_lambda_flow(
        occ_t: np.ndarray,
        occ_prev: np.ndarray,
        shellEntropyDelta: float,
        alpha_flow: float = 2.0,
        alpha_entropy: float = 1.0,
    ) -> float:
        occ_curr = np.asarray(occ_t, dtype=float)
        occ_prev = np.asarray(occ_prev, dtype=float)
        if occ_prev.size == 0:
            occ_prev = np.zeros_like(occ_curr)
        if occ_curr.size == 0:
            occ_curr = np.zeros_like(occ_prev)
        length = max(len(occ_curr), len(occ_prev))
        if len(occ_curr) != length:
            occ_curr = np.pad(occ_curr, (0, length - len(occ_curr)))
        if len(occ_prev) != length:
            occ_prev = np.pad(occ_prev, (0, length - len(occ_prev)))
        delta_occ_L1 = float(np.sum(np.abs(occ_curr - occ_prev)))
        return float(
            np.exp(-alpha_flow * delta_occ_L1)
            - np.exp(-alpha_entropy * abs(shellEntropyDelta))
        )

    @staticmethod
    def R_stagnation(
        Q_t: float,
        Q_prev: float,
        H_t: float,
        H_prev: float,
        E_t: float,
        E_prev: float,
        eps_Q: float = 0.02,
        eps_H: float = 1e-4,
        eps_E: float = 0.1,
        lambda_stag: float = 0.3,
    ) -> float:
        stagnant = (
            abs(Q_t - Q_prev) < eps_Q
            and abs(H_t - H_prev) < eps_H
            and abs(E_t - E_prev) < eps_E
        )
        return float(-lambda_stag * (1.0 if stagnant else 0.0))

    @staticmethod
    def R_validator(
        adjustmentCount: int,
        meanClampDistance: float,
        lambda_adj: float = 0.08,
        lambda_clamp: float = 0.02,
    ) -> float:
        return float(-(lambda_adj * adjustmentCount + lambda_clamp * meanClampDistance))

    @staticmethod
    def R_accept_ligand(num_accepted: int, beam_width: int, rho_accept: float = 0.5) -> float:
        if beam_width <= 0:
            return 0.0
        return float(rho_accept * (num_accepted / float(beam_width)))

    @staticmethod
    def R_diversity(diversity_index: float, rho_div: float = 0.3) -> float:
        return float(rho_div * diversity_index)

