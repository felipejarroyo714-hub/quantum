#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment

# Reuse Phase 3 building blocks
from kg_scale_invariant_metric import (
    GeometryParams,
    FieldParams,
    integrate_profile,
    build_kg_operator,
    compute_modes,
    normalize_on_z,
)


@dataclass
class BackreactionParams:
    # Geometry / grid
    lam: float = math.sqrt(6.0) / 2.0
    z_min: float = -10.0
    z_max: float = 10.0
    num_z: int = 2400  # refined grid (can increase further if needed)
    r0: float = 1.0
    epsilon0: float = 0.05  # initial λ-periodic modulation to relax away

    # Quantum field / spectrum
    k_eig: int = 48  # number of modes sampled for ⟨Tμν⟩
    mu: float = 0.5
    xi: float = 0.0
    m_theta: int = 0

    # Semiclassical backreaction weights
    lambda_Q: float = 0.25  # > 0, strength of quantum backreaction
    lambda_R: float = 0.35  # base smoothing on ρ via Laplacian(ρ)
    kappa: float = 1.0      # base smoothing on u = ln r via Laplacian(u)
    # Localized feedback boosts (scaled by smoothed |ρ-1| in [0,1])
    kappa_boost: float = 3.0
    lambdaR_boost: float = 2.0

    # Time stepping for energy descent on ||ρ−1||
    dt_init: float = 2e-3
    dt_min: float = 5e-6  # tighter minimum for stiff regimes
    dt_max: float = 2e-2
    decay_factor: float = 0.5
    grow_factor: float = 1.1
    max_iters: int = 200
    tol_rho: float = 5e-3

    # Adiabatic subtraction
    adiabatic_order: int = 2  # 0 or 2 (least-squares curvature fit)
    robust_quantile: float = 0.95  # robust scaling for e_ren normalization

    # Step control
    du_cap: float = 5e-3       # cap on |Δu| per step (L∞)
    backtracking_max_steps: int = 60

    # Local smoothing window (in grid points) for feedback weights
    local_smooth_window: int = 21


# ---------- Helpers ----------

def second_derivative(arr: np.ndarray, dz: float) -> np.ndarray:
    n = len(arr)
    out = np.zeros_like(arr)
    # central differences
    out[1:-1] = (arr[:-2] - 2.0 * arr[1:-1] + arr[2:]) / (dz * dz)
    # Neumann at boundaries (zero second derivative)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def gradient(arr: np.ndarray, dz: float) -> np.ndarray:
    out = np.zeros_like(arr)
    out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dz)
    out[0] = (arr[1] - arr[0]) / dz
    out[-1] = (arr[-1] - arr[-2]) / dz
    return out


def smooth1d(arr: np.ndarray, window: int) -> np.ndarray:
    # Simple edge-padded moving-average smoothing; window must be odd
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w == 1:
        return arr.copy()
    pad = w // 2
    kernel = np.ones(w, dtype=float) / float(w)
    arr_pad = np.pad(arr, (pad, pad), mode='edge')
    sm = np.convolve(arr_pad, kernel, mode='same')
    return sm[pad:-pad]


def compute_curvature_from_r(r: np.ndarray, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rp = gradient(r, dz)
    rpp = second_derivative(r, dz)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = -rpp / np.clip(r, 1e-18, None)
        R = 2.0 * K
    return rp, rpp, R


def build_operator_and_modes(z: np.ndarray, r: np.ndarray, R: np.ndarray, field: FieldParams, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A, pot = build_kg_operator(z, r, R, field)
    w2, modes = compute_modes(A, k=k)
    # normalize modes on z
    for j in range(modes.shape[1]):
        modes[:, j] = normalize_on_z(z, modes[:, j])
    w = np.sqrt(np.clip(w2, 0.0, None))
    return w, modes, pot


def local_stress_components(z: np.ndarray, r: np.ndarray, modes: np.ndarray, w: np.ndarray, pot: np.ndarray) -> Dict[str, np.ndarray]:
    dz = z[1] - z[0]
    # gradients of modes
    du = np.zeros_like(modes)
    du[1:-1, :] = (modes[2:, :] - modes[:-2, :]) / (2.0 * dz)
    du[0, :] = (modes[1, :] - modes[0, :]) / dz
    du[-1, :] = (modes[-1, :] - modes[-2, :]) / dz

    # Canonical static energy-momentum (per mode) on time-slice:
    # e_j = 0.5 (|∂_z u_j|^2 + pot |u_j|^2)
    # pz_j = 0.5 (|∂_z u_j|^2 - pot |u_j|^2)
    # Sum over j (vacuum occupancy 1/2 already implicit in 0.5 factor)
    abs_u2 = np.abs(modes) ** 2
    abs_d2 = np.abs(du) ** 2

    e_density = 0.5 * (abs_d2.sum(axis=1) + (pot[:, None] * abs_u2).sum(axis=1))
    pz_density = 0.5 * (abs_d2.sum(axis=1) - (pot[:, None] * abs_u2).sum(axis=1))

    # For m_theta=0, p_theta contribution from angular term vanishes effectively at this level
    ptheta_density = np.zeros_like(e_density)

    return dict(E=e_density, Pz=pz_density, Ptheta=ptheta_density)


def adiabatic_subtraction(e_cur: np.ndarray,
                          basis_fields: Dict[str, np.ndarray],
                          order: int = 2) -> Tuple[np.ndarray, Dict[str, float]]:
    # Subtract low-order local curvature fit to emulate adiabatic counterterms
    if order <= 0:
        return e_cur.copy(), {}

    # Build design matrix B with columns [1, R, (r'/r)^2, (r''/r)]
    Bcols: List[np.ndarray] = [
        np.ones_like(next(iter(basis_fields.values()))),
    ]
    names: List[str] = ["const"]

    for key in ["R", "rho_grad2", "rpp_over_r"]:
        if key in basis_fields:
            Bcols.append(basis_fields[key])
            names.append(key)

    B = np.vstack(Bcols).T  # (n, p)

    # Solve least squares for coefficients
    coeffs, *_ = np.linalg.lstsq(B, e_cur, rcond=None)
    fit = B @ coeffs
    e_ren = e_cur - fit
    return e_ren, {name: float(c) for name, c in zip(names, coeffs)}


def renormalized_stress(z: np.ndarray, r: np.ndarray, R: np.ndarray,
                        field: FieldParams, k: int, adiabatic_order: int,
                        ref_profile: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
                        ref_loc: Dict[str, np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    # Current background
    w_cur, modes_cur, pot_cur = build_operator_and_modes(z, r, R, field, k)
    loc_cur = local_stress_components(z, r, modes_cur, w_cur, pot_cur)

    # Reference background (epsilon=0) local stress, computed once if provided
    if ref_loc is not None:
        loc_ref = ref_loc
    else:
        assert ref_profile is not None, "Either ref_loc or ref_profile must be provided"
        z_ref, r_ref, R_ref = ref_profile
        w_ref, modes_ref, pot_ref = build_operator_and_modes(z_ref, r_ref, R_ref, field, k)
        loc_ref = local_stress_components(z_ref, r_ref, modes_ref, w_ref, pot_ref)

    # Basic subtraction (adiabatic 0): current - reference
    e0 = loc_cur["E"] - loc_ref["E"]
    pz0 = loc_cur["Pz"] - loc_ref["Pz"]
    pt0 = loc_cur["Ptheta"] - loc_ref["Ptheta"]

    # Build basis fields for adiabatic 2 fit
    dz = z[1] - z[0]
    rp = gradient(r, dz)
    rpp = second_derivative(r, dz)
    alpha = math.log(field_params_to_lam(field))
    rho = rp / (alpha * np.clip(r, 1e-18, None))

    basis = {
        "R": R,
        "rho_grad2": gradient(rho, dz) ** 2,
        "rpp_over_r": rpp / np.clip(r, 1e-18, None),
    }

    coeff_map_e: Dict[str, float] = {}
    coeff_map_p: Dict[str, float] = {}

    if adiabatic_order > 0:
        e_ren, coeffs_e = adiabatic_subtraction(e0, basis, order=adiabatic_order)
        pz_ren, coeffs_p = adiabatic_subtraction(pz0, basis, order=adiabatic_order)
        pt_ren, _ = adiabatic_subtraction(pt0, basis, order=adiabatic_order)
        coeff_map_e.update({f"E_{k}": v for k, v in coeffs_e.items()})
        coeff_map_p.update({f"Pz_{k}": v for k, v in coeffs_p.items()})
    else:
        e_ren, pz_ren, pt_ren = e0, pz0, pt0

    return dict(E=e_ren, Pz=pz_ren, Ptheta=pt_ren), {**coeff_map_e, **coeff_map_p}


def field_params_to_lam(field: FieldParams) -> float:
    # All GeometryParams share lam; use Phase-3 default mapping
    return GeometryParams().lam


def refined_covariance_overlap(lam: float, z_min: float, z_max: float, num_z: int,
                               field: FieldParams, k: int) -> Dict[str, float]:
    # Build two backgrounds differing by r0 -> λ r0
    geoA = GeometryParams(lam=lam, z_min=z_min, z_max=z_max, num_z=num_z, r0=1.0, epsilon=0.0)
    zA, rA, _, RA = integrate_profile(geoA)
    w2A, vA = compute_modes(build_kg_operator(zA, rA, RA, field)[0], k=min(20, k))

    geoB = GeometryParams(lam=lam, z_min=z_min, z_max=z_max, num_z=num_z, r0=lam, epsilon=0.0)
    zB, rB, _, RB = integrate_profile(geoB)
    w2B, vB = compute_modes(build_kg_operator(zB, rB, RB, field)[0], k=min(20, k))

    # Normalize
    for j in range(vA.shape[1]):
        vA[:, j] = normalize_on_z(zA, vA[:, j])
    for j in range(vB.shape[1]):
        vB[:, j] = normalize_on_z(zB, vB[:, j])

    # Scan small shifts δ to maximize pairwise overlaps
    deltas = np.linspace(-1.25, 1.25, 11)

    def overlap_at_delta(uA: np.ndarray, zA: np.ndarray, uB: np.ndarray, zB: np.ndarray, delta: float) -> float:
        z_shift = zA + delta
        uB_shift = np.interp(z_shift, zB, uB, left=0.0, right=0.0)
        return abs(np.trapezoid(uA * uB_shift, zA))

    # Build cost matrix as negative max overlap over δ
    nA = vA.shape[1]
    nB = vB.shape[1]
    M = np.zeros((nA, nB))
    for i in range(nA):
        for j in range(nB):
            ov_max = 0.0
            for d in deltas:
                ov = overlap_at_delta(vA[:, i], zA, vB[:, j], zB, d)
                if ov > ov_max:
                    ov_max = ov
            M[i, j] = -ov_max

    # Optimal pairing
    row_ind, col_ind = linear_sum_assignment(M)
    overlaps = [-M[i, j] for i, j in zip(row_ind, col_ind)]
    return {
        "mean_overlap": float(np.mean(overlaps)),
        "max_overlap": float(np.max(overlaps)),
        "num_pairs": int(len(overlaps)),
    }


# ---------- Main backreaction loop ----------

def run_unification() -> None:
    p = BackreactionParams()
    alpha = math.log(p.lam)
    z = np.linspace(p.z_min, p.z_max, p.num_z)
    dz = z[1] - z[0]

    # Initial background with small λ-periodic perturbation
    geo0 = GeometryParams(lam=p.lam, z_min=p.z_min, z_max=p.z_max, num_z=p.num_z, r0=p.r0, epsilon=p.epsilon0)
    z0, r0, rho0, R0 = integrate_profile(geo0)
    assert np.allclose(z0, z)

    # Reference background (epsilon=0) on same z grid for subtraction
    georef = GeometryParams(lam=p.lam, z_min=p.z_min, z_max=p.z_max, num_z=p.num_z, r0=p.r0, epsilon=0.0)
    z_ref, r_ref, _, R_ref = integrate_profile(georef)
    # Precompute reference local stress once
    field = FieldParams(mu=p.mu, xi=p.xi, m_theta=p.m_theta, k_eig=p.k_eig)
    w_ref, modes_ref, pot_ref = build_operator_and_modes(z_ref, r_ref, R_ref, field, p.k_eig)
    ref_loc = local_stress_components(z_ref, r_ref, modes_ref, w_ref, pot_ref)

    # Field parameters (already instantiated above)

    # Log-profile to evolve
    u = np.log(r0.copy())

    # Diagnostics
    hist_norm: List[float] = []
    hist_dt: List[float] = []

    dt = p.dt_init

    # Precompute initial norm
    rp = gradient(np.exp(u), dz)
    rho = rp / (alpha * np.exp(u))
    norm_prev = float(np.sqrt(np.trapezoid((rho - 1.0) ** 2, z)))

    for it in range(p.max_iters):
        r = np.exp(u)
        rp, rpp, R = compute_curvature_from_r(r, dz)
        rho = rp / (alpha * np.clip(r, 1e-18, None))

        # Compute renormalized stress on current background
        stress, coeffs = renormalized_stress(
            z, r, R, field, p.k_eig, p.adiabatic_order,
            ref_loc=ref_loc
        )
        e_ren = stress["E"]

        # Backreaction driving term in u (log r)
        # Robustly normalize e_ren and lightly smooth
        scale_e = max(np.quantile(np.abs(e_ren), p.robust_quantile), 1e-12)
        e_norm = e_ren / scale_e
        e_norm = smooth1d(e_norm, p.local_smooth_window)

        # Components for update
        uzz = second_derivative(u, dz)
        rhozz = second_derivative(rho, dz)
        delta_rho = 1.0 - rho

        # Localized feedback weights based on smoothed |ρ−1|
        err = smooth1d(np.abs(1.0 - rho), p.local_smooth_window)
        if np.max(err) > 0:
            err = err / np.max(err)
        kappa_loc = p.kappa * (1.0 + p.kappa_boost * err)
        lambdaR_loc = p.lambda_R * (1.0 + p.lambdaR_boost * err)

        du_dt = (
            delta_rho
            - p.lambda_Q * e_norm
            + kappa_loc * uzz
            + lambdaR_loc * rhozz
        )

        # Backtracking line search for descent of ||ρ−1|| with windowed acceptance
        accepted = False
        # Initialize windowing parameters
        accept_window = int(getattr(p, 'accept_window', 5))
        accept_rel_budget = float(getattr(p, 'accept_window_rel_budget', 5e-6))
        accept_abs_budget = float(getattr(p, 'accept_window_abs_budget', 1e-9))
        if 'window_count' not in globals():
            globals()['window_count'] = 0
            globals()['last_anchor_norm'] = norm_prev
        for _ in range(p.backtracking_max_steps):
            # Cap per-step Δu to avoid overshoot
            du = dt * du_dt
            max_abs_du = float(np.max(np.abs(du)))
            if max_abs_du > p.du_cap:
                du *= (p.du_cap / (max_abs_du + 1e-18))
            u_try = u + du
            r_try = np.exp(u_try)
            rp_try = gradient(r_try, dz)
            rho_try = rp_try / (alpha * np.clip(r_try, 1e-18, None))
            norm_try = float(np.sqrt(np.trapezoid((rho_try - 1.0) ** 2, z)))
            # Windowed acceptance logic
            budget = globals()['last_anchor_norm'] * accept_rel_budget + accept_abs_budget
            is_window_end = ((globals()['window_count'] + 1) % accept_window) == 0
            if is_window_end:
                enforce_margin = max(globals()['last_anchor_norm'] * p.ls_rel_tol, p.ls_abs_tol)
                ok = norm_try <= (globals()['last_anchor_norm'] - enforce_margin + getattr(p, 'ls_pos_slack', 0.0))
            else:
                ok = norm_try <= (globals()['last_anchor_norm'] + budget)
            if ok:
                # accept step
                u = u_try
                norm_prev = norm_try
                accepted = True
                if is_window_end:
                    globals()['last_anchor_norm'] = norm_try
                    globals()['window_count'] = 0
                else:
                    globals()['window_count'] += 1
                break
            else:
                dt = max(p.dt_min, dt * p.decay_factor)
        if not accepted:
            # Could not decrease; stop early
            break

        # Optional slight dt growth if very stable
        dt = min(p.dt_max, dt * p.grow_factor)

        hist_norm.append(norm_prev)
        hist_dt.append(dt)

        # Early stopping
        if norm_prev < p.tol_rho:
            break

    # Final state
    r_final = np.exp(u)
    rpF, rppF, RF = compute_curvature_from_r(r_final, dz)
    rho_final = rpF / (alpha * np.clip(r_final, 1e-18, None))

    # Final renormalized stress
    stress_final, coeffs_final = renormalized_stress(z, r_final, RF, field, p.k_eig, p.adiabatic_order,
                                                     ref_profile=(z_ref, r_ref, R_ref))

    # Refined λ-covariant overlap diagnostics (pairing improvement)
    cov_metrics = refined_covariance_overlap(p.lam, p.z_min, p.z_max, p.num_z, field, p.k_eig)

    # Outputs
    os.makedirs('outputs', exist_ok=True)

    # Plots: ρ and ||ρ−1||
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(z, rho0, label='initial ρ(z)')
    ax[0].plot(z, rho_final, label='final ρ(z)')
    ax[0].axhline(1.0, color='k', ls='--', lw=0.8)
    ax[0].set_xlabel('z')
    ax[0].set_ylabel('ρ(z)')
    ax[0].set_title('Index density relaxation with quantum backreaction (λ_Q>0)')
    ax[0].legend()

    ax[1].plot(np.arange(len(hist_norm)), hist_norm, 'o-')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('||ρ−1||_2')
    ax[1].set_title('Monotone descent of ||ρ−1||_2')

    plt.tight_layout()
    plt.savefig('outputs/phase4_unification_rho_descent.png', dpi=150)
    plt.close(fig)

    # Plot renormalized energy density
    plt.figure(figsize=(10, 4))
    plt.plot(z, stress_final['E'], label='E_ren(z)')
    plt.plot(z, stress_final['Pz'], label='Pz_ren(z)')
    plt.xlabel('z')
    plt.ylabel('density')
    plt.title('Renormalized local stress components')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/phase4_unification_stress.png', dpi=150)
    plt.close()

    # Save numeric results
    np.savez('outputs/phase4_unification_results.npz',
             lam=p.lam,
             z=z,
             r_initial=r0,
             r_final=r_final,
             rho_initial=rho0,
             rho_final=rho_final,
             R_final=RF,
             hist_norm=np.array(hist_norm),
             hist_dt=np.array(hist_dt),
             stress_E=stress_final['E'],
             stress_Pz=stress_final['Pz'],
             stress_Ptheta=stress_final['Ptheta'],
             adiabatic_coeffs=np.array(list(coeffs_final.values()) if coeffs_final else []),
             cov_mean_overlap=cov_metrics['mean_overlap'],
             cov_max_overlap=cov_metrics['max_overlap'],
             cov_num_pairs=cov_metrics['num_pairs'],
             params=dict(
                 lambda_Q=p.lambda_Q,
                 lambda_R=p.lambda_R,
                 kappa=p.kappa,
                 dt_min=p.dt_min,
                 dt_init=p.dt_init,
                 adiabatic_order=p.adiabatic_order,
                 k_eig=p.k_eig,
                 num_z=p.num_z,
             ))

    # Console summary
    print(f"Unification run complete. Iters={len(hist_norm)}, final ||rho-1||_2={hist_norm[-1] if hist_norm else norm_prev:.6f}.")
    print(f"Refined λ-covariant pairing overlaps: mean={cov_metrics['mean_overlap']:.3f}, max={cov_metrics['max_overlap']:.3f}, pairs={cov_metrics['num_pairs']}.")
    print("Saved plots to outputs/phase4_unification_rho_descent.png and outputs/phase4_unification_stress.png")


if __name__ == '__main__':
    run_unification()
