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

from kg_scale_invariant_metric import (
    GeometryParams,
    FieldParams,
    integrate_profile,
    build_kg_operator,
    compute_modes,
    normalize_on_z,
)

# Reuse enhanced engine from phase4_unification.py via import or duplication
from phase4_unification import (
    BackreactionParams as BaseParams,
    compute_curvature_from_r,
    build_operator_and_modes,
    local_stress_components,
    smooth1d,
    second_derivative,
    gradient,
    renormalized_stress,
    refined_covariance_overlap,
)


@dataclass
class EnhancedParams(BaseParams):
    # Longer schedule, looser cap, slightly stronger λ_Q to encourage progress while keeping monotonicity
    max_iters: int = 600
    du_cap: float = 2.0e-2
    lambda_Q: float = 0.32
    dt_init: float = 3e-3
    dt_min: float = 2.5e-6
    robust_quantile: float = 0.98
    k_eig: int = 56
    local_smooth_window: int = 25
    # Line-search tolerance (relative/absolute) to allow tiny numerical jitter
    ls_rel_tol: float = 1e-6
    ls_abs_tol: float = 1e-9


def run_unification_enhanced() -> None:
    p = EnhancedParams()
    alpha = math.log(p.lam)
    z = np.linspace(p.z_min, p.z_max, p.num_z)
    dz = z[1] - z[0]

    # Initial and reference profiles
    geo0 = GeometryParams(lam=p.lam, z_min=p.z_min, z_max=p.z_max, num_z=p.num_z, r0=p.r0, epsilon=p.epsilon0)
    z0, r0, rho0, R0 = integrate_profile(geo0)
    assert np.allclose(z0, z)

    georef = GeometryParams(lam=p.lam, z_min=p.z_min, z_max=p.z_max, num_z=p.num_z, r0=p.r0, epsilon=0.0)
    z_ref, r_ref, _, R_ref = integrate_profile(georef)

    field = FieldParams(mu=p.mu, xi=p.xi, m_theta=p.m_theta, k_eig=p.k_eig)

    # Precompute reference local stress once
    w_ref, modes_ref, pot_ref = build_operator_and_modes(z_ref, r_ref, R_ref, field, p.k_eig)
    ref_loc = local_stress_components(z_ref, r_ref, modes_ref, w_ref, pot_ref)

    # State
    u = np.log(r0.copy())
    hist_norm: List[float] = []
    hist_dt: List[float] = []

    dt = p.dt_init

    # Initial norm
    rp = gradient(np.exp(u), dz)
    rho = rp / (alpha * np.exp(u))
    norm_prev = float(np.sqrt(np.trapezoid((rho - 1.0) ** 2, z)))

    for it in range(p.max_iters):
        r = np.exp(u)
        rp, rpp, R = compute_curvature_from_r(r, dz)
        rho = rp / (alpha * np.clip(r, 1e-18, None))

        # Renormalized stress on current background using precomputed ref_loc
        stress, coeffs = renormalized_stress(z, r, R, field, p.k_eig, p.adiabatic_order, ref_profile=(z_ref, r_ref, R_ref))
        e_ren = stress['E']

        # Robust normalization + smoothing
        scale_e = max(np.quantile(np.abs(e_ren), p.robust_quantile), 1e-12)
        e_norm = smooth1d(e_ren / scale_e, p.local_smooth_window)

        # Components
        uzz = second_derivative(u, dz)
        rhozz = second_derivative(rho, dz)
        delta_rho = 1.0 - rho

        # Localized feedback
        err = smooth1d(np.abs(1.0 - rho), p.local_smooth_window)
        if np.max(err) > 0:
            err = err / np.max(err)
        kappa_loc = p.kappa * (1.0 + p.kappa_boost * err)
        lambdaR_loc = p.lambda_R * (1.0 + p.lambdaR_boost * err)

        du_dt = delta_rho - p.lambda_Q * e_norm + kappa_loc * uzz + lambdaR_loc * rhozz

        # Backtracking with Δu cap and adaptive λ_Q scaling
        accepted = False
        q = 1.0  # scale for λ_Q within this iteration
        for _ in range(p.backtracking_max_steps):
            du = dt * (delta_rho - (p.lambda_Q * q) * e_norm + kappa_loc * uzz + lambdaR_loc * rhozz)
            max_abs_du = float(np.max(np.abs(du)))
            if max_abs_du > p.du_cap:
                du *= (p.du_cap / (max_abs_du + 1e-18))
            u_try = u + du
            r_try = np.exp(u_try)
            rp_try = gradient(r_try, dz)
            rho_try = rp_try / (alpha * np.clip(r_try, 1e-18, None))
            norm_try = float(np.sqrt(np.trapezoid((rho_try - 1.0) ** 2, z)))
            thresh = norm_prev * (1.0 - p.ls_rel_tol) + p.ls_abs_tol
            if norm_try <= thresh:
                u = u_try
                norm_prev = norm_try
                accepted = True
                break
            else:
                if dt <= p.dt_min + 1e-15:
                    q *= 0.5  # weaken λ_Q contribution if at dt floor
                    if q < 1e-3:
                        # give up this iteration
                        break
                else:
                    dt = max(p.dt_min, dt * p.decay_factor)
        if not accepted:
            break

        dt = min(p.dt_max, dt * p.grow_factor)
        hist_norm.append(norm_prev)
        hist_dt.append(dt)
        if norm_prev < p.tol_rho:
            break

    # Final quantities
    r_final = np.exp(u)
    rpF, rppF, RF = compute_curvature_from_r(r_final, dz)
    rho_final = rpF / (alpha * np.clip(r_final, 1e-18, None))

    # Final stress and overlaps
    stress_final, coeffs_final = renormalized_stress(z, r_final, RF, field, p.k_eig, p.adiabatic_order,
                                                     ref_profile=(z_ref, r_ref, R_ref))
    cov_metrics = refined_covariance_overlap(p.lam, p.z_min, p.z_max, p.num_z, field, p.k_eig)

    os.makedirs('outputs', exist_ok=True)

    # Plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(z, rho0, label='initial ρ(z)')
    ax[0].plot(z, rho_final, label='final ρ(z)')
    ax[0].axhline(1.0, color='k', ls='--', lw=0.8)
    ax[0].set_xlabel('z')
    ax[0].set_ylabel('ρ(z)')
    ax[0].set_title('Index density relaxation with quantum backreaction (enhanced)')
    ax[0].legend()

    ax[1].plot(np.arange(len(hist_norm)), hist_norm, 'o-')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('||ρ−1||_2')
    ax[1].set_title('Monotone descent (enhanced schedule)')
    plt.tight_layout()
    plt.savefig('outputs/phase4_unification_enhanced_rho_descent.png', dpi=150)
    plt.close(fig)

    # Stress plot
    plt.figure(figsize=(10,4))
    plt.plot(z, stress_final['E'], label='E_ren(z)')
    plt.plot(z, stress_final['Pz'], label='Pz_ren(z)')
    plt.xlabel('z'); plt.ylabel('density')
    plt.title('Renormalized local stress (enhanced)')
    plt.legend(); plt.tight_layout()
    plt.savefig('outputs/phase4_unification_enhanced_stress.png', dpi=150)
    plt.close()

    # Report export
    report_lines = []
    report_lines.append(f"iters: {len(hist_norm)}")
    report_lines.append(f"dt first/last: {hist_dt[0] if hist_dt else p.dt_init:.6e} / {hist_dt[-1] if hist_dt else p.dt_init:.6e}")
    report_lines.append(f"norm first/last: {hist_norm[0] if hist_norm else float('nan'):.6f} / {hist_norm[-1] if hist_norm else float('nan'):.6f}")
    report_lines.append(f"cov_mean/max: {cov_metrics['mean_overlap']:.3f} / {cov_metrics['max_overlap']:.3f}")
    if isinstance(stress_final, dict):
        E = stress_final['E']; Pz = stress_final['Pz']
        report_lines.append(f"E_ren[min,median,max]: {E.min():.6f}, {np.median(E):.6f}, {E.max():.6f}")
        report_lines.append(f"Pz_ren[min,median,max]: {Pz.min():.6f}, {np.median(Pz):.6f}, {Pz.max():.6f}")
    if isinstance(coeffs_final, dict) and coeffs_final:
        # Save coefficients sorted by name
        for k in sorted(coeffs_final.keys()):
            report_lines.append(f"coeff {k}: {coeffs_final[k]:.6e}")

    with open('outputs/phase4_unification_enhanced_report.txt', 'w') as f:
        f.write('\n'.join(report_lines) + '\n')

    # Save npz
    np.savez('outputs/phase4_unification_enhanced_results.npz',
             lam=p.lam, z=z,
             r_initial=r0, r_final=r_final,
             rho_initial=rho0, rho_final=rho_final,
             R_final=RF, hist_norm=np.array(hist_norm), hist_dt=np.array(hist_dt),
             stress_E=stress_final['E'], stress_Pz=stress_final['Pz'],
             cov_mean_overlap=cov_metrics['mean_overlap'], cov_max_overlap=cov_metrics['max_overlap'],
             adiabatic_coeffs=np.array([coeffs_final[k] for k in sorted(coeffs_final.keys())]) if coeffs_final else np.array([]),
             params=dict(lambda_Q=p.lambda_Q, lambda_R=p.lambda_R, kappa=p.kappa, du_cap=p.du_cap, max_iters=p.max_iters))

    print('Enhanced unification complete.')
    print('Saved outputs: outputs/phase4_unification_enhanced_*.png, results.npz, and report.txt')


if __name__ == '__main__':
    run_unification_enhanced()