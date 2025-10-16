#!/usr/bin/env python3
import os
import math
import json
from dataclasses import dataclass, replace
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import building blocks
from kg_scale_invariant_metric import (
    GeometryParams,
    FieldParams,
    integrate_profile,
)

from phase4_unification import (
    BackreactionParams,
    compute_curvature_from_r,
    build_operator_and_modes,
    local_stress_components,
    smooth1d,
    second_derivative,
    gradient,
    renormalized_stress,
)

from phase4_unification_enhanced import EnhancedParams


def simulate_one(
    p: EnhancedParams,
    overrides: Optional[Dict] = None,
    noise_std: float = 0.0,
    adiabatic_order: Optional[int] = None,
    basis_variant: Optional[str] = None,
    ref_profile_variant: Optional[str] = None,
    long_time: bool = False,
) -> Dict:
    # Apply overrides
    if overrides:
        for k, v in overrides.items():
            if hasattr(p, k):
                setattr(p, k, v)

    alpha = math.log(p.lam)
    z = np.linspace(p.z_min, p.z_max, p.num_z)
    dz = z[1] - z[0]

    # Initial and reference (variant) profiles
    geo0 = GeometryParams(lam=p.lam, z_min=p.z_min, z_max=p.z_max, num_z=p.num_z, r0=p.r0, epsilon=p.epsilon0)
    z0, r0, rho0, R0 = integrate_profile(geo0)
    assert np.allclose(z0, z)

    # Add optional initial noise in u
    u = np.log(r0.copy())
    if noise_std > 0.0:
        rng = np.random.default_rng(42)
        u += rng.normal(scale=noise_std, size=u.shape)

    # Reference profile variant
    r0_ref = p.r0
    if ref_profile_variant == 'lam':
        r0_ref = p.r0 * p.lam
    elif ref_profile_variant == 'lam2':
        r0_ref = p.r0 * (p.lam ** 2)
    georef = GeometryParams(lam=p.lam, z_min=p.z_min, z_max=p.z_max, num_z=p.num_z, r0=r0_ref, epsilon=0.0)
    z_ref, r_ref, _, R_ref = integrate_profile(georef)

    # Field
    field = FieldParams(mu=p.mu, xi=p.xi, m_theta=p.m_theta, k_eig=p.k_eig)

    # Precompute reference local stress
    w_ref, modes_ref, pot_ref = build_operator_and_modes(z_ref, r_ref, R_ref, field, p.k_eig)
    ref_loc = local_stress_components(z_ref, r_ref, modes_ref, w_ref, pot_ref)

    # Diagnostics
    hist_norm: List[float] = []
    hist_dt: List[float] = []
    hist_energy: List[float] = []

    dt = p.dt_init

    # Initial norm
    rp = gradient(np.exp(u), dz)
    rho = rp / (alpha * np.exp(u))
    norm_prev = float(np.sqrt(np.trapezoid((rho - 1.0) ** 2, z)))

    max_steps = p.max_iters if long_time else min(60, p.max_iters)

    # Iterative relaxation
    for it in range(max_steps):
        r = np.exp(u)
        rp, rpp, R = compute_curvature_from_r(r, dz)
        rho = rp / (alpha * np.clip(r, 1e-18, None))

        # Choose adiabatic order
        ao = p.adiabatic_order if adiabatic_order is None else adiabatic_order

        # Compute renormalized stress with precomputed reference
        stress, coeffs = renormalized_stress(z, r, R, field, p.k_eig, ao, ref_loc=ref_loc)
        e_ren = stress['E']

        # Optional basis variation for robustness: build alternate fit and overwrite e_ren
        if basis_variant in {'no_rpp', 'no_rho_grad2', 'R_only'}:
            dz_local = z[1] - z[0]
            rp_local = rp
            rpp_local = rpp
            rho_local = rho
            basis = {'const': np.ones_like(z), 'R': R, 'rho_grad2': gradient(rho_local, dz_local)**2, 'rpp_over_r': rpp_local/np.clip(r,1e-18,None)}
            # Current minus reference (adiabatic order 0) as target to fit
            stress0, _ = renormalized_stress(z, r, R, field, p.k_eig, 0, ref_loc=ref_loc)
            target = stress0['E']
            cols = [basis['const'], basis['R']]
            if basis_variant == 'no_rho_grad2':
                cols.append(basis['rpp_over_r'])
            elif basis_variant == 'no_rpp':
                cols.append(basis['rho_grad2'])
            elif basis_variant == 'R_only':
                pass
            else:
                cols += [basis['rho_grad2'], basis['rpp_over_r']]
            B = np.vstack(cols).T
            coeffs_fit, *_ = np.linalg.lstsq(B, target, rcond=None)
            fit = B @ coeffs_fit
            e_ren = target - fit

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

        # Backtracking with cap
        accepted = False
        q = 1.0
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
            thresh = norm_prev * (1.0 - getattr(p, 'ls_rel_tol', 1e-6)) + getattr(p, 'ls_abs_tol', 1e-9)
            if norm_try <= thresh:
                u = u_try
                norm_prev = norm_try
                accepted = True
                break
            else:
                if dt <= p.dt_min + 1e-15:
                    q *= 0.5
                    if q < 1e-3:
                        break
                else:
                    dt = max(p.dt_min, dt * p.decay_factor)
        if not accepted:
            break

        dt = min(p.dt_max, dt * p.grow_factor)
        hist_norm.append(norm_prev)
        hist_dt.append(dt)

        # Track integrated energy (quasi-local) as a sanity check
        hist_energy.append(float(np.trapezoid(e_ren, z)))

    # Final state
    r_final = np.exp(u)
    rpF, rppF, RF = compute_curvature_from_r(r_final, dz)
    rho_final = rpF / (alpha * np.clip(r_final, 1e-18, None))

    # Final stress for reporting
    stress_final, coeffs_final = renormalized_stress(z, r_final, RF, field, p.k_eig, p.adiabatic_order, ref_profile=(z_ref, r_ref, R_ref))

    return dict(
        z=z, r0=r0, r=r_final, rho0=rho0, rho=rho_final, R=RF,
        hist_norm=np.array(hist_norm), hist_dt=np.array(hist_dt), hist_energy=np.array(hist_energy),
        E_ren=stress_final['E'], Pz_ren=stress_final['Pz'], coeffs=coeffs_final,
        params=p,
    )


def l2_norm_on(z: np.ndarray, f: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(f*f, z)))


def pointwise_diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = a - b
    return dict(l2=float(np.sqrt(np.mean(d*d))), max=float(np.max(np.abs(d))), med=float(np.median(np.abs(d))))


def fit_exponential(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    # y(t) ~ A exp(-t/tau) + C ; crude fit with linearization on y-C
    if len(y) < 3:
        return dict(A=np.nan, tau=np.nan, C=np.nan, r2=np.nan)
    C = y[-1]
    y_shift = y - C
    y_shift = np.clip(y_shift, 1e-18, None)
    X = np.vstack([t, np.ones_like(t)]).T
    beta, *_ = np.linalg.lstsq(X, -np.log(y_shift), rcond=None)
    tau = 1.0 / beta[0]
    A = math.exp(beta[1])
    # r2
    y_pred = A * np.exp(-t/tau) + C
    ss_res = float(np.sum((y - y_pred)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-18
    r2 = 1.0 - ss_res/ss_tot
    return dict(A=A, tau=tau, C=C, r2=r2)


def compute_einstein_residual(z: np.ndarray, R: np.ndarray, E_ren: np.ndarray) -> Dict[str, float]:
    Delta = R - 8.0 * math.pi * E_ren
    return dict(l2=l2_norm_on(z, Delta), max=float(np.max(np.abs(Delta))), mean=float(np.mean(np.abs(Delta))))


def convergence_campaign():
    os.makedirs('outputs', exist_ok=True)

    base = EnhancedParams()
    studies = []

    # Convergence grid & mode
    for mult in [1, 2]:  # ×1, ×2 (keep runtime reasonable)
        for k in [40, 80, 160]:
            p = EnhancedParams()
            overrides = dict(num_z=base.num_z//2 * mult, k_eig=k, du_cap=0.01, lambda_Q=0.2, dt_init=1e-3, dt_min=1e-6)
            res = simulate_one(p, overrides=overrides)
            z = res['z']
            studies.append(dict(tag=f"nz{overrides['num_z']}_k{k}",
                                final_norm=l2_norm_on(z, res['rho']-1.0),
                                E=res['E_ren'], z=z,
                                overlaps_mean=np.nan, overlaps_max=np.nan))

    # Use highest-resolution run as reference for pointwise E
    ref = max(studies, key=lambda s: (int(s['tag'].split('_k')[0][2:]), int(s['tag'].split('_k')[1])))
    for s in studies:
        stats = pointwise_diff_stats(np.interp(ref['z'], s['z'], s['E']), ref['E'])
        s.update(dict(E_diff_l2=stats['l2'], E_diff_max=stats['max'], E_diff_med=stats['med']))

    # Long-time monotone run with many iterations accepted
    p_long = EnhancedParams(max_iters=800, du_cap=0.03, lambda_Q=0.15, dt_init=5e-4, dt_min=1e-7, k_eig=48, local_smooth_window=21)
    res_long = simulate_one(p_long, overrides=None, long_time=True)
    t = np.arange(len(res_long['hist_norm']), dtype=float)
    fit = fit_exponential(t, res_long['hist_norm'])

    # Einstein residual initial/final
    ein_init = compute_einstein_residual(res_long['z'], compute_curvature_from_r(res_long['r0'], res_long['z'][1]-res_long['z'][0])[2], res_long['E_ren'])
    ein_final = compute_einstein_residual(res_long['z'], res_long['R'], res_long['E_ren'])

    # Renorm robustness: adiabatic 0 vs 2; basis variants; reference variants
    robust = []
    for ao in [0, 2]:
        for bv in [None, 'no_rpp', 'no_rho_grad2', 'R_only']:
            for rv in [None, 'lam']:
                rtest = simulate_one(EnhancedParams(k_eig=40), overrides=None, adiabatic_order=ao, basis_variant=bv, ref_profile_variant=rv)
                robust.append(dict(ao=ao, bv=bv or 'full', rv=rv or 'base',
                                   E_stats=dict(min=float(np.min(rtest['E_ren'])), max=float(np.max(rtest['E_ren'])), med=float(np.median(rtest['E_ren']))),
                                   final_norm=l2_norm_on(rtest['z'], rtest['rho']-1.0)))

    # Basin tests: epsilon0 variations and noise
    basins = []
    for eps0 in [0.0, 0.05, 0.10]:
        p_b = EnhancedParams(epsilon0=eps0)
        r_b = simulate_one(p_b, overrides=None)
        basins.append(dict(eps0=eps0, final_norm=l2_norm_on(r_b['z'], r_b['rho']-1.0)))
    # Random noise
    p_n = EnhancedParams(epsilon0=0.05)
    r_n = simulate_one(p_n, overrides=None, noise_std=0.01)
    basins.append(dict(eps0='0.05+noise', final_norm=l2_norm_on(r_n['z'], r_n['rho']-1.0)))

    # Parameter sweep (coarse)
    sweep = []
    for lQ in [0.1, 0.2, 0.35]:
        for lR in [0.25, 0.35, 0.6]:
            for kap in [0.8, 1.0, 1.5]:
                p_s = EnhancedParams(lambda_Q=lQ, lambda_R=lR, kappa=kap, k_eig=32, num_z=1200)
                r_s = simulate_one(p_s, overrides=None)
                init_norm = l2_norm_on(r_s['z'], r_s['rho0']-1.0)
                final_norm = l2_norm_on(r_s['z'], r_s['rho']-1.0)
                status = 'stable' if final_norm < init_norm else 'stagnant'
                sweep.append(dict(lambda_Q=lQ, lambda_R=lR, kappa=kap, init_norm=init_norm, final_norm=final_norm, status=status, iters=int(len(r_s['hist_norm']))))

    # Spectral checks (continuum) – compute E_n for n=0..8 and fit to an^2+b
    from continuum_logscale_solver import ContinuumParams, make_W_function, find_eigenvalue_near, reconstruct_psi_from_phi, normalize_density
    cp = ContinuumParams()
    ns = list(range(0, 9))
    x_grid = np.linspace(cp.x_min, cp.x_max, 2500)
    W = make_W_function(cp)
    En = []
    for n in ns:
        E_n, phi_n = find_eigenvalue_near(n, cp, W, x_grid)
        En.append(E_n)
    En = np.array(En)
    X = np.vstack([np.array(ns)**2, np.ones_like(ns)]).T
    coeff, *_ = np.linalg.lstsq(X, En, rcond=None)
    En_fit = X @ coeff
    rmse = float(np.sqrt(np.mean((En - En_fit)**2)))
    ss_res = float(np.sum((En - En_fit)**2))
    ss_tot = float(np.sum((En - np.mean(En))**2)) + 1e-18
    R2 = 1.0 - ss_res/ss_tot

    # Save campaign results
    # Ensure everything is JSON-serializable
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    results = dict(
        convergence=studies,
        reference_tag=ref['tag'],
        longrun=dict(iters=int(len(res_long['hist_norm'])), hist_norm=res_long['hist_norm'].tolist(), fit=fit),
        einstein_residual=dict(initial=ein_init, final=ein_final),
        robustness=robust,
        basins=basins,
        sweep=sweep,
        spectral=dict(ns=ns, E=En.tolist(), fit_coeff=coeff.tolist(), rmse=rmse, R2=R2),
    )

    with open('outputs/phase4_unification_enhanced_campaign.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)

    # Plot long-run decay
    if results['longrun']['iters'] > 0:
        y = np.array(results['longrun']['hist_norm'])
        x = np.arange(len(y))
        plt.figure(figsize=(6,4))
        plt.semilogy(x, y, 'o-', label='||ρ−1||_2')
        if np.isfinite(results['longrun']['fit'].get('tau', float('nan'))):
            A = results['longrun']['fit']['A']; tau = results['longrun']['fit']['tau']; C = results['longrun']['fit']['C']
            plt.semilogy(x, A*np.exp(-x/tau)+C, 'r--', label=f"exp fit τ≈{tau:.2f}")
        plt.xlabel('iteration'); plt.ylabel('||ρ−1||_2'); plt.legend(); plt.tight_layout()
        plt.savefig('outputs/phase4_unification_enhanced_longrun_decay.png', dpi=150)
        plt.close()

    print('Campaign complete. Artifacts written to outputs/*.')


if __name__ == '__main__':
    convergence_campaign()
