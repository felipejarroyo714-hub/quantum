#!/usr/bin/env python3
import os
import math
import json
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase4_unification_enhanced import EnhancedParams
from phase4_unification_enhanced_campaign import simulate_one, fit_exponential


def run_longrun() -> Dict:
    os.makedirs('outputs', exist_ok=True)
    # Many-step monotone schedule per directive
    p = EnhancedParams(
        max_iters=40000,
        du_cap=0.30,
        ls_rel_tol=2e-5,
        ls_abs_tol=1e-9,
        robust_quantile=0.90,
        lambda_Q=0.15,
        dt_init=2e-3,
        dt_min=1e-12,
        k_eig=12,
        local_smooth_window=15,
        epsilon0=0.50,
        num_z=400,
    )
    # Apply tiny positive acceptance slack to avoid numerical pinning
    try:
        setattr(p, 'ls_pos_slack', 0.0)
        setattr(p, 'accept_window', 3)
        setattr(p, 'accept_window_rel_budget', 5e-7)
        setattr(p, 'accept_window_abs_budget', 1e-10)
        setattr(p, 'window_end_abs_margin', 5e-6)
        setattr(p, 'lambda_Q_decay', 0.98)
    except Exception:
        pass
    # Stage A: reduce curvature feedback by 50% and stronger λ_Q decay with stricter window end
    # Copy only dataclass fields
    pA = EnhancedParams(
        max_iters=p.max_iters, du_cap=p.du_cap, ls_rel_tol=p.ls_rel_tol, ls_abs_tol=p.ls_abs_tol,
        robust_quantile=p.robust_quantile, lambda_Q=p.lambda_Q, dt_init=p.dt_init, dt_min=p.dt_min,
        k_eig=p.k_eig, local_smooth_window=p.local_smooth_window, epsilon0=p.epsilon0, num_z=p.num_z
    )
    pA.kappa *= 0.5
    pA.lambda_R *= 0.5
    pA.lambda_Q = 0.10
    setattr(pA, 'lambda_Q_decay', 0.88)
    setattr(pA, 'accept_window', 3)
    setattr(pA, 'window_end_abs_margin', 2e-5)
    setattr(pA, 'accept_window_rel_budget', 1e-7)
    setattr(pA, 'smooth_window', 5)
    pA.du_cap = 0.38
    pA.max_iters = 20000
    resA = simulate_one(pA, long_time=True)

    # Stage B: restore curvature to nominal and continue from Stage A final state
    pB = EnhancedParams(
        max_iters=p.max_iters, du_cap=p.du_cap, ls_rel_tol=p.ls_rel_tol, ls_abs_tol=p.ls_abs_tol,
        robust_quantile=p.robust_quantile, lambda_Q=p.lambda_Q, dt_init=p.dt_init, dt_min=p.dt_min,
        k_eig=p.k_eig, local_smooth_window=p.local_smooth_window, epsilon0=p.epsilon0, num_z=p.num_z
    )
    init_u = np.log(resA['r'])
    # Stage B settings (chain multiple segments)
    setattr(pB, 'lambda_Q_decay', 0.98)
    setattr(pB, 'accept_window', 3)
    setattr(pB, 'window_end_abs_margin', 2e-5)
    setattr(pB, 'accept_window_rel_budget', 1e-7)
    setattr(pB, 'smooth_window', 5)
    pB.du_cap = 0.30
    pB.max_iters = 12000
    # Chain multiple Stage B segments to build a composite curve
    B_segments = []
    res_prev_u = init_u
    for _ in range(3):
        resBi = simulate_one(pB, long_time=True, init_u=res_prev_u)
        B_segments.append(resBi)
        res_prev_u = np.log(resBi['r'])

    # Combine histories for fitting/plotting
    all_hist = np.concatenate([resA['hist_norm']] + [seg['hist_norm'] for seg in B_segments])
    res = {'hist_norm': all_hist}

    # Fit exponential to decay of ||rho-1||_2
    t = np.arange(len(all_hist), dtype=float)
    fit = fit_exponential(t, all_hist)

    # Plot decay
    if len(t) > 0:
        plt.figure(figsize=(6,4))
        plt.semilogy(t, all_hist, 'o-', label='||ρ−1||_2 (Stage A→B)')
        if np.isfinite(fit.get('tau', float('nan'))):
            A = fit['A']; tau = fit['tau']; C = fit['C']
            plt.semilogy(t, A*np.exp(-t/tau)+C, 'r--', label=f"exp fit τ≈{tau:.2f}")
        plt.xlabel('iteration'); plt.ylabel('||ρ−1||_2'); plt.legend(); plt.tight_layout()
        plt.savefig('outputs/phase4_unification_enhanced_longrun_decay_v2.png', dpi=150)
        plt.close()

    # sanitize fit for JSON
    fit_clean = {k: (float(v) if isinstance(v, (np.floating, float)) else (float(v) if hasattr(v, 'item') else v)) for k, v in fit.items()}
    out = dict(
        iters=int(len(all_hist)),
        first_norm=float(all_hist[0]) if len(all_hist)>0 else None,
        last_norm=float(all_hist[-1]) if len(all_hist)>0 else None,
        fit=fit_clean,
        hist_norm=[float(x) for x in all_hist],
        hist_norm_A=[float(x) for x in resA['hist_norm']],
        hist_norm_B_segments=[[float(x) for x in seg['hist_norm']] for seg in B_segments],
    )
    with open('outputs/phase4_unification_enhanced_longrun_v2.json', 'w') as f:
        json.dump(out, f, indent=2)

    print('Long-run enhanced complete.')
    if out['iters']>0:
        print(f"iters={out['iters']}, first_norm={out['first_norm']:.6f}, last_norm={out['last_norm']:.6f}")
        print(f"fit: {out['fit']}")
    return out


if __name__ == '__main__':
    run_longrun()
