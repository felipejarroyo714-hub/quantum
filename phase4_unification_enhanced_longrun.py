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
        max_iters=6000,
        du_cap=0.30,
        ls_rel_tol=2e-5,
        ls_abs_tol=1e-8,
        robust_quantile=0.90,
        lambda_Q=0.20,
        dt_init=1e-3,
        dt_min=1e-10,
        k_eig=18,
        local_smooth_window=15,
        epsilon0=0.30,
        num_z=700,
    )
    # Apply tiny positive acceptance slack to avoid numerical pinning
    try:
        setattr(p, 'ls_pos_slack', 1e-6)
    except Exception:
        pass
    res = simulate_one(p, long_time=True)

    # Fit exponential to decay of ||rho-1||_2
    t = np.arange(len(res['hist_norm']), dtype=float)
    fit = fit_exponential(t, res['hist_norm'])

    # Plot decay
    if len(t) > 0:
        plt.figure(figsize=(6,4))
        plt.semilogy(t, res['hist_norm'], 'o-', label='||ρ−1||_2')
        if np.isfinite(fit.get('tau', float('nan'))):
            A = fit['A']; tau = fit['tau']; C = fit['C']
            plt.semilogy(t, A*np.exp(-t/tau)+C, 'r--', label=f"exp fit τ≈{tau:.2f}")
        plt.xlabel('iteration'); plt.ylabel('||ρ−1||_2'); plt.legend(); plt.tight_layout()
        plt.savefig('outputs/phase4_unification_enhanced_longrun_decay_v2.png', dpi=150)
        plt.close()

    # sanitize fit for JSON
    fit_clean = {k: (float(v) if isinstance(v, (np.floating, float)) else (float(v) if hasattr(v, 'item') else v)) for k, v in fit.items()}
    out = dict(
        iters=int(len(res['hist_norm'])),
        first_norm=float(res['hist_norm'][0]) if len(res['hist_norm'])>0 else None,
        last_norm=float(res['hist_norm'][-1]) if len(res['hist_norm'])>0 else None,
        fit=fit_clean,
        hist_norm=[float(x) for x in res['hist_norm']],
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
