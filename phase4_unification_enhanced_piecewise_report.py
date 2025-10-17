#!/usr/bin/env python3
import json
import os
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fit_exp_segment(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n < 3:
        return dict(A=np.nan, tau=np.nan, C=np.nan, r2=np.nan, n=n)
    C = float(y[-1])
    y_shift = np.maximum(y - C, 1e-18)
    t = np.arange(n, dtype=float)
    X = np.vstack([t, np.ones_like(t)]).T
    beta, *_ = np.linalg.lstsq(X, -np.log(y_shift), rcond=None)
    if beta[0] <= 0:
        tau = np.inf
    else:
        tau = 1.0 / beta[0]
    A = float(math.exp(beta[1]))
    y_pred = A * np.exp(-t / (tau if tau != np.inf else 1.0)) + C
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-18
    r2 = 1.0 - ss_res / ss_tot
    return dict(A=A, tau=tau, C=C, r2=r2, n=n)


def main() -> None:
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/phase4_unification_enhanced_longrun_v2.json') as f:
        R = json.load(f)
    # Extract segments
    yA = np.array(R.get('hist_norm_A', []), dtype=float)
    yB_list = [np.array(seg, dtype=float) for seg in R.get('hist_norm_B_segments', [])]
    # Fit stage A (fast timescale)
    fitA = fit_exp_segment(yA) if len(yA) >= 3 else dict(A=np.nan, tau=np.nan, C=np.nan, r2=np.nan, n=len(yA))
    # Fit each stage B segment individually (slow timescales)
    fitB_list = [fit_exp_segment(yB) for yB in yB_list]
    fitB_last = fitB_list[-1] if fitB_list else dict(A=np.nan, tau=np.nan, C=np.nan, r2=np.nan, n=0)

    # Create diagnostic plots: Stage A and each Stage B segment
    segs = [('Stage A', yA, fitA)] + [(f'Stage B{i+1}', yB_list[i], fitB_list[i]) for i in range(len(yB_list))]
    nplots = len(segs)
    nrows = int(math.ceil(nplots / 2))
    ncols = 2 if nplots > 1 else 1
    plt.figure(figsize=(10, 4 * nrows))
    for idx, (name, y, fit) in enumerate(segs):
        ax = plt.subplot(nrows, ncols, idx + 1)
        if len(y) > 0:
            t = np.arange(len(y), dtype=float)
            ax.semilogy(t, y, 'o-', label=f'{name}')
            if np.isfinite(fit.get('tau', np.nan)):
                A = fit['A']; tau = fit['tau']; C = fit['C']
                yfit = A * np.exp(-t / (tau if tau not in (0, np.inf) else 1.0)) + C
                ax.semilogy(t, yfit, 'r--', label=f'fit τ≈{tau:.2f}, R²≈{fit["r2"]:.3f}')
            ax.set_xlabel('iteration (segment)')
            ax.set_ylabel('||ρ−1||_2')
            ax.set_title(f'{name} (n={len(y)})')
            ax.legend()
        else:
            ax.set_visible(False)
    plt.tight_layout()
    plt.savefig('outputs/phase4_unification_enhanced_piecewise_decay.png', dpi=150)
    plt.close()

    # Write report
    lines: List[str] = []
    lines.append('Piecewise exponential decay report')
    lines.append('')
    lines.append(f"Stage A (fast): n={fitA.get('n')}, A={fitA.get('A')}, tau={fitA.get('tau')}, C={fitA.get('C')}, R2={fitA.get('r2')}")
    for i, fb in enumerate(fitB_list):
        lines.append(f"Stage B{i+1} (slow): n={fb.get('n')}, A={fb.get('A')}, tau={fb.get('tau')}, C={fb.get('C')}, R2={fb.get('r2')}")
    lines.append('')
    lines.append(f"Composite iters={R.get('iters')}, first_norm={R.get('first_norm')}, last_norm={R.get('last_norm')}")
    with open('outputs/phase4_unification_enhanced_piecewise_report.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('Piecewise report complete.')


if __name__ == '__main__':
    main()
