#!/usr/bin/env python3
import json
import os
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def _r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-18
    return 1.0 - ss_res / ss_tot


def fit_exp_segment(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n < 3:
        return dict(model='single', A=np.nan, tau=np.nan, C=np.nan, r2=np.nan, n=n)
    t = np.arange(n, dtype=float)

    def f_single(t, A, tau, C):
        return C + A * np.exp(-t / np.maximum(tau, 1e-12))

    # Initial guesses
    C0 = float(y[-1])
    A0 = float(max(y[0] - C0, 1e-12))
    tau0 = max(n / 10.0, 1.0)
    try:
        popt, _ = curve_fit(
            f_single, t, y,
            p0=[A0, tau0, C0],
            bounds=([-np.inf, 1e-6, min(y) - abs(y.ptp())], [np.inf, 1e6, max(y) + abs(y.ptp())]),
            maxfev=20000,
        )
        A, tau, C = float(popt[0]), float(popt[1]), float(popt[2])
        yfit = f_single(t, A, tau, C)
        r2 = _r2(y, yfit)
        return dict(model='single', A=A, tau=tau, C=C, r2=r2, n=n)
    except Exception:
        return dict(model='single', A=np.nan, tau=np.nan, C=np.nan, r2=np.nan, n=n)


def fit_double_exp_segment(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n < 5:
        return dict(model='double', A1=np.nan, t1=np.nan, A2=np.nan, t2=np.nan, C=np.nan, r2=np.nan, n=n)
    t = np.arange(n, dtype=float)

    def f_double(t, A1, t1, A2, t2, C):
        t1 = np.maximum(t1, 1e-12)
        t2 = np.maximum(t2, 1e-12)
        return C + A1 * np.exp(-t / t1) + A2 * np.exp(-t / t2)

    C0 = float(y[-1])
    A0 = float(max(y[0] - C0, 1e-12))
    A1_0 = 0.7 * A0
    A2_0 = 0.3 * A0
    t1_0 = max(n / 20.0, 1.0)
    t2_0 = max(n / 2.0, 5.0)
    try:
        popt, _ = curve_fit(
            f_double, t, y,
            p0=[A1_0, t1_0, A2_0, t2_0, C0],
            bounds=([-np.inf, 1e-6, -np.inf, 1e-6, min(y) - abs(y.ptp())], [np.inf, 1e6, np.inf, 1e6, max(y) + abs(y.ptp())]),
            maxfev=40000,
        )
        A1, t1, A2, t2, C = map(float, popt)
        yfit = f_double(t, A1, t1, A2, t2, C)
        r2 = _r2(y, yfit)
        return dict(model='double', A1=A1, t1=t1, A2=A2, t2=t2, C=C, r2=r2, n=n)
    except Exception:
        return dict(model='double', A1=np.nan, t1=np.nan, A2=np.nan, t2=np.nan, C=np.nan, r2=np.nan, n=n)


def main() -> None:
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/phase4_unification_enhanced_longrun_v2.json') as f:
        R = json.load(f)
    # Extract segments
    yA = np.array(R.get('hist_norm_A', []), dtype=float)
    yB_list = [np.array(seg, dtype=float) for seg in R.get('hist_norm_B_segments', [])]
    # Fit stage A (single and double); choose better by R²
    fitA_single = fit_exp_segment(yA) if len(yA) >= 3 else dict(model='single', A=np.nan, tau=np.nan, C=np.nan, r2=np.nan, n=len(yA))
    fitA_double = fit_double_exp_segment(yA) if len(yA) >= 5 else dict(model='double', A1=np.nan, t1=np.nan, A2=np.nan, t2=np.nan, C=np.nan, r2=np.nan, n=len(yA))
    fitA = fitA_double if (np.isfinite(fitA_double.get('r2', np.nan)) and fitA_double.get('r2', -np.inf) > fitA_single.get('r2', -np.inf)) else fitA_single
    # Fit Stage B segments similarly
    fitB_list = []
    for yB in yB_list:
        fs = fit_exp_segment(yB)
        fd = fit_double_exp_segment(yB)
        fb = fd if (np.isfinite(fd.get('r2', np.nan)) and fd.get('r2', -np.inf) > fs.get('r2', -np.inf)) else fs
        fitB_list.append(fb)

    # Create diagnostic plots: Stage A and each Stage B segment with best-fit (single/double)
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
            if fit.get('model') == 'single' and np.isfinite(fit.get('tau', np.nan)):
                A = fit['A']; tau = fit['tau']; C = fit['C']
                yfit = C + A * np.exp(-t / (tau if tau not in (0, np.inf) else 1.0))
                ax.semilogy(t, yfit, 'r--', label=f'single τ≈{tau:.2f}, R²≈{fit["r2"]:.3f}')
            elif fit.get('model') == 'double' and np.isfinite(fit.get('t1', np.nan)):
                A1 = fit['A1']; t1 = fit['t1']; A2 = fit['A2']; t2 = fit['t2']; C = fit['C']
                yfit = C + A1 * np.exp(-t / max(t1, 1e-12)) + A2 * np.exp(-t / max(t2, 1e-12))
                ax.semilogy(t, yfit, 'r--', label=f'double t1≈{t1:.2f}, t2≈{t2:.2f}, R²≈{fit["r2"]:.3f}')
            ax.set_xlabel('iteration (segment)')
            ax.set_ylabel('||ρ−1||_2')
            ax.set_title(f'{name} (n={len(y)})')
            ax.legend()
        else:
            ax.set_visible(False)
    plt.tight_layout()
    plt.savefig('outputs/phase4_unification_enhanced_piecewise_decay.png', dpi=150)
    plt.close()

    # Write human-readable report
    lines: List[str] = []
    lines.append('Piecewise exponential decay report')
    lines.append('')
    if fitA.get('model') == 'single':
        lines.append(f"Stage A (fast, single): n={fitA.get('n')}, A={fitA.get('A')}, tau={fitA.get('tau')}, C={fitA.get('C')}, R2={fitA.get('r2')}")
    else:
        lines.append(f"Stage A (fast, double): n={fitA.get('n')}, A1={fitA.get('A1')}, t1={fitA.get('t1')}, A2={fitA.get('A2')}, t2={fitA.get('t2')}, C={fitA.get('C')}, R2={fitA.get('r2')}")
    for i, fb in enumerate(fitB_list):
        if fb.get('model') == 'single':
            lines.append(f"Stage B{i+1} (slow, single): n={fb.get('n')}, A={fb.get('A')}, tau={fb.get('tau')}, C={fb.get('C')}, R2={fb.get('r2')}")
        else:
            lines.append(f"Stage B{i+1} (slow, double): n={fb.get('n')}, A1={fb.get('A1')}, t1={fb.get('t1')}, A2={fb.get('A2')}, t2={fb.get('t2')}, C={fb.get('C')}, R2={fb.get('r2')}")
    lines.append('')
    lines.append(f"Composite iters={R.get('iters')}, first_norm={R.get('first_norm')}, last_norm={R.get('last_norm')}")
    with open('outputs/phase4_unification_enhanced_piecewise_report.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Write CSV for quantitative analysis
    import csv
    csv_rows: List[Dict[str, float]] = []
    # Stage A row
    rowA = {'segment': 'Stage A', 'n': fitA.get('n'), 'model': fitA.get('model'), 'R2': fitA.get('r2')}
    if fitA.get('model') == 'single':
        rowA.update({'A': fitA.get('A'), 'tau': fitA.get('tau'), 'C': fitA.get('C')})
    else:
        rowA.update({'A1': fitA.get('A1'), 't1': fitA.get('t1'), 'A2': fitA.get('A2'), 't2': fitA.get('t2'), 'C': fitA.get('C')})
    csv_rows.append(rowA)
    # Stage B rows
    for i, fb in enumerate(fitB_list):
        row = {'segment': f'Stage B{i+1}', 'n': fb.get('n'), 'model': fb.get('model'), 'R2': fb.get('r2')}
        if fb.get('model') == 'single':
            row.update({'A': fb.get('A'), 'tau': fb.get('tau'), 'C': fb.get('C')})
        else:
            row.update({'A1': fb.get('A1'), 't1': fb.get('t1'), 'A2': fb.get('A2'), 't2': fb.get('t2'), 'C': fb.get('C')})
        csv_rows.append(row)

    # Determine all CSV fieldnames
    fieldnames = sorted({k for r in csv_rows for k in r.keys()})
    with open('outputs/phase4_unification_enhanced_piecewise_fits.csv', 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    print('Piecewise report complete.')


if __name__ == '__main__':
    main()
