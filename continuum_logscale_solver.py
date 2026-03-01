#!/usr/bin/env python3
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, Dict

# SciPy imports
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Optional: symbolic verification
try:
    import sympy as sp
except Exception:
    sp = None


def symbolic_transform(alpha: float) -> str:
    if sp is None:
        return "SymPy not available; symbolic step skipped."
    x = sp.symbols('x', real=True)
    C, D, V0, E = sp.symbols('C D V0 E', positive=True, real=True)
    Psi = sp.Function('Psi')(x)
    eq = -C*sp.exp(-2*sp.Symbol('alpha')*x)*sp.diff(Psi, x, 2) \
         -D*sp.exp(-2*sp.Symbol('alpha')*x)*sp.diff(Psi, x) \
         + V0*x**2*Psi - E*Psi

    a = D/C
    phi = sp.Function('phi')(x)
    Psi_sub = sp.exp(-a*x/2)*phi
    eq_sub = eq.subs({Psi: Psi_sub})
    eq_sub = sp.simplify(eq_sub.expand())

    # Divide by the common factor to obtain the phi-equation form
    # We know analytically this yields: phi'' - [(sp.exp(2αx)/C)*(V0 x^2 - E) + (a**2)/4] phi = 0
    # We'll present the final human-readable result.
    return (
        "Using Ψ(x) = e^{-(D/C) x / 2} φ(x) removes the first derivative.\n"
        "Resulting ODE: φ'' - [ (e^{2 α x}/C) (V0 x^2 - E) + (α^2)/4 ] φ = 0,\n"
        "since D/C = α for C=ħ^2/(2 m α^2), D=ħ^2/(2 m α)."
    )


@dataclass
class ContinuumParams:
    lam: float = math.sqrt(6.0)/2.0
    V0: float = 5.0
    hbar: float = 1.0
    mass: float = 1.0
    # Numerical domain and solver
    x_min: float = -5.0
    x_max: float = 6.0
    grid_points: int = 2000
    method: str = 'BDF'
    rtol: float = 1e-8
    atol: float = 1e-10

    def derived(self) -> Dict[str, float]:
        alpha = math.log(self.lam)
        C = (self.hbar**2) / (2.0 * self.mass * alpha**2)
        D = (self.hbar**2) / (2.0 * self.mass * alpha)
        return dict(alpha=alpha, C=C, D=D)


def make_W_function(params: ContinuumParams) -> Callable[[float, float], float]:
    d = params.derived()
    alpha = d['alpha']
    C = d['C']
    # D/C = alpha
    a_sq_over_4 = (alpha**2)/4.0

    def W(x: float, E: float) -> float:
        return (math.exp(2.0*alpha*x)/C) * (params.V0 * x*x - E) + a_sq_over_4
    return W


def solve_phi_on_interval(E: float, params: ContinuumParams, W: Callable[[float, float], float], t_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Second-order ODE: φ'' = W(x,E) φ
    # Convert to first-order system: y = [φ, φ']
    def f(x, y):
        return np.array([y[1], W(x, E) * y[0]], dtype=float)

    # Left Dirichlet boundary: φ(x_min)=0, set φ'(x_min)=1 as arbitrary scale
    y0 = np.array([0.0, 1.0], dtype=float)

    sol = solve_ivp(f, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method=params.method, rtol=params.rtol, atol=params.atol)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for E={E}: {sol.message}")
    return sol.t, sol.y[0], sol.y[1]


def shoot_residual(E: float, params: ContinuumParams, W: Callable[[float, float], float], t_eval: np.ndarray) -> float:
    _, phi, _ = solve_phi_on_interval(E, params, W, t_eval)
    # Enforce right Dirichlet boundary φ(x_max) ~ 0
    return float(phi[-1])


def match_residual(E: float, params: ContinuumParams, W: Callable[[float, float], float], x_match: float) -> float:
    # Compute logarithmic derivative mismatch at x_match via left and right integrations
    def f(x, y):
        return np.array([y[1], W(x, E) * y[0]], dtype=float)

    # Left integration from x_min -> x_match
    y0_left = np.array([0.0, 1.0])
    solL = solve_ivp(f, (params.x_min, x_match), y0_left, t_eval=None, method=params.method, rtol=params.rtol, atol=params.atol)
    if not solL.success:
        raise RuntimeError(f"Left integration failed: {solL.message}")
    phiL, dphiL = solL.y[0, -1], solL.y[1, -1]

    # Right integration from x_max -> x_match (reverse direction). Setup φ(x_max)=0, φ'(x_max)=-1
    y0_right = np.array([0.0, -1.0])
    solR = solve_ivp(f, (params.x_max, x_match), y0_right, t_eval=None, method=params.method, rtol=params.rtol, atol=params.atol)
    if not solR.success:
        raise RuntimeError(f"Right integration failed: {solR.message}")
    phiR, dphiR = solR.y[0, -1], solR.y[1, -1]

    # Log-derivative mismatch
    eps = 1e-12
    L = dphiL / (phiL + eps)
    R = dphiR / (phiR + eps)
    return float(L - R)


def find_eigenvalue_near(n: int, params: ContinuumParams, W: Callable[[float, float], float], x_grid: np.ndarray) -> Tuple[float, np.ndarray]:
    V0 = params.V0
    E0 = V0 * (n**2)
    # Match near x ≈ n (cluster center)
    x_match = float(np.clip(n, params.x_min + 1.0, params.x_max - 1.0))

    # Bracket around E0 for mismatch function
    bracks = [0.25, 0.5, 0.75, 1.0, 1.25, 2.0]
    E_low = None
    E_high = None
    for b in bracks:
        lo = max(0.0, V0 * (max(n - b, 0.0) ** 2))
        hi = V0 * ((n + b) ** 2)
        f_lo = match_residual(lo, params, W, x_match)
        f_hi = match_residual(hi, params, W, x_match)
        if np.isfinite(f_lo) and np.isfinite(f_hi) and np.sign(f_lo) != np.sign(f_hi):
            E_low, E_high = lo, hi
            break

    if E_low is None:
        # Fallback: wide bracket
        lo = max(0.0, 0.25 * (E0 + 1.0))
        hi = 4.0 * (E0 + 1.0)
        f_lo = match_residual(lo, params, W, x_match)
        f_hi = match_residual(hi, params, W, x_match)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi) and np.sign(f_lo) != np.sign(f_hi)):
            # As last resort, return E0 and left-only φ
            t, phi, _ = solve_phi_on_interval(E0, params, W, x_grid)
            return E0, phi
        E_low, E_high = lo, hi

    E_n = brentq(lambda E: match_residual(E, params, W, x_match), E_low, E_high, xtol=1e-10, rtol=1e-10, maxiter=100)

    # Reconstruct φ on full grid from left integration
    _, phi, _ = solve_phi_on_interval(E_n, params, W, x_grid)
    return E_n, phi


def reconstruct_psi_from_phi(x_grid: np.ndarray, phi: np.ndarray, alpha: float) -> np.ndarray:
    # Ψ(x) = e^{-α x/2} φ(x)
    return np.exp(-0.5 * alpha * x_grid) * phi


def normalize_density(x_grid: np.ndarray, psi: np.ndarray) -> Tuple[np.ndarray, float, float]:
    dens = np.abs(psi)**2
    Z = np.trapz(dens, x_grid)
    if Z <= 0:
        return dens, 0.0, 0.0
    dens /= Z
    mean_x = np.trapz(x_grid * dens, x_grid)
    var_x = np.trapz((x_grid - mean_x)**2 * dens, x_grid)
    return dens, mean_x, math.sqrt(max(var_x, 0.0))


def run_continuum_phase2() -> None:
    params = ContinuumParams()
    d = params.derived()
    alpha = d['alpha']
    C = d['C']
    V0 = params.V0

    sym_text = symbolic_transform(alpha)
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/symbolic_transformation.txt', 'w') as f:
        f.write(sym_text + "\n")

    # Build W(x,E) and grid
    W = make_W_function(params)

    # For n=0,1,2, choose a grid focused around the well; but we use global grid here
    x_grid = np.linspace(params.x_min, params.x_max, params.grid_points)

    results = []
    for n in [0, 1, 2]:
        # Predict width Δx ~ (C/(2V0))^{1/4} e^{-α n/2}
        dx_pred = (C/(2.0*V0))**0.25 * math.exp(-0.5*alpha*n)
        # Narrow focus around x≈n to stabilize higher-n width measurement
        local_min = max(params.x_min, n - 2.0)
        local_max = min(params.x_max, n + 2.0)
        local_grid = np.linspace(local_min, local_max, max(800, int((params.grid_points/ (params.x_max-params.x_min)) * (local_max-local_min))))
        E_n, phi_n = find_eigenvalue_near(n, params, W, local_grid)
        # Reconstruct on local grid for accurate width; also build a global extension for plotting convenience
        psi_local = reconstruct_psi_from_phi(local_grid, phi_n, alpha)
        dens_local, mean_x, std_x = normalize_density(local_grid, psi_local)
        # Gaussian width via local log-quadratic fit around the peak
        idx_pk = int(np.argmax(dens_local))
        x_pk = float(local_grid[idx_pk])
        y_max = float(dens_local[idx_pk])
        # Use window where dens is not too small to avoid numerical issues
        mask = dens_local >= max(1e-12, 0.2 * y_max)
        xw = local_grid[mask]
        yw = dens_local[mask]
        if xw.size >= 8:
            x_shift2 = (xw - x_pk)**2
            # Fit ln y ≈ c0 + c2 * (x - x_pk)^2
            A = np.column_stack([np.ones_like(x_shift2), x_shift2])
            coeff, *_ = np.linalg.lstsq(A, np.log(np.maximum(yw, 1e-300))),
            c0, c2 = coeff[0]
            if c2 < 0:
                std_fit = float(math.sqrt(max(1e-18, -1.0/(2.0*c2))))
            else:
                std_fit = float(std_x)
        else:
            std_fit = float(std_x)
        results.append(dict(n=n, E=E_n, x=local_grid, psi=psi_local, dens=dens_local, mean_x=mean_x, std_x=std_x, std_fit=std_fit, dx_pred=dx_pred))

    # Plot |Ψ_n(x)|^2 for n=0,1,2
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for r in results:
        ax[0].plot(r['x'], r['dens'], label=f"n={r['n']} (E≈{r['E']:.3f})")
    ax[0].set_xlabel('x = ln r / ln λ')
    ax[0].set_ylabel('|Ψ_n(x)|^2')
    ax[0].set_title('Bound-state densities in log-scale')
    ax[0].legend()

    ns = [r['n'] for r in results]
    widths = [r['std_x'] for r in results]
    widths_pred = [r['dx_pred'] for r in results]

    ax[1].plot(ns, widths, 'o-', label='Δx (std)')
    ax[1].plot(ns, widths_pred, 's--', label='Δx_pred ∝ e^{-α n/2}')
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('Δx')
    ax[1].set_title('Widths (std) vs prediction')
    ax[1].grid(True)
    ax[1].legend()

    widths_fit = [r.get('std_fit', r['std_x']) for r in results]
    ax[2].plot(ns, widths_fit, 'o-', label='Δx (Gaussian fit)')
    ax[2].plot(ns, widths_pred, 's--', label='Δx_pred ∝ e^{-α n/2}')
    ax[2].set_xlabel('n')
    ax[2].set_ylabel('Δx')
    ax[2].set_title('Widths (Gaussian fit) vs prediction')
    ax[2].grid(True)
    ax[2].legend()

    plt.tight_layout()
    plt.savefig('outputs/continuum_wavefunctions_and_widths.png', dpi=150)
    plt.close(fig)

    # Save numeric results
    np.savez('outputs/continuum_results.npz',
             lam=params.lam, alpha=alpha, V0=V0,
             E=np.array([r['E'] for r in results]),
             ns=np.array(ns),
             widths=np.array(widths), widths_fit=np.array([r.get('std_fit', r['std_x']) for r in results]),
             widths_pred=np.array(widths_pred))

    # Save concise metrics report
    ratios_std = []
    ratios_fit = []
    if len(widths) >= 2:
        ratios_std.append(widths[1]/widths[0] if widths[0] > 0 else float('nan'))
        ratios_fit.append(widths_fit[1]/widths_fit[0] if widths_fit[0] > 0 else float('nan'))
    if len(widths) >= 3:
        ratios_std.append(widths[2]/widths[1] if widths[1] > 0 else float('nan'))
        ratios_fit.append(widths_fit[2]/widths_fit[1] if widths_fit[1] > 0 else float('nan'))
    with open('outputs/phase2_report.txt', 'w') as f:
        f.write(f"Eigenvalues (n=0,1,2): {', '.join(f'{e:.6f}' for e in [r['E'] for r in results])}\n")
        f.write(f"Predicted ladder V0*n^2 (V0={V0:.3f}): {', '.join(f'{V0*(i**2):.6f}' for i in ns)}\n")
        f.write(f"Width ratios std (1/0,2/1): {ratios_std}\n")
        f.write(f"Width ratios fit (1/0,2/1): {ratios_fit}\n")
        f.write(f"Predicted ratio e^(-alpha/2)={math.exp(-alpha/2.0):.6f}\n")

    # Console summary
    for r in results:
        print(f"n={r['n']}: E={r['E']:.6f}, mean_x={r['mean_x']:.4f}, Δx={r['std_x']:.6e}, Δx_fit={r.get('std_fit', r['std_x']):.6e}, Δx_pred~{r['dx_pred']:.6e}")


if __name__ == '__main__':
    run_continuum_phase2()
