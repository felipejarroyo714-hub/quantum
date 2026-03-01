#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Callable, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


@dataclass
class GeometryParams:
    lam: float = math.sqrt(6.0)/2.0  # λ
    z_min: float = -10.0
    z_max: float = 10.0
    num_z: int = 1200
    r0: float = 1.0
    # optional scale fluctuations (λ-periodic in x = ln r / ln λ)
    epsilon: float = 0.0  # set small e.g. 0.05 for fluctuations

    def derived(self) -> Dict[str, float]:
        alpha = math.log(self.lam)
        return dict(alpha=alpha)


@dataclass
class FieldParams:
    mu: float = 0.5  # field mass m
    xi: float = 0.0  # curvature coupling (conformal in 2D is 0)
    m_theta: int = 0  # angular momentum quantum number on S^1 fiber
    k_eig: int = 40   # number of modes to compute


# --- Profile r(z) from master ODE r' = h(r) ---

def make_h_function(lam: float, epsilon: float = 0.0) -> Callable[[float], float]:
    alpha = math.log(lam)
    def h(r: float) -> float:
        if r <= 0:
            return 0.0
        if epsilon == 0.0:
            return alpha * r
        # λ-periodic modulation in x = ln r / ln λ
        x = math.log(r) / alpha
        return alpha * r * (1.0 + epsilon * math.cos(2.0 * math.pi * x))
    return h


def integrate_profile(params: GeometryParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = params.derived()
    alpha = d['alpha']
    z_grid = np.linspace(params.z_min, params.z_max, params.num_z)
    dz = z_grid[1] - z_grid[0]

    h = make_h_function(params.lam, params.epsilon)

    r = np.empty_like(z_grid)
    r[0] = params.r0
    # simple forward integration (explicit Euler) with stabilization; small dz chosen
    for i in range(1, params.num_z):
        r[i] = max(1e-12, r[i-1] + dz * h(r[i-1]))

    # index density ρ(z) = h(r)/(ln λ ⋅ r)
    rho = np.array([h(rv) / (alpha * rv) for rv in r])

    # curvature K = - r'' / r; in 2D, Ricci scalar R = 2 K
    # compute r' ≈ h(r), r'' ≈ h'(r) h(r)
    def h_prime(rval: float) -> float:
        if params.epsilon == 0.0:
            return alpha
        if rval <= 0:
            return alpha
        x = math.log(rval) / alpha
        # derivative wrt r of alpha*r*(1 + eps cos(2π x))
        # h' = alpha*(1 + eps cos(2π x)) + alpha*r*(-eps*2π sin(2π x)) * d x/dr
        # d x/dr = 1/(r * alpha)
        return alpha * (1.0 + params.epsilon * math.cos(2.0 * math.pi * x) \
                        - params.epsilon * (2.0 * math.pi) * math.sin(2.0 * math.pi * x) / alpha)

    hp = np.array([h_prime(rv) for rv in r])
    rp = np.array([h(rv) for rv in r])
    rpp = hp * rp
    with np.errstate(divide='ignore', invalid='ignore'):
        K = -rpp / np.clip(r, 1e-18, None)
        R = 2.0 * K

    return z_grid, r, rho, R


# --- Laplace-Beltrami and KG spatial operator ---

def build_kg_operator(z: np.ndarray, r: np.ndarray, R: np.ndarray, field: FieldParams) -> Tuple[diags, np.ndarray]:
    n = len(z)
    dz = z[1] - z[0]

    # Metric: ds^2 = dz^2 + r(z)^2 dθ^2
    # Laplace-Beltrami on axisymmetric warped product: Δ = ∂_z^2 + (r'/r) ∂_z + (1/r^2) ∂_θ^2
    # We discretize -Δ + (mu^2 + xi R) as symmetric tridiagonal in z for fixed m_theta (separation)

    # Compute r' numerically for better stability even if r'≈h(r)
    rp = np.gradient(r, dz)

    # Coefficients for -[u'' + (r'/r) u'] term using central differences
    # Discretization: u'' ≈ (u_{i-1} - 2u_i + u_{i+1})/dz^2
    # First-derivative term handled in symmetric form via flux: - (1/r) d/dz ( r du/dz )
    # This yields a symmetric stencil:
    r_mid_plus = 0.5 * (r[1:] + r[:-1])
    r_mid_minus = r_mid_plus

    main = np.zeros(n)
    off = np.zeros(n-1)

    # interior points i=1..n-2
    for i in range(1, n-1):
        a_plus = r_mid_plus[i] / (r[i] * dz * dz)
        a_minus = r_mid_minus[i-1] / (r[i] * dz * dz)
        main[i] = a_plus + a_minus
        off[i-1] = -a_minus
        # we will add off[i] later for a_plus at i contributing to (i,i+1)

    # boundary conditions: Dirichlet u=0 at both ends
    main[0] = 1.0
    main[-1] = 1.0

    # assemble upper off-diagonal for symmetry
    off_upper = np.zeros(n-1)
    for i in range(1, n-1):
        a_plus = r_mid_plus[i] / (r[i] * dz * dz)
        off_upper[i] = -a_plus

    # angular and mass/curvature terms
    ang_term = (field.m_theta**2) / np.clip(r**2, 1e-18, None)
    pot = ang_term + (field.mu**2 + field.xi * R)

    # add potential to main diagonal
    main += pot

    A = diags([off, main, off_upper], offsets=[-1, 0, 1], format='csr')
    return A, pot


def compute_modes(A, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # Solve A u = ω^2 u for lowest eigenpairs
    n = A.shape[0]
    k = min(k, n-2)
    evals, evecs = eigsh(A, k=k, which='SA')
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def normalize_on_z(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    norm = math.sqrt(np.trapezoid(u*u, z))
    return u / (norm + 1e-18)


def check_lambda_covariance(
    params: GeometryParams,
    field: FieldParams,
    shift_steps: int = 1,
) -> Dict[str, float]:
    """Evaluate how well eigenmodes align after a λ-rescaling.

    Parameters
    ----------
    params
        Baseline geometric parameters.
    field
        Field parameters used to build the Klein–Gordon operator.
    shift_steps
        Number of logarithmic scale steps (integer multiples of ln λ) used for the
        comparison. ``shift_steps=1`` reproduces the original behaviour where the
        background is compared against a single λ-rescaling. Larger values probe
        multi-step covariance.
    """

    if shift_steps < 1:
        raise ValueError("shift_steps must be a positive integer")

    # Background A: r0
    zA, rA, rhoA, RA = integrate_profile(params)
    A, _ = build_kg_operator(zA, rA, RA, field)
    w2A, vA = compute_modes(A, k=min(20, field.k_eig))

    # Background B: r0 * λ (equiv. z-shift by +1 since r(z+1)=λ r(z) for h=α r)
    paramsB = GeometryParams(
        lam=params.lam,
        z_min=params.z_min,
        z_max=params.z_max,
        num_z=params.num_z,
        r0=params.r0 * (params.lam ** shift_steps),
        epsilon=params.epsilon,
    )
    zB, rB, rhoB, RB = integrate_profile(paramsB)
    B, _ = build_kg_operator(zB, rB, RB, field)
    w2B, vB = compute_modes(B, k=min(20, field.k_eig))

    # Compare shapes after z-shift alignment by Δz = shift_steps
    delta = float(shift_steps)
    # shift vB by Δz: we need to interpolate onto zA grid
    overlaps = []
    for j in range(min(vA.shape[1], vB.shape[1])):
        uA = normalize_on_z(zA, vA[:, j])
        # Build interpolant of uB at shifted coordinates
        z_shift = zA + delta
        uB_shift = np.interp(z_shift, zB, normalize_on_z(zB, vB[:, j]), left=0.0, right=0.0)
        # Compute overlap
        ov = np.trapezoid(uA * uB_shift, zA)
        overlaps.append(abs(ov))

    # Return simple metrics
    return {
        'w2A_min': float(w2A[0]), 'w2B_min': float(w2B[0]),
        'mean_overlap': float(np.mean(overlaps)), 'max_overlap': float(np.max(overlaps)),
        'num_compared': int(len(overlaps))
    }


def run_phase3() -> None:
    geo = GeometryParams()
    field = FieldParams(mu=0.5, xi=0.0, m_theta=0, k_eig=40)

    z, r, rho, R = integrate_profile(geo)
    A, pot = build_kg_operator(z, r, R, field)
    w2, modes = compute_modes(A, k=field.k_eig)

    # Save and plot diagnostics
    os.makedirs('outputs', exist_ok=True)

    # Plot background: r(z), ρ(z), R(z)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(z, r)
    ax[0].set_ylabel('r(z)')
    ax[0].set_title('Scale-invariant axisymmetric profile')
    ax[1].plot(z, rho)
    ax[1].set_ylabel('ρ(z)')
    ax[2].plot(z, R)
    ax[2].set_ylabel('R(z)')
    ax[2].set_xlabel('z')
    plt.tight_layout()
    plt.savefig('outputs/phase3_background.png', dpi=150)
    plt.close(fig)

    # Plot a few lowest modes
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for j in range(min(5, modes.shape[1])):
        ax.plot(z, normalize_on_z(z, modes[:, j]), label=f'j={j}, ω^2={w2[j]:.3f}')
    ax.set_xlabel('z')
    ax.set_ylabel('mode amplitude')
    ax.set_title('Lowest normal modes (m_theta=0)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/phase3_modes.png', dpi=150)
    plt.close(fig)

    # Covariance under λ-rescaling
    cov_metrics = check_lambda_covariance(geo, field)

    # Save results
    np.savez('outputs/phase3_results.npz', z=z, r=r, rho=rho, R=R, w2=w2, pot=pot, modes=modes)
    with open('outputs/phase3_covariance.txt', 'w') as f:
        for k, v in cov_metrics.items():
            f.write(f'{k}: {v}\n')

    print(f"Computed {len(w2)} modes. Lowest ω^2={w2[0]:.6f}. Covariance overlap≈{cov_metrics['mean_overlap']:.3f} (max {cov_metrics['max_overlap']:.3f}).")


if __name__ == '__main__':
    run_phase3()
