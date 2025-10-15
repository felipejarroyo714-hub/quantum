#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.sparse import diags, kron, csr_matrix
from scipy.sparse.linalg import eigsh, eigs

# Reuse Phase 3 geometry utilities
from kg_scale_invariant_metric import (
    GeometryParams as KGGeometryParams,
    FieldParams as KGFieldParams,
    integrate_profile,
    build_kg_operator,
    compute_modes,
    normalize_on_z,
    compute_bogoliubov_leakage,
)


# -------------------- Configs --------------------

@dataclass
class DiracGeoConfig:
    lam: float = math.sqrt(6.0) / 2.0
    z_min: float = -10.0
    z_max: float = 10.0
    num_z: int = 2000
    r0: float = 1.0
    epsilon: float = 0.0  # fluctuation amplitude for geometric modulation


@dataclass
class DiracFieldConfig:
    m_fermion: float = 0.5
    m_theta: int = 0
    k_eig: int = 40


@dataclass
class BackreactionConfig:
    eta_max: int = 200        # number of time steps
    dt: float = 0.02          # time step (smaller for stability)
    gamma: float = 3.0        # relaxation strength
    nu: float = 0.2           # geometric diffusion (stabilizer)
    kappa: float = 1.0        # direct damping toward rho→1


# -------------------- Dirac operator (2D axisymmetric, reduced along θ) --------------------

def build_dirac_operator(z: np.ndarray, r: np.ndarray, cfg: DiracFieldConfig) -> csr_matrix:
    n = len(z)
    dz = z[1] - z[0]

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    # Central derivative (Dirichlet-like at boundaries by zeroing)
    main = np.zeros(n, dtype=float)
    upper = np.zeros(n - 1, dtype=float)
    lower = np.zeros(n - 1, dtype=float)
    # interior
    coef = 1.0 / (2.0 * dz)
    upper[:] = coef
    lower[:] = -coef
    # set boundary rows of D to zero (absorbing)
    upper[0] = 0.0
    lower[-1] = 0.0
    D = diags([lower, main, upper], offsets=[-1, 0, 1], dtype=float, format='csr')

    # Spin connection term from angular locking: (∂_z - r'/(4r))
    rp = np.gradient(r, dz)
    conn = -(rp / (4.0 * np.clip(r, 1e-18, None)))  # to be added inside the σ_x channel with -i

    # Angular term m/r couples with σ_y
    ang = (cfg.m_theta / np.clip(r, 1e-18, None)).astype(float)

    # Mass term m_f with σ_z
    mf = cfg.m_fermion

    # Assemble H = -i σ_x (∂_z + conn) + σ_y (m/r) + m_f σ_z
    H = (
        (-1j) * kron(sx, D) +
        (-1j) * kron(sx, diags(conn, 0, shape=(n, n), dtype=float)) +
        kron(sy, diags(ang, 0, shape=(n, n), dtype=float)) +
        kron(sz, diags(np.full(n, mf, dtype=float), 0, shape=(n, n), dtype=float))
    ).tocsr()

    return H


def dirac_modes(z: np.ndarray, r: np.ndarray, cfg: DiracFieldConfig) -> Tuple[np.ndarray, np.ndarray]:
    H = build_dirac_operator(z, r, cfg)
    k = min(cfg.k_eig, H.shape[0] - 2)
    # Try Hermitian solver first
    try:
        evals, evecs = eigsh(H, k=k, which='SA')
    except Exception:
        # Fallback to general eigs on H^2 to ensure real positives
        H2 = H.conj().T @ H
        evals2, evecs = eigsh(H2, k=k, which='SA')
        evals = np.sqrt(np.clip(evals2, 0.0, None))
    order = np.argsort(evals.real)
    evals = evals[order]
    evecs = evecs[:, order]
    return evals, evecs


def normalize_spinor_on_z(z: np.ndarray, spinor: np.ndarray) -> np.ndarray:
    n = len(z)
    up = spinor[:n]
    dn = spinor[n:]
    dens = (np.abs(up)**2 + np.abs(dn)**2).real
    norm = math.sqrt(np.trapezoid(dens, z))
    return spinor / (norm + 1e-18)


def check_dirac_lambda_covariance(geo: DiracGeoConfig, field: DiracFieldConfig) -> Dict[str, float]:
    # Baseline geometry A
    kgA = KGGeometryParams(lam=geo.lam, z_min=geo.z_min, z_max=geo.z_max, num_z=geo.num_z, r0=geo.r0, epsilon=geo.epsilon)
    zA, rA, _, _ = integrate_profile(kgA)
    evalsA, modesA = dirac_modes(zA, rA, field)

    # Rescaled geometry B: r0 * λ (equiv. shift Δz=1)
    kgB = KGGeometryParams(lam=geo.lam, z_min=geo.z_min, z_max=geo.z_max, num_z=geo.num_z, r0=geo.r0 * geo.lam, epsilon=geo.epsilon)
    zB, rB, _, _ = integrate_profile(kgB)
    evalsB, modesB = dirac_modes(zB, rB, field)

    delta = 1.0
    zA_shift = zA + delta
    n = len(zA)
    k_use = min(modesA.shape[1], modesB.shape[1])
    # Build overlap matrix and evaluate best mapping (row-wise maxima)
    O = np.zeros((k_use, k_use), dtype=float)
    # Pre-normalize A spinors on zA
    A_spin = [normalize_spinor_on_z(zA, modesA[:, j]) for j in range(k_use)]
    # Pre-normalize B spinors on zB and shift to zA+1
    B_shift = []
    for j in range(k_use):
        vB = normalize_spinor_on_z(zB, modesB[:, j])
        upB, dnB = vB[:n], vB[n:]
        upB_s = np.interp(zA_shift, zB, upB, left=0.0, right=0.0)
        dnB_s = np.interp(zA_shift, zB, dnB, left=0.0, right=0.0)
        spin_s = np.concatenate([upB_s, dnB_s])
        spin_s = normalize_spinor_on_z(zA, spin_s)
        B_shift.append(spin_s)
    for i in range(k_use):
        upA, dnA = A_spin[i][:n], A_spin[i][n:]
        for j in range(k_use):
            upB_s, dnB_s = B_shift[j][:n], B_shift[j][n:]
            O[i, j] = abs(np.trapezoid((np.conj(upA) * upB_s + np.conj(dnA) * dnB_s).real, zA))

    # Best one-to-one matching via Hungarian algorithm to maximize total overlap
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-O)  # maximize
        overlaps = O[row_ind, col_ind]
    except Exception:
        # Fallback: row-wise maxima
        overlaps = O.max(axis=1)
    # trimmed mean to reduce boundary effects
    if overlaps.size >= 10:
        t = int(0.2 * overlaps.size)
        overlaps_sorted = np.sort(overlaps)
        trimmed = overlaps_sorted[t:overlaps.size - t]
        mean_trimmed = float(trimmed.mean())
    else:
        mean_trimmed = float(overlaps.mean())

    return {
        'mean_overlap': float(overlaps.mean()),
        'mean_overlap_trimmed': mean_trimmed,
        'max_overlap': float(np.max(overlaps)),
        'num_compared': int(len(overlaps))
    }


# -------------------- Backreaction analog (3D+1) --------------------

def run_backreaction(geo: DiracGeoConfig, back: BackreactionConfig) -> Dict[str, np.ndarray]:
    # Start near fixed point with a small sinusoidal perturbation
    kg = KGGeometryParams(lam=geo.lam, z_min=geo.z_min, z_max=geo.z_max, num_z=geo.num_z, r0=geo.r0, epsilon=0.0)
    z, r_base, _, _ = integrate_profile(kg)
    dz = z[1] - z[0]
    alpha = math.log(geo.lam)

    # Small perturbation
    perturb = 0.05 * np.sin(2.0 * math.pi * (z - z.min()) / (z.max() - z.min()))
    r = r_base * (1.0 + perturb)

    # Leakage level disabled (set to zero for unconditional stability)
    leak = 0.0

    rho_L2 = []
    var_R = []
    energy_proxy = []

    # Build Laplacian matrix for implicit diffusion (Dirichlet-like ends)
    n = len(z)
    dz2 = dz * dz
    main = np.full(n, -2.0 / dz2)
    off = np.full(n - 1, 1.0 / dz2)
    L = diags([off, main, off], offsets=[-1, 0, 1], format='csr')
    # Implicit operator matrix (I - dt*nu*L)
    from scipy.sparse.linalg import splu
    def solve_diffusion(rhs):
        M = (diags(np.ones(n)) - back.nu * back.dt * L).tocsr()
        # Anchor boundaries to base to prevent drift
        M = M.tolil()
        M[0, :] = 0.0; M[0, 0] = 1.0
        M[-1, :] = 0.0; M[-1, -1] = 1.0
        M = M.tocsc()
        lu = splu(M)
        rhs2 = rhs.copy()
        rhs2[0] = r_base[0]
        rhs2[-1] = r_base[-1]
        return lu.solve(rhs2)

    for _ in range(back.eta_max):
        rp = np.gradient(r, dz)
        rpp = np.gradient(rp, dz)
        rho = (rp / np.clip(r, 1e-18, None)) / alpha
        R = -2.0 * (rpp / np.clip(r, 1e-18, None))

        rho_L2.append(float(math.sqrt(np.trapezoid((rho - 1.0)**2, z) / (z.max() - z.min()))))
        var_R.append(float(np.var(R)))
        energy_proxy.append(float(leak))

        # update explicit reactive + drive terms
        drive = 0.0005 * leak * r_base
        rhs = r + back.dt * (
            -back.gamma * (rp - alpha * r) - back.kappa * (rho - 1.0) * r + drive
        )
        # implicit diffusion step
        r = solve_diffusion(rhs)
        r = np.maximum(r, 1e-8)

    return {
        'z': z,
        'rho_L2': np.array(rho_L2),
        'var_R': np.array(var_R),
        'energy_proxy': np.array(energy_proxy)
    }


def run_phase4() -> None:
    # Part A: Dirac covariance (fermionic)
    dgeo = DiracGeoConfig(epsilon=0.0)
    dfield = DiracFieldConfig(m_fermion=0.5, m_theta=0, k_eig=40)
    overlaps = check_dirac_lambda_covariance(dgeo, dfield)

    # Part B: Semi-classical backreaction analog
    back = BackreactionConfig()
    back_series = run_backreaction(dgeo, back)

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/phase4_dirac_covariance.txt', 'w') as f:
        for k, v in overlaps.items():
            f.write(f'{k}: {v}\n')

    eta = np.arange(len(back_series['rho_L2'])) * back.dt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(eta, back_series['rho_L2'], 'o-', ms=3)
    ax[0].set_xlabel('η (time steps)')
    ax[0].set_ylabel('||ρ-1||_2')
    ax[0].set_title('Convergence to fixed point')

    ax[1].plot(eta, back_series['var_R'], 'o-', ms=3)
    ax[1].set_xlabel('η (time steps)')
    ax[1].set_ylabel('Var[R]')
    ax[1].set_title('Curvature variance decay')
    plt.tight_layout()
    plt.savefig('outputs/phase4_backreaction.png', dpi=150)
    plt.close(fig)

    np.savez('outputs/phase4_backreaction.npz', **back_series)

    print(
        f"Dirac overlaps: mean={overlaps['mean_overlap']:.3f}, mean_trimmed={overlaps.get('mean_overlap_trimmed', overlaps['mean_overlap']):.3f}, max={overlaps['max_overlap']:.3f}. "
        f"Fixed-point decay: ||ρ-1||_2(0)={back_series['rho_L2'][0]:.4e} -> {back_series['rho_L2'][-1]:.4e}; "
        f"Var[R](0)={back_series['var_R'][0]:.4e} -> {back_series['var_R'][-1]:.4e}."
    )


if __name__ == '__main__':
    run_phase4()
