#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity, kron, csr_matrix
from scipy.sparse.linalg import eigsh


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class DiracGeoConfig:
    lam: float = math.sqrt(6.0) / 2.0
    z_min: float = -10.0
    z_max: float = 10.0
    num_z: int = 1200
    r0: float = 1.0
    epsilon: float = 0.0  # index density fluctuation amplitude

    def derived(self) -> Dict[str, float]:
        alpha = math.log(self.lam)
        return dict(alpha=alpha)


@dataclass
class DiracFieldConfig:
    m_theta: int = 0     # angular momentum on S^1 fiber
    m_fermion: float = 0.5  # fermion mass
    k_eig: int = 40      # number of eigenmodes to compute
    bcap: float = 1e6    # boundary penalty (Dirichlet-ish)


# ----------------------------
# Geometry: r(z), ρ(z), curvature R(z)
# ----------------------------

def make_h(lam: float, epsilon: float = 0.0):
    alpha = math.log(lam)
    def h(r: float) -> float:
        if r <= 0:
            return 0.0
        if epsilon == 0.0:
            return alpha * r
        x = math.log(r) / alpha
        return alpha * r * (1.0 + epsilon * math.cos(2.0 * math.pi * x))
    return h

def integrate_profile(cfg: DiracGeoConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = cfg.derived()['alpha']
    z = np.linspace(cfg.z_min, cfg.z_max, cfg.num_z)
    dz = z[1] - z[0]

    h = make_h(cfg.lam, cfg.epsilon)
    r = np.empty_like(z)
    r[0] = cfg.r0
    for i in range(1, cfg.num_z):
        r[i] = max(1e-18, r[i-1] + dz * h(r[i-1]))

    rho = np.array([h(rv) / (alpha * rv if rv > 0 else alpha) for rv in r])

    def h_prime(rv: float) -> float:
        if cfg.epsilon == 0.0:
            return alpha
        if rv <= 0:
            return alpha
        x = math.log(rv) / alpha
        return alpha * (1.0 + cfg.epsilon * math.cos(2.0 * math.pi * x)
                        - cfg.epsilon * (2.0 * math.pi) * math.sin(2.0 * math.pi * x) / alpha)

    rp = np.array([h(rv) for rv in r])
    rpp = np.array([h_prime(rv) for rv in r]) * rp
    with np.errstate(divide='ignore', invalid='ignore'):
        K = -rpp / np.clip(r, 1e-18, None)
        R = 2.0 * K

    return z, r, rho, R


# ----------------------------
# Curved-space Dirac operator after θ separation
# H = -i σ_x (∂_z - (r'/(4r))) + σ_y (mθ / r) + m_f σ_z
# ----------------------------

def build_dirac_operator(z: np.ndarray, r: np.ndarray, field: DiracFieldConfig, bcap: float) -> csr_matrix:
    n = len(z)
    dz = z[1] - z[0]

    # central difference derivative matrix with zeros at boundaries
    off_p = np.full(n-1, -1.0/(2.0*dz))
    off_m = np.full(n-1,  1.0/(2.0*dz))
    D = diags([off_m, np.zeros(n), off_p], offsets=[-1, 0, 1], dtype=complex, format='csr')

    # spin connection term: A(z) = r'/(4 r) with r'≈dr/dz (numerical)
    rp = np.gradient(r, dz)
    A = (rp / np.clip(r, 1e-18, None)) * 0.25

    # angular term B(z) = mθ / r
    B = field.m_theta / np.clip(r, 1e-18, None)

    I_n = identity(n, dtype=complex, format='csr')
    diagA = diags(A, 0, dtype=complex, format='csr')
    diagB = diags(B, 0, dtype=complex, format='csr')

    # Pauli matrices
    sx = csr_matrix(np.array([[0, 1],[1, 0]], dtype=complex))
    sy = csr_matrix(np.array([[0, -1j],[1j, 0]], dtype=complex))
    sz = csr_matrix(np.array([[1, 0],[0, -1]], dtype=complex))
    s0 = csr_matrix(np.eye(2, dtype=complex))

    # Boundary penalty to simulate Dirichlet: add large diagonal at edges
    wall = np.zeros(n, dtype=complex)
    wall[0] = bcap; wall[-1] = bcap
    W = diags(wall, 0, dtype=complex, format='csr')

    Hz = kron(sx, (-1j)*D + (1j)*diagA)
    Hy = kron(sy, diagB)
    Hm = kron(sz, field.m_fermion * I_n)
    Hb = kron(s0, W)

    H = Hz + Hy + Hm + Hb
    return H


def compute_dirac_modes(H: csr_matrix, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n2 = H.shape[0]
    k = min(k, n2 - 2)
    evals, evecs = eigsh(H, k=k, which='SA')  # Hermitian
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def normalize_spinors(z: np.ndarray, psi: np.ndarray, weight: np.ndarray = None) -> np.ndarray:
    # psi: (2n, m), stack spin components: [u(z), v(z)]
    n2, m = psi.shape
    n = n2 // 2
    u = psi[:n, :]
    v = psi[n:, :]
    if weight is None:
        weight = np.ones_like(z)
    W = weight[:, None]
    norms = np.sqrt(np.trapezoid((np.abs(u)**2 + np.abs(v)**2) * W, z, axis=0)) + 1e-18
    return psi / norms


def shift_and_overlap(z: np.ndarray, psiA: np.ndarray, zB: np.ndarray, psiB: np.ndarray, delta: float, weight: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    # Interpolate psiB at z+delta and compute overlaps <A|B_shift>
    n2, mA = psiA.shape
    n = len(z)
    if weight is None:
        weight = np.ones_like(z)
    # components
    uA = psiA[:n, :]
    vA = psiA[n:, :]
    uB = psiB[:n, :]
    vB = psiB[n:, :]
    # complex interpolation: interp real and imag separately
    def cintp(y, x_new):
        return np.interp(x_new, zB, y.real, left=0.0, right=0.0) + 1j*np.interp(x_new, zB, y.imag, left=0.0, right=0.0)
    z_shift = z + delta
    overlaps = []
    max_per_col = []
    for j in range(mA):
        uAs = uA[:, j]; vAs = vA[:, j]
        # project onto each B mode
        proj_vals = []
        for k in range(psiB.shape[1]):
            uBs = cintp(uB[:, k], z_shift)
            vBs = cintp(vB[:, k], z_shift)
            integrand = (np.conj(uAs)*uBs + np.conj(vAs)*vBs) * weight
            proj = np.trapezoid(integrand, z)
            proj_vals.append(proj)
        proj_vals = np.array(proj_vals)
        overlaps.append(np.abs(proj_vals))
        max_per_col.append(np.max(np.abs(proj_vals)))
    return np.array(overlaps), np.array(max_per_col)


# ----------------------------
# Main
# ----------------------------

def main():
    geoA = DiracGeoConfig(epsilon=0.0)
    geoB = DiracGeoConfig(epsilon=0.0, r0=1.0*DiracGeoConfig().lam)  # λ-rescaled
    field = DiracFieldConfig(m_theta=0, m_fermion=0.5, k_eig=40)

    # Backgrounds
    zA, rA, rhoA, RA = integrate_profile(geoA)
    zB, rB, rhoB, RB = integrate_profile(geoB)

    # Operators
    HA = build_dirac_operator(zA, rA, field, bcap=field.bcap)
    HB = build_dirac_operator(zB, rB, field, bcap=field.bcap)

    # Modes
    EA, PSA = compute_dirac_modes(HA, k=field.k_eig)
    EB, PSB = compute_dirac_modes(HB, k=field.k_eig)

    # Normalize (plain dz measure)
    PSA = normalize_spinors(zA, PSA)
    PSB = normalize_spinors(zB, PSB)

    # λ-covariance: compare A vs B shifted by Δz=1
    delta = 1.0
    absO, max_per = shift_and_overlap(zA, PSA, zB, PSB, delta=delta)
    mean_max_overlap = float(np.mean(max_per))
    max_overlap = float(np.max(max_per))

    # Basic numerics
    stats = {
        'EA_minmax': (float(EA.min()), float(EA.max())),
        'EB_minmax': (float(EB.min()), float(EB.max())),
        'mean_max_overlap': mean_max_overlap,
        'max_overlap': max_overlap,
        'rhoA_minmax': (float(rhoA.min()), float(rhoA.max())),
        'rhoB_minmax': (float(rhoB.min()), float(rhoB.max())),
        'R_A': (float(RA.min()), float(RA.max())),
        'R_B': (float(RB.min()), float(RB.max())),
    }

    os.makedirs('outputs', exist_ok=True)
    import json
    with open('outputs/dirac_unification_results.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Quick plot: first few spinor components for A
    n = len(zA)
    mshow = min(3, PSA.shape[1])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for j in range(mshow):
        u = PSA[:n, j]; v = PSA[n:, j]
        ax.plot(zA, u.real, label=f'u{j} Re, E={EA[j]:.3f}')
    ax.set_xlabel('z'); ax.set_ylabel('spinor component')
    ax.set_title('Dirac spinor components (subset)')
    ax.legend(); plt.tight_layout()
    plt.savefig('outputs/dirac_unification_modes.png', dpi=150)
    plt.close(fig)

    print('Dirac Unification run complete.')
    print('Stats:', stats)
    print('Outputs: outputs/dirac_unification_results.json, outputs/dirac_unification_modes.png')


if __name__ == '__main__':
    main()
