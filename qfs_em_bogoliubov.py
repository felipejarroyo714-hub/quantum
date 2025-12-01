#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class GeoConfig:
    lam: float = math.sqrt(6.0) / 2.0  # λ (Tetrahedral Kernel)
    z_min: float = -10.0
    z_max: float = 10.0
    num_z: int = 1200
    r0: float = 1.0
    epsilon: float = 0.0  # scale-density fluctuation amplitude (ρ deviation)

    def derived(self) -> Dict[str, float]:
        alpha = math.log(self.lam)
        return dict(alpha=alpha)


@dataclass
class FieldConfig:
    mu: float = 0.5      # scalar mass
    xi: float = 0.0      # curvature coupling
    m_theta: int = 0     # angular momentum on S^1 fiber
    k_eig: int = 40      # number of modes to compute


# ----------------------------
# Geometry: profile, index density, curvature
# ----------------------------

def make_h_function(lam: float, epsilon: float = 0.0) -> Callable[[float], float]:
    alpha = math.log(lam)

    def h(r: float) -> float:
        if r <= 0.0:
            return 0.0
        if epsilon == 0.0:
            return alpha * r
        # periodic modulation in x = ln r / ln λ
        x = math.log(r) / alpha
        return alpha * r * (1.0 + epsilon * math.cos(2.0 * math.pi * x))

    return h


def integrate_profile(geo: GeoConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = geo.derived()
    alpha = d['alpha']
    z = np.linspace(geo.z_min, geo.z_max, geo.num_z)
    dz = z[1] - z[0]

    h = make_h_function(geo.lam, geo.epsilon)

    r = np.empty_like(z)
    r[0] = geo.r0
    # explicit Euler; small dz keeps stable and monotone since h ~ O(r)
    for i in range(1, geo.num_z):
        r[i] = max(1e-18, r[i - 1] + dz * h(r[i - 1]))

    # index density ρ(z) = h(r) / (ln λ ⋅ r)
    rho = np.array([h(rv) / (alpha * rv if rv > 0 else alpha) for rv in r])

    # curvature: K = - r" / r ; R = 2 K in 2D axisymmetric surface
    def h_prime(rv: float) -> float:
        if geo.epsilon == 0.0:
            return alpha
        if rv <= 0.0:
            return alpha
        x = math.log(rv) / alpha
        # h(r) = alpha * r * (1 + eps cos(2π x)), x = ln r / alpha
        # dh/dr = alpha * [ (1 + eps cos(2π x)) + r * ( - eps * 2π sin(2π x) ) * dx/dr ]
        # dx/dr = 1 / (alpha r)
        return alpha * (1.0 + geo.epsilon * math.cos(2.0 * math.pi * x)
                         - geo.epsilon * (2.0 * math.pi) * math.sin(2.0 * math.pi * x) / alpha)

    rp = np.array([h(rv) for rv in r])
    rpp = np.array([h_prime(rv) for rv in r]) * rp
    with np.errstate(divide='ignore', invalid='ignore'):
        K = -rpp / np.clip(r, 1e-18, None)
        R = 2.0 * K

    return z, r, rho, R


# ----------------------------
# KG operator on ds^2 = dz^2 + r(z)^2 dθ^2
# ----------------------------

def build_kg_operator(z: np.ndarray, r: np.ndarray, R: np.ndarray, field: FieldConfig):
    n = len(z)
    dz = z[1] - z[0]

    # symmetric discretization of - (1/r) d/dz [ r du/dz ]
    r_mid = 0.5 * (r[1:] + r[:-1])
    main = np.zeros(n)
    off = np.zeros(n - 1)

    for i in range(1, n - 1):
        a_plus = r_mid[i] / (r[i] * dz * dz)
        a_minus = r_mid[i - 1] / (r[i] * dz * dz)
        main[i] = a_plus + a_minus
        off[i - 1] = -a_minus
    # Dirichlet at ends
    main[0] = 1.0
    main[-1] = 1.0

    off_upper = np.zeros(n - 1)
    for i in range(1, n - 1):
        a_plus = r_mid[i] / (r[i] * dz * dz)
        off_upper[i] = -a_plus

    # angular, mass, curvature
    ang = (field.m_theta ** 2) / np.clip(r ** 2, 1e-18, None)
    pot = ang + (field.mu ** 2 + field.xi * R)
    main += pot

    A = diags([off, main, off_upper], offsets=[-1, 0, 1], format='csr')
    return A, pot


def compute_modes(A, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    k = min(k, n - 2)
    evals, evecs = eigsh(A, k=k, which='SA')
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def normalize_on_z(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    # L2 normalization on z with plain dz measure (consistent with the discretization used)
    norm = math.sqrt(np.trapezoid(u * u, z))
    return u / (norm + 1e-18)


# ----------------------------
# Covariance and Bogoliubov-like overlaps
# ----------------------------

def lambda_covariance_metrics(geo: GeoConfig, field: FieldConfig) -> Dict[str, float]:
    # Background A (r0), B (r0 * λ) -> expect z-shift by +1 for ε=0
    zA, rA, rhoA, RA = integrate_profile(geo)
    A, _ = build_kg_operator(zA, rA, RA, field)
    w2A, vA = compute_modes(A, k=min(20, field.k_eig))

    geoB = GeoConfig(lam=geo.lam, z_min=geo.z_min, z_max=geo.z_max, num_z=geo.num_z, r0=geo.r0 * geo.lam, epsilon=geo.epsilon)
    zB, rB, rhoB, RB = integrate_profile(geoB)
    B, _ = build_kg_operator(zB, rB, RB, field)
    w2B, vB = compute_modes(B, k=min(20, field.k_eig))

    delta = 1.0  # shift by one λ-period in z because r(z+1)=λ r(z) for ε=0
    overlaps = []
    for j in range(min(vA.shape[1], vB.shape[1])):
        uA = normalize_on_z(zA, vA[:, j])
        uB_shift = np.interp(zA + delta, zB, normalize_on_z(zB, vB[:, j]), left=0.0, right=0.0)
        overlaps.append(abs(np.trapezoid(uA * uB_shift, zA)))

    return {
        'w2A_min': float(w2A[0]), 'w2B_min': float(w2B[0]),
        'mean_overlap': float(np.mean(overlaps)),
        'max_overlap': float(np.max(overlaps)),
        'num_compared': int(len(overlaps)),
    }


def bogoliubov_overlap_metrics(geo0: GeoConfig, geoE: GeoConfig, field: FieldConfig) -> Dict[str, float]:
    # Baseline (ε=0) vs fluctuating (ε>0) mode overlap matrix
    z0, r0, rho0, R0 = integrate_profile(geo0)
    A0, _ = build_kg_operator(z0, r0, R0, field)
    w2_0, V0 = compute_modes(A0, k=field.k_eig)
    U0 = np.stack([normalize_on_z(z0, V0[:, j]) for j in range(V0.shape[1])], axis=1)

    zE, rE, rhoE, RE = integrate_profile(geoE)
    AE, _ = build_kg_operator(zE, rE, RE, field)
    w2_E, VE = compute_modes(AE, k=field.k_eig)
    UE = np.stack([normalize_on_z(zE, VE[:, j]) for j in range(VE.shape[1])], axis=1)

    assert np.allclose(z0, zE)
    z = z0

    # Overlap matrix O_{jk} = ∫ U0_j(z) UE_k(z) dz
    O = np.trapezoid(U0[:, :, None] * UE[:, None, :], z, axis=0)
    absO = np.abs(O)

    # Metrics
    max_per_row = absO.max(axis=1)
    argmax_per_row = absO.argmax(axis=1)
    diag_power = float(np.sum(max_per_row ** 2))
    total_power = float(np.sum(absO ** 2)) + 1e-18
    leakage_fraction = max(0.0, 1.0 - diag_power / total_power)

    nearest_pairs = []
    for j in range(len(w2_0)):
        k = int(np.argmin(np.abs(w2_E - w2_0[j])))
        nearest_pairs.append(absO[j, k])
    mean_nearest = float(np.mean(nearest_pairs))

    # Spectral instability counts
    neg0 = int((w2_0 < 0).sum())
    negE = int((w2_E < 0).sum())

    return {
        'rho0_minmax': (float(rho0.min()), float(rho0.max())),
        'rhoE_minmax': (float(rhoE.min()), float(rhoE.max())),
        'R0_minmax': (float(R0.min()), float(R0.max())),
        'RE_minmax': (float(RE.min()), float(RE.max())),
        'minmax_w2_baseline': (float(w2_0.min()), float(w2_0.max())),
        'minmax_w2_fluct': (float(w2_E.min()), float(w2_E.max())),
        'neg_w2_count_baseline': neg0,
        'neg_w2_count_fluct': negE,
        'mean_max_overlap': float(max_per_row.mean()),
        'mean_nearest_freq_overlap': mean_nearest,
        'offdiag_leakage_fraction': leakage_fraction,
    }


# ----------------------------
# Plots
# ----------------------------

def plot_background(z: np.ndarray, rho: np.ndarray, R: np.ndarray, title: str, out_png: str) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(z, rho)
    ax[0].set_ylabel('ρ(z)')
    ax[0].set_title(title)
    ax[1].plot(z, R)
    ax[1].set_ylabel('R(z)')
    ax[1].set_xlabel('z')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_modes(z: np.ndarray, modes: np.ndarray, w2: np.ndarray, n_show: int, title: str, out_png: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    n = min(n_show, modes.shape[1])
    for j in range(n):
        ax.plot(z, modes[:, j] / (np.max(np.abs(modes[:, j])) + 1e-18), label=f'j={j}, ω^2={w2[j]:.3f}')
    ax.set_xlabel('z')
    ax.set_ylabel('mode amplitude (normed)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_overlap_matrix(absO: np.ndarray, out_png: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(absO, interpolation='nearest', aspect='auto', origin='lower', cmap='magma')
    ax.set_xlabel('k (fluctuating)')
    ax.set_ylabel('j (baseline)')
    ax.set_title('|Overlap| between baseline and fluctuating modes')
    plt.colorbar(im, ax=ax, label='|O_{jk}|')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    geo0 = GeoConfig(epsilon=0.0)
    field = FieldConfig(mu=0.5, xi=0.0, m_theta=0, k_eig=40)

    # Baseline background and modes
    z0, r0, rho0, R0 = integrate_profile(geo0)
    A0, _ = build_kg_operator(z0, r0, R0, field)
    w2_0, V0 = compute_modes(A0, k=field.k_eig)
    U0 = np.stack([normalize_on_z(z0, V0[:, j]) for j in range(V0.shape[1])], axis=1)

    plot_background(z0, rho0, R0, 'Baseline λ-invariant background (ε=0)', 'outputs/qfsem_background_baseline.png')
    plot_modes(z0, U0, w2_0, n_show=5, title='Baseline lowest modes (ε=0)', out_png='outputs/qfsem_modes_baseline.png')

    # Covariance under λ-rescaling
    cov = lambda_covariance_metrics(geo0, field)

    # Fluctuating geometry and Bogoliubov-like overlaps
    geoE = GeoConfig(epsilon=0.05)
    zE, rE, rhoE, RE = integrate_profile(geoE)
    AE, _ = build_kg_operator(zE, rE, RE, field)
    w2_E, VE = compute_modes(AE, k=field.k_eig)
    UE = np.stack([normalize_on_z(zE, VE[:, j]) for j in range(VE.shape[1])], axis=1)

    plot_background(zE, rhoE, RE, 'Fluctuating background (ε=0.05)', 'outputs/qfsem_background_fluct.png')
    plot_modes(zE, UE, w2_E, n_show=5, title='Fluctuating lowest modes (ε=0.05)', out_png='outputs/qfsem_modes_fluct.png')

    # Overlap matrix and metrics
    O = np.trapezoid(U0[:, :, None] * UE[:, None, :], z0, axis=0)
    absO = np.abs(O)
    plot_overlap_matrix(absO, 'outputs/qfsem_overlap_matrix.png')

    bog = bogoliubov_overlap_metrics(geo0, geoE, field)

    # Save metrics
    out = {
        'lambda': geo0.lam,
        'alpha': math.log(geo0.lam),
        'z_min': float(geo0.z_min),
        'z_max': float(geo0.z_max),
        'num_z': int(geo0.num_z),
        'covariance': cov,
        'bogoliubov': bog,
    }
    os.makedirs('outputs', exist_ok=True)
    import json
    with open('outputs/qfsem_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    print('QFS-EM Phase 3 run complete.')
    print('Covariance:', cov)
    print('Bogoliubov metrics:', bog)
    print('Plots: outputs/qfsem_background_baseline.png, outputs/qfsem_background_fluct.png,')
    print('       outputs/qfsem_modes_baseline.png, outputs/qfsem_modes_fluct.png, outputs/qfsem_overlap_matrix.png')
    print('JSON:  outputs/qfsem_results.json')


if __name__ == '__main__':
    main()
