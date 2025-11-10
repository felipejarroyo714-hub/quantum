#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh, splu

# Reuse Phase 3 geometry utilities
from kg_scale_invariant_metric import (
    GeometryParams as KGGeometryParams,
    FieldParams as KGFieldParams,
    integrate_profile,
    build_kg_operator,
    compute_modes,
    normalize_on_z,
)

# Reuse Phase 4 Dirac tools
from phase4_unification import (
    DiracGeoConfig,
    DiracFieldConfig,
    build_dirac_operator,
    dirac_modes,
    normalize_spinor_on_z,
)


@dataclass
class AdvancedBackreactionConfig:
    # time stepping
    eta_max: int = 200
    dt_init: float = 0.02
    dt_min: float = 5e-6
    dt_max: float = 0.05
    ls_shrink: float = 0.5
    dt_growth: float = 1.1
    dt_shrink: float = 0.5

    # geometry feedback
    gamma: float = 2.5     # relax (rp - alpha r)
    kappa: float = 2.0     # push rho->1
    nu: float = 0.2        # diffusion (implicit)
    kappa_local_p: float = 2.0  # localization exponent for |rho-1|^p

    # quantum coupling weights
    lambda_R: float = 10.0     # stronger curvature penalty
    lambda_Q: float = 0.1      # turn on quantum backreaction weight
    scale_Q: float = 1.0       # normalization for e_ren

    # spectra resolution
    k_scalar: int = 20
    k_dirac: int = 20


@dataclass
class QuantumFieldParams:
    # scalar
    mu: float = 0.5
    xi: float = 0.0
    m_theta: int = 0
    # fermion
    m_fermion: float = 0.5
    m_theta_f: int = 0


def compute_scalar_modes_local(z: np.ndarray, r: np.ndarray, q: QuantumFieldParams, k: int) -> Tuple[np.ndarray, np.ndarray]:
    field = KGFieldParams(mu=q.mu, xi=q.xi, m_theta=q.m_theta, k_eig=k)
    # build KG operator with current geometry
    # For stability, cap k to n-2
    A, _ = build_kg_operator(z, r, np.zeros_like(r), field)
    k_eff = min(k, A.shape[0] - 2)
    w2, modes = eigsh(A, k=k_eff, which='SA')
    order = np.argsort(w2)
    w = np.sqrt(np.clip(w2[order], 1e-12, None))
    modes = modes[:, order]
    # normalize modes on z grid
    modes = np.column_stack([normalize_on_z(z, modes[:, j]) for j in range(modes.shape[1])])
    return w, modes


def scalar_adiabatic_freq0(z: np.ndarray, r: np.ndarray, R: np.ndarray, q: QuantumFieldParams) -> np.ndarray:
    # Zeroth-order adiabatic frequency (local WKB): omega_ad(z) = sqrt(mu^2 + xi R + m_theta^2 / r^2)
    base = q.mu**2 + q.xi * R + (q.m_theta**2) / np.clip(r**2, 1e-18, None)
    return np.sqrt(np.clip(base, 1e-12, None))

def scalar_adiabatic_freq2(z: np.ndarray, r: np.ndarray, R: np.ndarray, q: QuantumFieldParams) -> np.ndarray:
    # Simple second-order correction proxy using curvature and radial gradient scales
    dz = z[1] - z[0]
    rp = np.gradient(r, dz)
    rpp = np.gradient(rp, dz)
    base0 = q.mu**2 + q.xi * R + (q.m_theta**2) / np.clip(r**2, 1e-18, None)
    corr = 0.25 * ((rpp / np.clip(r, 1e-18, None))**2 + (rp / np.clip(r, 1e-18, None))**2)
    return np.sqrt(np.clip(base0 + corr, 1e-12, None))


def fermion_adiabatic_energy0(z: np.ndarray, r: np.ndarray, q: QuantumFieldParams) -> np.ndarray:
    # Local positive energy for 2D Dirac (rest + angular): eps_ad(z) = sqrt(m_f^2 + (m_theta/r)^2)
    base = q.m_fermion**2 + (q.m_theta_f**2) / np.clip(r**2, 1e-18, None)
    return np.sqrt(np.clip(base, 1e-12, None))

def fermion_adiabatic_energy2(z: np.ndarray, r: np.ndarray, q: QuantumFieldParams) -> np.ndarray:
    # Include simple geometric correction proxy similar to scalar
    dz = z[1] - z[0]
    rp = np.gradient(r, dz)
    rpp = np.gradient(rp, dz)
    base0 = q.m_fermion**2 + (q.m_theta_f**2) / np.clip(r**2, 1e-18, None)
    corr = 0.25 * ((rpp / np.clip(r, 1e-18, None))**2 + (rp / np.clip(r, 1e-18, None))**2)
    return np.sqrt(np.clip(base0 + corr, 1e-12, None))


def compute_renormalized_energy_density(z: np.ndarray, r: np.ndarray, q: QuantumFieldParams, cfg: AdvancedBackreactionConfig) -> np.ndarray:
    # Geometry metrics
    dz = z[1] - z[0]
    rp = np.gradient(r, dz)
    rpp = np.gradient(rp, dz)
    alpha = math.log(DiracGeoConfig().lam)
    R = -2.0 * (rpp / np.clip(r, 1e-18, None))

    # Scalar contribution (bosonic + sign +)
    w, modes = compute_scalar_modes_local(z, r, q, cfg.k_scalar)
    omega_ad = scalar_adiabatic_freq2(z, r, R, q)
    # local energy density: 1/2 * sum_j (omega_j - omega_ad(z)) |u_j(z)|^2
    e_phi = np.zeros_like(r, dtype=float)
    for j in range(len(w)):
        u = modes[:, j]
        e_phi += 0.5 * (w[j] - omega_ad) * (np.abs(u) ** 2)

    # Fermionic contribution (vacuum negative sign)
    # Compute a few Dirac modes on this geometry
    from phase4_unification import DiracFieldConfig
    dcfg = DiracFieldConfig(m_fermion=q.m_fermion, m_theta=q.m_theta_f, k_eig=cfg.k_dirac)
    # reconstruct Dirac modes
    # We'll import dirac_modes from phase4_unification
    from phase4_unification import dirac_modes
    evals, spinors = dirac_modes(z, r, dcfg)
    n = len(z)
    eps_ad = fermion_adiabatic_energy2(z, r, q)
    e_psi = np.zeros_like(r, dtype=float)
    for j in range(min(len(evals), spinors.shape[1])):
        psi = spinors[:, j]
        up = psi[:n]
        dn = psi[n:]
        dens = (np.abs(up) ** 2 + np.abs(dn) ** 2).real
        e_psi += -0.5 * (np.real(evals[j]) - eps_ad) * dens

    # Total renormalized energy density (scaled)
    e_tot = cfg.scale_Q * (e_phi + e_psi)
    return e_tot


def energy_functional(z: np.ndarray, r: np.ndarray, e_ren: np.ndarray, cfg: AdvancedBackreactionConfig) -> float:
    dz = z[1] - z[0]
    alpha = math.log(DiracGeoConfig().lam)
    rp = np.gradient(r, dz)
    rpp = np.gradient(rp, dz)
    rho = (rp / np.clip(r, 1e-18, None)) / alpha
    R = -2.0 * (rpp / np.clip(r, 1e-18, None))
    term_rho = (rho - 1.0) ** 2
    term_R = cfg.lambda_R * (R + 2.0 * (alpha ** 2)) ** 2
    term_Q = cfg.lambda_Q * (e_ren ** 2)
    E = float(np.trapezoid(term_rho + term_R + term_Q, z))
    return E


def implicit_diffusion_solver(n: int, dz: float, nu: float, dt: float, r_base: np.ndarray):
    dz2 = dz * dz
    main = np.full(n, -2.0 / dz2)
    off = np.full(n - 1, 1.0 / dz2)
    L = diags([off, main, off], offsets=[-1, 0, 1], format='csr')
    M = (diags(np.ones(n)) - nu * dt * L).tocsr().tocsc()
    # Anchor boundaries to base to prevent drift
    M = M.tolil()
    M[0, :] = 0.0; M[0, 0] = 1.0
    M[-1, :] = 0.0; M[-1, -1] = 1.0
    M = M.tocsc()
    lu = splu(M)
    def solve(rhs: np.ndarray) -> np.ndarray:
        rhs2 = rhs.copy()
        rhs2[0] = r_base[0]
        rhs2[-1] = r_base[-1]
        return lu.solve(rhs2)
    return solve


def run_advanced_phase4() -> Dict[str, object]:
    # Geometry baseline
    geo = DiracGeoConfig(epsilon=0.0)
    q = QuantumFieldParams()
    cfg = AdvancedBackreactionConfig()

    kg = KGGeometryParams(lam=geo.lam, z_min=geo.z_min, z_max=geo.z_max, num_z=geo.num_z, r0=geo.r0, epsilon=0.0)
    z, r_base, _, _ = integrate_profile(kg)
    dz = z[1] - z[0]
    alpha = math.log(geo.lam)

    # initialize geometry with small perturbation
    perturb = 0.03 * np.sin(2.0 * math.pi * (z - z.min()) / (z.max() - z.min()))
    r = r_base * (1.0 + perturb)

    # implicit solver will be rebuilt when dt changes
    solve_diff = implicit_diffusion_solver(len(z), dz, cfg.nu, cfg.dt_init, r_base)

    # adaptive dt and line search
    dt = cfg.dt_init
    successes = 0

    # initial quantum energy
    e_ren = compute_renormalized_energy_density(z, r, q, cfg)
    E_prev = energy_functional(z, r, e_ren, cfg)

    rho_L2 = []
    var_R = []
    dt_series = []

    for step in range(cfg.eta_max):
        # geometry diagnostics
        rp = np.gradient(r, dz)
        rpp = np.gradient(rp, dz)
        rho = (rp / np.clip(r, 1e-18, None)) / alpha
        R = -2.0 * (rpp / np.clip(r, 1e-18, None))
        rho_L2.append(float(math.sqrt(np.trapezoid((rho - 1.0) ** 2, z) / (z.max() - z.min()))))
        var_R.append(float(np.var(R)))
        dt_series.append(dt)

        # localized feedback weight
        dev = np.abs(rho - 1.0)
        wloc = dev ** cfg.kappa_local_p
        wloc = wloc / (wloc.mean() + 1e-12)

        # compute renormalized energy density (can be throttled if needed)
        if cfg.lambda_Q > 0.0:
            e_ren = compute_renormalized_energy_density(z, r, q, cfg)
        else:
            e_ren = np.zeros_like(r)

        # explicit reactive step (no external drive)
        rhs = r + dt * (
            -cfg.gamma * (rp - alpha * r) - cfg.kappa * wloc * (rho - 1.0) * r
        )
        # implicit diffusion
        r_candidate = solve_diff(rhs)
        r_candidate = np.maximum(r_candidate, 1e-8)

        # line search on energy functional
        s = 1.0
        accepted = False
        while s > 0.05:
            r_try = r + s * (r_candidate - r)
            # always recompute energy at trial state
            if cfg.lambda_Q > 0.0:
                e_ren_try = compute_renormalized_energy_density(z, r_try, q, cfg)
            else:
                e_ren_try = np.zeros_like(r)
            E_try = energy_functional(z, r_try, e_ren_try, cfg)
            if E_try < E_prev:
                r = r_try
                E_prev = E_try
                accepted = True
                successes += 1
                if successes >= 5:
                    dt = min(cfg.dt_max, dt * cfg.dt_growth)
                    successes = 0
                break
            s *= cfg.ls_shrink
        if not accepted:
            # backtrack too aggressive; shrink dt and continue
            dt = max(cfg.dt_min, dt * cfg.dt_shrink)
            successes = 0
            # rebuild implicit solver for new dt
            solve_diff = implicit_diffusion_solver(len(z), dz, cfg.nu, dt, r_base)

    return {
        'z': z,
        'rho_L2': np.array(rho_L2),
        'var_R': np.array(var_R),
        'dt': np.array(dt_series),
        'E_final': E_prev,
    }


def report_and_plot(results: Dict[str, object]) -> None:
    os.makedirs('outputs', exist_ok=True)
    z = results['z']
    rho_L2 = results['rho_L2']
    var_R = results['var_R']
    dt = results['dt']
    E_final = results['E_final']

    eta = np.arange(len(rho_L2))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(eta, rho_L2, 'o-', ms=3)
    ax[0].set_xlabel('η (step)'); ax[0].set_ylabel('||ρ-1||_2')
    ax[0].set_title('Deviation from fixed point')

    ax[1].plot(eta, var_R, 'o-', ms=3)
    ax[1].set_xlabel('η (step)'); ax[1].set_ylabel('Var[R]')
    ax[1].set_title('Curvature variance')

    ax[2].plot(eta, dt, 'o-', ms=3)
    ax[2].set_xlabel('η (step)'); ax[2].set_ylabel('dt')
    ax[2].set_title('Adaptive dt')
    plt.tight_layout()
    plt.savefig('outputs/phase4_advanced_backreaction.png', dpi=150)
    plt.close(fig)

    with open('outputs/phase4_advanced_report.txt', 'w') as f:
        f.write(f"Final energy functional: {E_final:.6e}\n")
        f.write(f"||ρ-1||_2: start={rho_L2[0]:.6e} end={rho_L2[-1]:.6e}\n")
        f.write(f"Var[R]: start={var_R[0]:.6e} end={var_R[-1]:.6e}\n")
        f.write(f"dt: min={dt.min():.3e} max={dt.max():.3e}\n")

    print(f"Advanced backreaction: ||ρ-1||_2 {rho_L2[0]:.3e} -> {rho_L2[-1]:.3e}, Var[R] {var_R[0]:.3e} -> {var_R[-1]:.3e}, E_final={E_final:.3e}.")


if __name__ == '__main__':
    results = run_advanced_phase4()
    report_and_plot(results)
