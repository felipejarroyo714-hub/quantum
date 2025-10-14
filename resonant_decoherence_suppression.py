#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, expm_multiply


# ----------------------------
# Configuration (leveraging Phase 1 QSLS)
# ----------------------------

@dataclass
class RDSConfig:
    lambda_scale: float = math.sqrt(6.0) / 2.0
    num_shells: int = 20
    nodes_per_shell: int = 4
    base_radius: float = 1.0

    t_hop: float = 1.0
    V0: float = 5.0

    between_shell_neighbor_factor: float = 0.45
    within_shell_neighbor_factor: float = 1.05
    random_seed: int = 123
    random_rotate_each_shell: bool = True

    # Drive/noise parameters
    drive_eps: float = 0.2
    noise_sigma: float = 0.10     # OU noise diffusion scale
    noise_tau: float = 0.50       # OU correlation time
    T: float = 10.0               # total time
    dt: float = 0.05              # time step

    n0: int = 0                   # lower scale level
    n1: int = 1                   # upper scale level for target transition

    def derived(self) -> Dict[str, float]:
        return {'ln_lambda': math.log(self.lambda_scale)}


# ----------------------------
# Phase 1 geometry and Hamiltonian
# ----------------------------

def tetrahedron_vertices(radius: float) -> np.ndarray:
    verts = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    verts *= radius
    return verts


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(3,3))
    Q,_ = np.linalg.qr(M)
    Q *= np.sign(np.linalg.det(Q))
    return Q


def build_geometry(cfg: RDSConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    positions = []
    radii = []
    shells = []
    for n in range(cfg.num_shells):
        r_n = cfg.base_radius * (cfg.lambda_scale ** n)
        verts = tetrahedron_vertices(r_n)
        if cfg.random_rotate_each_shell:
            R = random_rotation_matrix(rng)
            verts = (R @ verts.T).T
        positions.append(verts)
        radii.extend([r_n] * cfg.nodes_per_shell)
        shells.extend([n] * cfg.nodes_per_shell)
    return np.vstack(positions), np.array(radii), np.array(shells)


def build_adjacency(positions: np.ndarray, radii: np.ndarray, shell_idx: np.ndarray, cfg: RDSConfig) -> csr_matrix:
    N = positions.shape[0]
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    p = cfg.nodes_per_shell
    shell_to_indices = [np.arange(s*p,(s+1)*p) for s in range(cfg.num_shells)]

    theta = math.acos(-1.0/3.0)
    chord = 2.0 * math.sin(theta/2.0)

    # within-shell
    for s in range(cfg.num_shells):
        idx = shell_to_indices[s]
        cutoff = cfg.within_shell_neighbor_factor * chord * radii[idx[0]]
        for i_local in range(p):
            for j_local in range(i_local+1, p):
                i = int(idx[i_local]); j = int(idx[j_local])
                if np.linalg.norm(positions[i] - positions[j]) <= cutoff:
                    rows.extend([i, j]); cols.extend([j, i]); data.extend([1.0, 1.0])

    # between adjacent shells: nearest with connectivity fallback
    factor = cfg.between_shell_neighbor_factor
    for s in range(cfg.num_shells - 1):
        idx_s = shell_to_indices[s]
        idx_t = shell_to_indices[s + 1]
        r_s = radii[idx_s[0]]; r_t = radii[idx_t[0]]
        cutoff = factor * math.sqrt(r_s * r_t)
        has_f = {int(i): False for i in idx_s}
        has_b = {int(j): False for j in idx_t}
        for i in idx_s:
            d = np.linalg.norm(positions[idx_t] - positions[int(i)], axis=1)
            j = int(idx_t[int(np.argmin(d))])
            if d.min() <= cutoff:
                rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([1.0, 1.0])
                has_f[int(i)] = True; has_b[j] = True
        for j in idx_t:
            d = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
            i = int(idx_s[int(np.argmin(d))])
            if d.min() <= cutoff:
                rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([1.0, 1.0])
                has_f[i] = True; has_b[int(j)] = True
        for i in idx_s:
            if not has_f[int(i)]:
                d = np.linalg.norm(positions[idx_t] - positions[int(i)], axis=1)
                j = int(idx_t[int(np.argmin(d))])
                rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([1.0, 1.0])
        for j in idx_t:
            if not has_b[int(j)]:
                d = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
                i = int(idx_s[int(np.argmin(d))])
                rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([1.0, 1.0])

    A = coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(N, N)).tocsr()
    A.sum_duplicates(); A.data[:] = 1.0
    return A


def build_between_shell_hopping(A: csr_matrix, cfg: RDSConfig) -> csr_matrix:
    p = cfg.nodes_per_shell
    rows, cols = A.nonzero()
    data = []
    rr = []
    cc = []
    for i, j in zip(rows, cols):
        si = i // p
        sj = j // p
        if abs(si - sj) == 1:
            rr.append(i); cc.append(j); data.append(1.0)
    if not data:
        return A.copy() * 0.0
    M = coo_matrix((np.array(data), (np.array(rr), np.array(cc))), shape=A.shape)
    M = (M + M.T).tocsr()
    M.sum_duplicates(); M.data[:] = 1.0
    return M


def build_hamiltonian(A: csr_matrix, radii: np.ndarray, cfg: RDSConfig) -> csr_matrix:
    lnlam = math.log(cfg.lambda_scale)
    V = cfg.V0 * (np.log(radii) / lnlam) ** 2
    return (-cfg.t_hop) * A + diags(V, format='csr')


# ----------------------------
# Time evolution with drive and noise
# ----------------------------

def ou_step(x: float, dt: float, tau: float, sigma: float, rng: np.random.Generator) -> float:
    # Ornstein-Uhlenbeck step: dx = -(x/tau) dt + sigma sqrt(dt) N(0,1)
    return x + (-x / max(tau, 1e-9)) * dt + sigma * math.sqrt(max(dt, 0.0)) * rng.normal()


def time_evolve(H0: csr_matrix, Hc: csr_matrix, Hd: csr_matrix,
                 psi0: np.ndarray, omega: float,
                 drive_eps: float, sigma: float, tau: float,
                 T: float, dt: float, seed: int) -> Tuple[np.ndarray, List[float]]:
    psi = psi0.copy()
    rng = np.random.default_rng(seed)
    noise_amp = 0.0
    t = 0.0
    times = []
    while t < T - 1e-12:
        noise_amp = ou_step(noise_amp, dt, tau, sigma, rng)
        Ht = H0
        if abs(drive_eps) > 0:
            Ht = Ht + (drive_eps * math.cos(omega * t)) * Hc
        if sigma > 0:
            Ht = Ht + (noise_amp) * Hd
        psi = expm_multiply((-1j * dt) * Ht, psi)
        t += dt
        times.append(t)
    return psi, times


# ----------------------------
# Utilities
# ----------------------------

def project_shell_population(psi: np.ndarray, n: int, cfg: RDSConfig) -> float:
    p = cfg.nodes_per_shell
    idx = np.arange(n * p, min((n + 1) * p, psi.size))
    prob = np.abs(psi) ** 2
    return float(prob[idx].sum())


def fidelity(psi_ref: np.ndarray, psi: np.ndarray) -> float:
    a = np.vdot(psi_ref, psi)
    return float(np.abs(a) ** 2)


# ----------------------------
# Main experiment
# ----------------------------

def main() -> None:
    cfg = RDSConfig()
    d = cfg.derived()

    # Build system
    pos, radii, shells = build_geometry(cfg)
    A = build_adjacency(pos, radii, shells, cfg)
    H0 = build_hamiltonian(A, radii, cfg)
    Hc = build_between_shell_hopping(A, cfg)

    # Diagonal decoherence operator (random on-site)
    rng = np.random.default_rng(cfg.random_seed)
    diag_noise = rng.normal(size=H0.shape[0])
    diag_noise /= (np.max(np.abs(diag_noise)) + 1e-18)
    Hd = diags(diag_noise, format='csr')

    # Spectrum and localized representatives
    N = H0.shape[0]
    k = min(100, N - 2)
    evals, evecs = eigsh(H0, k=k, which='SA')
    ord = np.argsort(evals)
    evals = evals[ord]
    evecs = evecs[:, ord]

    # Scaled index x for localization
    x_vals = np.empty(k)
    for i in range(k):
        prob = np.abs(evecs[:, i]) ** 2
        mean_r = float(np.dot(prob, radii))
        x_vals[i] = math.log(mean_r) / d['ln_lambda']

    targets: Dict[int, int] = {}
    for i in range(k):
        n_est = int(round(x_vals[i]))
        if 0 <= n_est < cfg.num_shells and n_est not in targets:
            targets[n_est] = i

    if cfg.n0 not in targets or cfg.n1 not in targets:
        raise RuntimeError('Could not find representative states for requested shells.')

    i0 = targets[cfg.n0]
    i1 = targets[cfg.n1]

    omega_res = float(evals[i1] - evals[i0])

    # Initial state: balanced superposition of |n0> and |n1>
    psi0 = (evecs[:, i0] + evecs[:, i1])
    psi0 = psi0 / (np.linalg.norm(psi0) + 1e-18)

    # Ideal reference (no drive, no noise)
    psi_ref, _ = time_evolve(H0, Hc, Hd, psi0, omega=0.0, drive_eps=0.0, sigma=0.0, tau=cfg.noise_tau, T=cfg.T, dt=cfg.dt, seed=cfg.random_seed)

    scenarios = {
        'noise_only': dict(drive_eps=0.0, omega=omega_res, sigma=cfg.noise_sigma),
        'drive_only_resonant': dict(drive_eps=cfg.drive_eps, omega=omega_res, sigma=0.0),
        'drive_plus_noise_resonant': dict(drive_eps=cfg.drive_eps, omega=omega_res, sigma=cfg.noise_sigma),
        'drive_plus_noise_offres': dict(drive_eps=cfg.drive_eps, omega=0.5 * omega_res, sigma=cfg.noise_sigma),
    }

    results = {}
    for name, pars in scenarios.items():
        psi_T, _ = time_evolve(H0, Hc, Hd, psi0,
                               omega=pars['omega'], drive_eps=pars['drive_eps'],
                               sigma=pars['sigma'], tau=cfg.noise_tau,
                               T=cfg.T, dt=cfg.dt, seed=cfg.random_seed + 7)
        F = fidelity(psi_ref, psi_T)
        Pn0 = project_shell_population(psi_T, cfg.n0, cfg)
        Pn1 = project_shell_population(psi_T, cfg.n1, cfg)
        results[name] = dict(fidelity=F, Pn0=Pn0, Pn1=Pn1)

    os.makedirs('outputs', exist_ok=True)
    import json
    with open('outputs/rds_results.json', 'w') as f:
        json.dump({
            'omega_res': omega_res,
            'evals_pair': (float(evals[i0]), float(evals[i1])),
            'scenarios': results,
            'config': cfg.__dict__,
        }, f, indent=2)

    # Plot populations across time for resonant vs off-resonant (optional short trace)
    # Shorter trace to keep runtime modest
    def trace(name: str, omega: float):
        psi = psi0.copy()
        rng_trace = np.random.default_rng(cfg.random_seed + 11)
        noise_amp = 0.0
        t = 0.0
        times = []
        pn0 = []
        pn1 = []
        while t < cfg.T - 1e-12:
            noise_amp = ou_step(noise_amp, cfg.dt, cfg.noise_tau, cfg.noise_sigma, rng_trace)
            Ht = H0 + (cfg.drive_eps * math.cos(omega * t)) * Hc + (noise_amp) * Hd
            psi = expm_multiply((-1j * cfg.dt) * Ht, psi)
            t += cfg.dt
            times.append(t)
            pn0.append(project_shell_population(psi, cfg.n0, cfg))
            pn1.append(project_shell_population(psi, cfg.n1, cfg))
        return np.array(times), np.array(pn0), np.array(pn1)

    t1, pn0_res, pn1_res = trace('res', omega_res)
    t2, pn0_off, pn1_off = trace('off', 0.5 * omega_res)

    plt.figure(figsize=(10,4))
    plt.plot(t1, pn1_res, label='P(n=1) resonant')
    plt.plot(t2, pn1_off, '--', label='P(n=1) off-resonant')
    plt.xlabel('t'); plt.ylabel('Population in shell n=1')
    plt.title('Resonant transfer vs off-resonant under drive+noise')
    plt.legend(); plt.tight_layout()
    plt.savefig('outputs/rds_population_trace.png', dpi=150)
    plt.close()

    # Console summary
    print('Resonant Decoherence Suppression (RDS) complete.')
    print(f"Levels: n0={cfg.n0}, n1={cfg.n1}; E0={evals[i0]:.6f}, E1={evals[i1]:.6f}; omega_res={omega_res:.6f}")
    for name, r in results.items():
        print(f"{name}: fidelity={r['fidelity']:.6f}, Pn0={r['Pn0']:.6f}, Pn1={r['Pn1']:.6f}")


if __name__ == '__main__':
    main()
