#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, expm_multiply
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

    # drive + noise
    drive_eps: float = 0.2
    noise_eps: float = 0.1
    noise_tau: float = 0.2   # OU correlation time

    T: float = 20.0
    dt: float = 0.05

    def derived(self) -> Dict[str, float]:
        return {'ln_lambda': math.log(self.lambda_scale)}


def tetrahedron_vertices(radius: float) -> np.ndarray:
    v = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]], dtype=float)
    v /= np.linalg.norm(v[0])
    return v * radius


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(3,3))
    Q,_ = np.linalg.qr(M)
    Q *= np.sign(np.linalg.det(Q))
    return Q


def build_geometry(cfg: RDSConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    pos = []; radii = []; shells = []
    for n in range(cfg.num_shells):
        r_n = cfg.base_radius * (cfg.lambda_scale ** n)
        verts = tetrahedron_vertices(r_n)
        if cfg.random_rotate_each_shell:
            R = random_rotation_matrix(rng)
            verts = (R @ verts.T).T
        pos.append(verts)
        radii.extend([r_n]*cfg.nodes_per_shell)
        shells.extend([n]*cfg.nodes_per_shell)
    return np.vstack(pos), np.array(radii), np.array(shells)


def build_adjacency(positions: np.ndarray, radii: np.ndarray, shell_idx: np.ndarray, cfg: RDSConfig) -> csr_matrix:
    N = positions.shape[0]
    rows = []; cols = []; data = []
    p = cfg.nodes_per_shell
    shell_to_indices = [np.arange(s*p,(s+1)*p) for s in range(cfg.num_shells)]

    theta = math.acos(-1.0/3.0)
    chord = 2.0 * math.sin(theta/2.0)

    # within-shell full
    for s in range(cfg.num_shells):
        idx = shell_to_indices[s]
        cutoff = cfg.within_shell_neighbor_factor * chord * radii[idx[0]]
        for i_local in range(p):
            for j_local in range(i_local+1,p):
                i = int(idx[i_local]); j = int(idx[j_local])
                if np.linalg.norm(positions[i]-positions[j]) <= cutoff:
                    rows.extend([i,j]); cols.extend([j,i]); data.extend([1.0,1.0])

    # between adjacent shells with fallback
    factor = cfg.between_shell_neighbor_factor
    for s in range(cfg.num_shells-1):
        idx_s = shell_to_indices[s]
        idx_t = shell_to_indices[s+1]
        r_s = radii[idx_s[0]]; r_t = radii[idx_t[0]]
        cutoff = factor * math.sqrt(r_s*r_t)
        has_f = {int(i): False for i in idx_s}
        has_b = {int(j): False for j in idx_t}
        for i in idx_s:
            d = np.linalg.norm(positions[idx_t] - positions[int(i)], axis=1)
            j = int(idx_t[int(np.argmin(d))])
            if d.min() <= cutoff:
                rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([1.0,1.0])
                has_f[int(i)] = True; has_b[j] = True
        for j in idx_t:
            d = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
            i = int(idx_s[int(np.argmin(d))])
            if d.min() <= cutoff:
                rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([1.0,1.0])
                has_f[i] = True; has_b[int(j)] = True
        for i in idx_s:
            if not has_f[int(i)]:
                d = np.linalg.norm(positions[idx_t] - positions[int(i)], axis=1)
                j = int(idx_t[int(np.argmin(d))])
                rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([1.0,1.0])
        for j in idx_t:
            if not has_b[int(j)]:
                d = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
                i = int(idx_s[int(np.argmin(d))])
                rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([1.0,1.0])

    A = coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(N, N)).tocsr()
    A.sum_duplicates(); A.data[:] = 1.0
    return A


def build_hamiltonian(A: csr_matrix, radii: np.ndarray, cfg: RDSConfig) -> csr_matrix:
    lnlam = math.log(cfg.lambda_scale)
    V = cfg.V0 * (np.log(radii)/lnlam)**2
    return (-cfg.t_hop) * A + diags(V, format='csr')


def build_between_shell_hopping(A: csr_matrix, cfg: RDSConfig) -> csr_matrix:
    p = cfg.nodes_per_shell
    rows, cols = A.nonzero()
    keep = []
    for i, j in zip(rows, cols):
        if abs(i//p - j//p) == 1:
            keep.append((i, j))
    if not keep:
        return A*0
    rr = np.array([i for i,j in keep], dtype=int)
    cc = np.array([j for i,j in keep], dtype=int)
    M = coo_matrix((np.ones(len(keep)), (rr, cc)), shape=A.shape).tocsr()
    M = (M + M.T)
    M.sum_duplicates(); M.data[:] = 1.0
    return M


def ornstein_uhlenbeck(dt: float, T: float, tau: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    steps = int(np.ceil(T/dt))
    x = np.zeros(steps)
    a = math.exp(-dt/tau)
    s = sigma * math.sqrt(1 - a*a)
    for k in range(1, steps):
        x[k] = a*x[k-1] + s*rng.normal()
    return x


def time_evolve(H0: csr_matrix, Hcoup: csr_matrix, Hdecoh: csr_matrix, psi0: np.ndarray, w_drive: float, eps_drive: float,
                noise_t: np.ndarray, eps_noise: float, dt: float) -> np.ndarray:
    psi = psi0.copy()
    t = 0.0
    steps = len(noise_t)
    for k in range(steps):
        Hd = eps_drive * math.cos(w_drive * t) * Hcoup
        Hn = eps_noise * noise_t[k] * Hdecoh
        Ht = H0 + Hd + Hn
        psi = expm_multiply((-1j*dt) * Ht, psi)
        t += dt
    return psi


def fidelity(psi_ref: np.ndarray, psi: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi_ref, psi))**2)


def main() -> None:
    cfg = RDSConfig()
    rng = np.random.default_rng(7)

    # Build system
    pos, radii, shells = build_geometry(cfg)
    A = build_adjacency(pos, radii, shells, cfg)
    H0 = build_hamiltonian(A, radii, cfg)
    Hcoup = build_between_shell_hopping(A, cfg)

    # Spectrum
    N = H0.shape[0]
    k = min(120, N-2)
    evals, evecs = eigsh(H0, k=k, which='SA')
    ord = np.argsort(evals); evals = evals[ord]; evecs = evecs[:, ord]

    # Identify shell reps
    lnlam = cfg.derived()['ln_lambda']
    x_vals = np.zeros(k)
    for i in range(k):
        prob = np.abs(evecs[:, i])**2
        mean_r = float(np.dot(prob, radii))
        x_vals[i] = math.log(mean_r)/lnlam
    reps = {}
    for i in range(k):
        n = int(round(x_vals[i]))
        if 0 <= n < cfg.num_shells and n not in reps:
            reps[n] = i

    n0, n1 = 0, 1
    k0, k1 = reps[n0], reps[n1]
    w_res = float(evals[k1] - evals[k0])

    # Initial superposition
    psi0 = (evecs[:, k0] + evecs[:, k1]) / math.sqrt(2.0)

    # Noise operator: diagonal random potential
    Hdecoh = diags(rng.normal(size=N), 0, format='csr')

    # Reference (no drive, no noise) evolution
    steps = int(math.ceil(cfg.T/cfg.dt))
    psi_ref = psi0.copy()
    t = 0.0
    for _ in range(steps):
        psi_ref = expm_multiply((-1j*cfg.dt) * H0, psi_ref)
        t += cfg.dt

    # Scenarios
    noise_t = ornstein_uhlenbeck(cfg.dt, cfg.T, cfg.noise_tau, 1.0, rng)

    def run_case(eps_drive: float, eps_noise: float, w_drive: float) -> float:
        psi = time_evolve(H0, Hcoup, Hdecoh, psi0, w_drive, eps_drive, noise_t, eps_noise, cfg.dt)
        return fidelity(psi_ref, psi)

    # a) Noise only
    F_noise_only = run_case(0.0, cfg.noise_eps, w_res)
    # b) Resonant only
    F_res_only = run_case(cfg.drive_eps, 0.0, w_res)
    # c) Resonant + Noise (matched)
    F_res_plus_noise = run_case(cfg.drive_eps, cfg.noise_eps, w_res)
    # d) Off-resonant + Noise (control)
    F_off_plus_noise = run_case(cfg.drive_eps, cfg.noise_eps, 0.5*w_res)

    # Save
    os.makedirs('outputs', exist_ok=True)
    import json
    results = {
        'w_res': w_res,
        'F_noise_only': F_noise_only,
        'F_res_only': F_res_only,
        'F_res_plus_noise': F_res_plus_noise,
        'F_off_plus_noise': F_off_plus_noise,
        'params': cfg.__dict__,
    }
    with open('outputs/resonant_decoherence_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Quick bar plot
    labels = ['noise only', 'res only', 'res+noise', 'off+noise']
    vals = [F_noise_only, F_res_only, F_res_plus_noise, F_off_plus_noise]
    colors = ['#999', '#4caf50', '#2196f3', '#f44336']
    plt.figure(figsize=(7,4))
    plt.bar(labels, vals, color=colors)
    plt.ylabel('Fidelity vs noiseless reference')
    plt.title('Resonant Decoherence Suppression')
    plt.tight_layout()
    plt.savefig('outputs/resonant_decoherence_fidelity.png', dpi=150)
    plt.close()

    print('Resonant decoherence suppression complete.')
    print('w_res =', w_res)
    print('F(noise only)=', F_noise_only)
    print('F(res only)=', F_res_only)
    print('F(res+noise)=', F_res_plus_noise)
    print('F(off+noise)=', F_off_plus_noise)


if __name__ == '__main__':
    main()
