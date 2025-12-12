#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh


@dataclass
class TBConfig:
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
    periodic_shells: bool = True  # connect last shell to first (wrap)

    def derived(self) -> Dict[str, float]:
        return {'ln_lambda': math.log(self.lambda_scale)}


def tetrahedron_vertices(radius: float) -> np.ndarray:
    verts = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    verts *= radius
    return verts


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(M)
    Q *= np.sign(np.linalg.det(Q))
    return Q


def build_geometry(cfg: TBConfig, base_radius: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    positions: List[np.ndarray] = []
    radii: List[float] = []
    shell_indices: List[int] = []
    r0 = cfg.base_radius if base_radius is None else base_radius
    for n in range(cfg.num_shells):
        r_n = r0 * (cfg.lambda_scale ** n)
        verts = tetrahedron_vertices(r_n)
        if cfg.random_rotate_each_shell:
            R = random_rotation_matrix(rng)
            verts = (R @ verts.T).T
        positions.append(verts)
        radii.extend([r_n] * cfg.nodes_per_shell)
        shell_indices.extend([n] * cfg.nodes_per_shell)
    return np.vstack(positions), np.array(radii), np.array(shell_indices)


def build_adjacency(positions: np.ndarray, radii: np.ndarray, shell_indices: np.ndarray, cfg: TBConfig) -> csr_matrix:
    N = positions.shape[0]
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    S = cfg.num_shells
    p = cfg.nodes_per_shell
    shells = [np.arange(s * p, (s + 1) * p) for s in range(S)]

    # within-shell fully connected under chord cutoff
    theta = math.acos(-1.0 / 3.0)
    chord = 2.0 * math.sin(theta / 2.0)
    for s in range(S):
        idx = shells[s]
        r_s = radii[idx[0]]
        cutoff = cfg.within_shell_neighbor_factor * chord * r_s
        for i_loc in range(p):
            for j_loc in range(i_loc + 1, p):
                i = int(idx[i_loc]); j = int(idx[j_loc])
                if np.linalg.norm(positions[i] - positions[j]) <= cutoff:
                    rows.extend([i, j]); cols.extend([j, i]); data.extend([1.0, 1.0])

    # between shells: nearest neighbor connections to adjacent shell(s)
    factor = cfg.between_shell_neighbor_factor
    for s in range(S):
        next_s = (s + 1) % S if cfg.periodic_shells else (s + 1)
        if next_s >= S:
            continue
        idx_s = shells[s]
        idx_t = shells[next_s]
        r_s = radii[idx_s[0]]; r_t = radii[idx_t[0]]
        cutoff_between = factor * math.sqrt(r_s * r_t)
        # forward
        for i in idx_s:
            d = np.linalg.norm(positions[idx_t] - positions[int(i)], axis=1)
            j = int(idx_t[int(np.argmin(d))])
            if d.min() <= cutoff_between:
                rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([1.0, 1.0])
        # backward to ensure symmetry
        for j in idx_t:
            d = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
            i = int(idx_s[int(np.argmin(d))])
            if d.min() <= cutoff_between:
                rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([1.0, 1.0])

    A = coo_matrix((np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))), shape=(N, N)).tocsr()
    A.sum_duplicates(); A.data[:] = 1.0
    return A


def build_hamiltonian(A: csr_matrix, radii: np.ndarray, cfg: TBConfig) -> csr_matrix:
    lnlam = math.log(cfg.lambda_scale)
    V = cfg.V0 * (np.log(radii) / lnlam) ** 2
    return (-cfg.t_hop) * A + diags(V, format='csr')


def eigensolve(H: csr_matrix, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, H.shape[0] - 2)
    evals, evecs = eigsh(H, k=k, which='SA')
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def scaled_index(evecs: np.ndarray, radii: np.ndarray, lnlam: float) -> np.ndarray:
    x_vals = np.empty(evecs.shape[1])
    for i in range(evecs.shape[1]):
        prob = np.abs(evecs[:, i]) ** 2
        mean_r = float(np.dot(prob, radii))
        x_vals[i] = math.log(mean_r) / lnlam
    return x_vals


def shell_shift_map_vector(vec: np.ndarray, S: int, p: int, shift: int) -> np.ndarray:
    out = np.empty_like(vec)
    for s in range(S):
        src_s = (s + shift) % S
        out[s * p:(s + 1) * p] = vec[src_s * p:(src_s + 1) * p]
    return out


def overlap_metrics(evecs_A: np.ndarray, evecs_B: np.ndarray, S: int, p: int, shift: int) -> Tuple[float, float, np.ndarray]:
    # Build overlap matrix |<ψ_Aj | P_shift ψ_Bk>| where P_shift shifts shells of B by +shift
    # Normalize columns (eigsh returns orthonormal already)
    P_B = np.stack([shell_shift_map_vector(evecs_B[:, k], S, p, shift) for k in range(evecs_B.shape[1])], axis=1)
    O = np.abs(evecs_A.conj().T @ P_B)
    max_per_row = O.max(axis=1)
    return float(max_per_row.mean()), float(O.max()), O


def plot_e_vs_x(evals: np.ndarray, x_vals: np.ndarray, cfg: TBConfig, tag: str) -> None:
    n_pred = np.round(x_vals)
    coeffs = np.polyfit(n_pred, evals, 2)
    xs = np.linspace(n_pred.min() - 0.5, n_pred.max() + 0.5, 400)
    Ef = np.polyval(coeffs, xs)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.scatter(x_vals, evals, s=10, c='navy')
    ax.plot(xs, Ef, 'r--', label=f"fit: {coeffs[0]:.3f} n^2 + {coeffs[1]:.3f} n + {coeffs[2]:.3f}")
    for n in range(int(np.floor(x_vals.min())) - 1, int(np.ceil(x_vals.max())) + 2):
        ax.axvline(n, color='gray', alpha=0.2, lw=1)
    ax.set_xlabel(r"$x_k=\ln\langle r\rangle_k/\ln\lambda$")
    ax.set_ylabel(r"$E_k$")
    ax.set_title(f"E vs x ({tag})")
    ax.legend()
    os.makedirs('outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'outputs/tb_periodic_e_vs_x_{tag}.png', dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = TBConfig()
    d = cfg.derived()

    # Geometry/Hamiltonian A (base)
    posA, radA, shellsA = build_geometry(cfg, base_radius=cfg.base_radius)
    AA = build_adjacency(posA, radA, shellsA, cfg)
    HA = build_hamiltonian(AA, radA, cfg)
    evalsA, evecsA = eigensolve(HA, k=100)
    xA = scaled_index(evecsA, radA, d['ln_lambda'])
    plot_e_vs_x(evalsA, xA, cfg, tag='A')

    # Geometry/Hamiltonian B (λ-rescaled base radius)
    posB, radB, shellsB = build_geometry(cfg, base_radius=cfg.base_radius * cfg.lambda_scale)
    AB = build_adjacency(posB, radB, shellsB, cfg)
    HB = build_hamiltonian(AB, radB, cfg)
    evalsB, evecsB = eigensolve(HB, k=100)
    xB = scaled_index(evecsB, radB, d['ln_lambda'])
    plot_e_vs_x(evalsB, xB, cfg, tag='B')

    # Covariance under λ-rescaling: shift by +1 shell
    S = cfg.num_shells; p = cfg.nodes_per_shell
    mean_ov, max_ov, O = overlap_metrics(evecsA, evecsB, S=S, p=p, shift=+1)

    # Save metrics and matrix head
    os.makedirs('outputs', exist_ok=True)
    np.savez('outputs/tb_periodic_covariance_results.npz', evalsA=evalsA, evalsB=evalsB, xA=xA, xB=xB, overlap=O)
    with open('outputs/tb_periodic_covariance_metrics.txt', 'w') as f:
        f.write(f'mean_overlap: {mean_ov}\n')
        f.write(f'max_overlap: {max_ov}\n')

    print('TB periodic/mixed boundary covariance analysis complete.')
    print(f'Mode overlaps (λ-rescale, +1 shell shift): mean={mean_ov:.3f}, max={max_ov:.3f}')
    print('Plots: outputs/tb_periodic_e_vs_x_A.png, outputs/tb_periodic_e_vs_x_B.png')


if __name__ == '__main__':
    main()
