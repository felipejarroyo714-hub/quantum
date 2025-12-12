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


# ----------------------------
# Configuration (leverages Phase 1 parameters)
# ----------------------------

@dataclass
class HoloConfig:
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

    # Entanglement boosting options to approach capacity
    boost_cross_shell: float = 1.5   # scale factor on between-shell hopping
    mix_window: int = 1              # include coupling across up to 1 additional shell layer

    def derived(self) -> Dict[str, float]:
        return {'ln_lambda': math.log(self.lambda_scale), 'log2_lambda': math.log2(self.lambda_scale)}


# ----------------------------
# Phase 1 geometry and Hamiltonian (reused)
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


def build_geometry(cfg: HoloConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        radii.extend([r_n]*cfg.nodes_per_shell)
        shells.extend([n]*cfg.nodes_per_shell)
    return np.vstack(positions), np.array(radii), np.array(shells)


def build_adjacency(positions: np.ndarray, radii: np.ndarray, shell_idx: np.ndarray, cfg: HoloConfig) -> csr_matrix:
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
            for j_local in range(i_local+1,p):
                i = int(idx[i_local]); j = int(idx[j_local])
                if np.linalg.norm(positions[i]-positions[j]) <= cutoff:
                    rows.extend([i,j]); cols.extend([j,i]); data.extend([1.0,1.0])

    # between shells up to mix_window
    for s in range(cfg.num_shells - 1):
        for w in range(1, cfg.mix_window+1):
            if s + w >= cfg.num_shells: break
            idx_s = shell_to_indices[s]
            idx_t = shell_to_indices[s+w]
            r_s = radii[idx_s[0]]; r_t = radii[idx_t[0]]
            cutoff = cfg.between_shell_neighbor_factor * math.sqrt(r_s * r_t)
            for i in idx_s:
                dists = np.linalg.norm(positions[idx_t] - positions[int(i)], axis=1)
                j = int(idx_t[int(np.argmin(dists))])
                if dists.min() <= cutoff:
                    rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([cfg.boost_cross_shell]*2)
            for j in idx_t:
                dists = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
                i = int(idx_s[int(np.argmin(dists))])
                if dists.min() <= cutoff:
                    rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([cfg.boost_cross_shell]*2)

    A = coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(N,N)).tocsr()
    A.sum_duplicates(); A.data[:] = A.data  # boosted weights preserved
    return A


def build_hamiltonian(A: csr_matrix, radii: np.ndarray, cfg: HoloConfig) -> csr_matrix:
    lnlam = math.log(cfg.lambda_scale)
    V = cfg.V0 * (np.log(radii)/lnlam)**2
    return (-cfg.t_hop) * A + diags(V, format='csr')


# ----------------------------
# Entanglement utilities (free fermions)
# ----------------------------

def mask_first_N_shells(N: int, cfg: HoloConfig) -> np.ndarray:
    p = cfg.nodes_per_shell
    mask = np.zeros(cfg.num_shells * p, dtype=bool)
    upto = min(N*p, mask.size)
    mask[:upto] = True
    return mask


def correlation_matrix(evecs: np.ndarray, occ: List[int]) -> np.ndarray:
    V = evecs[:, occ]
    return V @ V.conj().T


def entanglement_entropy_bits(C: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask)[0]
    CA = C[np.ix_(idx, idx)]
    vals = np.clip(np.linalg.eigvalsh(CA), 0.0, 1.0)
    eps = 1e-18
    S = 0.0
    for v in vals:
        if v > eps and (1.0 - v) > eps:
            S += -v * math.log(v, 2) - (1.0 - v) * math.log(1.0 - v, 2)
    return float(S)


# ----------------------------
# Main pipeline
# ----------------------------

def main() -> None:
    cfg = HoloConfig()
    d = cfg.derived()

    # Geometry and Hamiltonian
    pos, radii, shells = build_geometry(cfg)
    A = build_adjacency(pos, radii, shells, cfg)
    H = build_hamiltonian(A, radii, cfg)

    # Eigensolve
    N = H.shape[0]
    k = min(100, N-2)
    evals, evecs = eigsh(H, k=k, which='SA')
    ord = np.argsort(evals)
    evals = evals[ord]
    evecs = evecs[:, ord]

    # Identify one representative per shell (nearest integer x)
    x_vals = np.empty(k)
    for i in range(k):
        prob = np.abs(evecs[:, i])**2
        mean_r = float(np.dot(prob, radii))
        x_vals[i] = math.log(mean_r) / d['ln_lambda']
    reps: Dict[int, int] = {}
    for i in range(k):
        n = int(round(x_vals[i]))
        if 0 <= n < cfg.num_shells and n not in reps:
            reps[n] = i

    occ_shells = sorted([n for n in reps.keys() if n <= 15])
    occ = [reps[n] for n in occ_shells]

    # Correlation matrix for occupied modes
    C = correlation_matrix(evecs, occ)

    # Measure S(N) and compare with capacity I_N = N log2 λ
    cuts = list(range(1, cfg.num_shells))
    S_bits = []
    I_bits = []
    for Ncut in cuts:
        mask = mask_first_N_shells(Ncut, cfg)
        S_bits.append(entanglement_entropy_bits(C, mask))
        I_bits.append(Ncut * d['log2_lambda'])

    # Save and plot
    os.makedirs('outputs', exist_ok=True)
    np.savez('outputs/holo_entropy_results.npz', cuts=np.array(cuts), S_bits=np.array(S_bits), I_bits=np.array(I_bits), evals=evals, x_vals=x_vals)

    plt.figure(figsize=(8,4))
    plt.plot(cuts, S_bits, 'o-', label='Measured S(N) [bits]')
    plt.plot(cuts, I_bits, 'k--', label='Capacity I_N = N log2 λ')
    plt.xlabel('N shells in A')
    plt.ylabel('Entropy / Capacity (bits)')
    plt.title('Holographic Analog: Entanglement Scaling vs Geometric Capacity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/holo_entropy_scaling.png', dpi=150)
    plt.close()

    # Console summary
    print('Holographic analog complete.')
    print(f"Occupying shells: {occ_shells}")
    print(f"First/last S_bits: {S_bits[0]:.6f}, {S_bits[-1]:.6f}")
    print(f"First/last I_bits: {I_bits[0]:.6f}, {I_bits[-1]:.6f}")


if __name__ == '__main__':
    main()
