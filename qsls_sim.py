#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class QG_ANALOG_CONFIG:
    """
    CONFIGURATION KEY for Modeling Quantized Spacetime and Quantum Gravity Analogs.
    Based on Phase 1: Validating the Discrete λ-Harmonic Spectrum.
    """
    # Geometric Quantization Parameters
    lambda_scale: float = math.sqrt(6.0) / 2.0  # Tetrahedral Kernel (λ)
    num_shells: int = 20                        # Number of concentric λ-scaled shells
    nodes_per_shell: int = 4                    # Tetrahedral corners per shell
    base_radius: float = 1.0                    # R0 in R(n) = R0 * λ^n

    # Hamiltonian Parameters (Tight-Binding Model)
    t_hop: float = 1.0                          # Nearest-neighbor hopping amplitude (-t * A)
    V0_potential: float = 5.0                   # Prefactor for the Log-Quadratic Potential

    # Geometric Connectivity Parameters
    between_shell_neighbor_factor: float = 0.45 # Adaptive cutoff factor between shells
    within_shell_neighbor_factor: float = 1.05  # Factor for within-shell chord cutoff
    random_seed: int = 123                      # Seed for random rotations
    random_rotate_each_shell: bool = True

    def derived_constants(self) -> Dict[str, float]:
        return {'ln_lambda': math.log(self.lambda_scale)}


# ---------------- Geometry -----------------

def tetrahedron_vertices(radius: float) -> np.ndarray:
    # Regular tetrahedron centered at origin with circumscribed sphere radius 1, then scaled by radius.
    verts = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])  # normalize to unit radius on circumsphere
    verts *= radius
    return verts


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    # Random rotation using QR decomposition (Haar-like)
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    Q *= np.sign(np.linalg.det(Q))  # ensure det=+1
    return Q


def build_geometry(cfg: QG_ANALOG_CONFIG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    positions: List[np.ndarray] = []
    radii: List[float] = []
    shell_indices: List[int] = []
    for n in range(cfg.num_shells):
        r_n = cfg.base_radius * (cfg.lambda_scale ** n)
        verts = tetrahedron_vertices(r_n)
        if cfg.random_rotate_each_shell:
            R = random_rotation_matrix(rng)
            verts = (R @ verts.T).T
        positions.append(verts)
        radii.extend([r_n] * cfg.nodes_per_shell)
        shell_indices.extend([n] * cfg.nodes_per_shell)
    return np.vstack(positions), np.array(radii), np.array(shell_indices)


def build_adjacency(positions: np.ndarray, radii: np.ndarray, shell_indices: np.ndarray, cfg: QG_ANALOG_CONFIG) -> csr_matrix:
    N = positions.shape[0]
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    num_shells = cfg.num_shells
    p = cfg.nodes_per_shell
    shell_to_indices = [np.arange(s * p, (s + 1) * p) for s in range(num_shells)]

    # Within-shell fully connect tetrahedral vertices if within chord cutoff
    theta = math.acos(-1.0 / 3.0)
    chord_length_factor = 2.0 * math.sin(theta / 2.0)  # ≈ 1.633 for regular tetrahedron pairs
    for s in range(num_shells):
        idx = shell_to_indices[s]
        r_s = radii[idx[0]]
        cutoff_within = cfg.within_shell_neighbor_factor * chord_length_factor * r_s
        for i_local in range(p):
            for j_local in range(i_local + 1, p):
                i = int(idx[i_local]); j = int(idx[j_local])
                if np.linalg.norm(positions[i] - positions[j]) <= cutoff_within:
                    rows.extend([i, j]); cols.extend([j, i]); data.extend([1.0, 1.0])

    # Between adjacent shells: nearest neighbor only, symmetric, with connectivity fallback
    factor = cfg.between_shell_neighbor_factor
    for s in range(num_shells - 1):
        idx_s = shell_to_indices[s]
        idx_sp1 = shell_to_indices[s + 1]
        r_s = radii[idx_s[0]]; r_sp1 = radii[idx_sp1[0]]
        cutoff_between = factor * math.sqrt(r_s * r_sp1)
        has_forward = {int(i): False for i in idx_s}
        has_backward = {int(j): False for j in idx_sp1}
        # forward
        for i in idx_s:
            dists = np.linalg.norm(positions[idx_sp1] - positions[int(i)], axis=1)
            jloc = int(np.argmin(dists)); j = int(idx_sp1[jloc])
            if dists[jloc] <= cutoff_between:
                rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([1.0, 1.0])
                has_forward[int(i)] = True; has_backward[j] = True
        # backward
        for j in idx_sp1:
            dists = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
            iloc = int(np.argmin(dists)); i = int(idx_s[iloc])
            if dists[iloc] <= cutoff_between:
                rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([1.0, 1.0])
                has_forward[i] = True; has_backward[int(j)] = True
        # fallbacks to ensure at least one cross-shell connection per node
        for i in idx_s:
            if not has_forward[int(i)]:
                dists = np.linalg.norm(positions[idx_sp1] - positions[int(i)], axis=1)
                jloc = int(np.argmin(dists)); j = int(idx_sp1[jloc])
                rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([1.0, 1.0])
        for j in idx_sp1:
            if not has_backward[int(j)]:
                dists = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
                iloc = int(np.argmin(dists)); i = int(idx_s[iloc])
                rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([1.0, 1.0])

    A = coo_matrix((np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))), shape=(N, N)).tocsr()
    A.sum_duplicates(); A.data[:] = 1.0
    return A


def build_hamiltonian(A: csr_matrix, radii: np.ndarray, cfg: QG_ANALOG_CONFIG) -> csr_matrix:
    ln_lambda = math.log(cfg.lambda_scale)
    V = cfg.V0_potential * (np.log(radii) / ln_lambda) ** 2
    H = (-cfg.t_hop) * A + diags(V, format='csr')
    return H


# ---------------- Metrics and analysis -----------------

def participation_ratio(psi: np.ndarray) -> float:
    prob = np.abs(psi) ** 2
    norm2 = prob.sum()
    if norm2 <= 0:
        return 0.0
    return float((norm2 ** 2) / (np.sum(prob ** 2) + 1e-18))


def compute_observables(evecs: np.ndarray, radii: np.ndarray, ln_lambda: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_states = evecs.shape[1]
    r_expect = np.empty(num_states, dtype=float)
    x_vals = np.empty(num_states, dtype=float)
    pr_vals = np.empty(num_states, dtype=float)
    for i in range(num_states):
        psi = evecs[:, i]
        prob = np.abs(psi) ** 2
        mean_r = float(np.dot(prob, radii))
        r_expect[i] = mean_r
        x_vals[i] = math.log(mean_r) / ln_lambda
        pr_vals[i] = participation_ratio(psi)
    return r_expect, x_vals, pr_vals


def analyze_and_plot(evals: np.ndarray, x_vals: np.ndarray, pr_vals: np.ndarray, cfg: QG_ANALOG_CONFIG) -> Dict[str, float]:
    n_pred = np.round(x_vals)
    coeffs = np.polyfit(n_pred, evals, deg=2)
    a, b, c = coeffs
    fit = np.polyval(coeffs, n_pred)
    resid = evals - fit
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((evals - evals.mean()) ** 2))
    R2 = 1.0 - ss_res / (ss_tot + 1e-18)

    # Plot E vs x with PR color and integer guides
    n_fit = np.linspace(n_pred.min() - 0.5, n_pred.max() + 0.5, 400)
    E_fit = np.polyval(coeffs, n_fit)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sc = ax[0].scatter(x_vals, evals, c=pr_vals, cmap='viridis', s=24, edgecolor='k', linewidths=0.3)
    ax[0].plot(n_fit, E_fit, 'r--', label=f'Quadratic fit: {a:.3f} n^2 + {b:.3f} n + {c:.3f}')
    ax[0].set_xlabel(r"$x_k = \ln \langle r \rangle_k / \ln \lambda$")
    ax[0].set_ylabel(r"$E_k$")
    ax[0].set_title("Eigenvalues vs scaled radial index")
    ax[0].legend()
    cbar = plt.colorbar(sc, ax=ax[0]); cbar.set_label("Participation Ratio (PR)")
    ax[1].plot(np.arange(len(evals)), pr_vals, 'o-', ms=3)
    ax[1].set_xlabel("Eigenstate index k (sorted)")
    ax[1].set_ylabel("Participation Ratio (PR)")
    ax[1].set_title("Participation ratios of lowest states")
    for n_int in range(int(np.floor(x_vals.min())) - 1, int(np.ceil(x_vals.max())) + 2):
        ax[0].axvline(n_int, color='gray', alpha=0.2, linewidth=1)
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/eigenvalues_vs_x_and_pr.png', dpi=150)
    plt.close(fig)

    # Clustering stats and PR stats
    delta = x_vals - n_pred
    cluster = {
        'mean_abs_delta': float(np.mean(np.abs(delta))),
        'p95_abs_delta': float(np.quantile(np.abs(delta), 0.95)),
    }
    pr_stats = {
        'min': float(pr_vals.min()), 'p25': float(np.quantile(pr_vals, 0.25)),
        'median': float(np.median(pr_vals)), 'mean': float(np.mean(pr_vals)),
        'p75': float(np.quantile(pr_vals, 0.75)), 'max': float(pr_vals.max()),
    }
    return {
        'a': float(a), 'b': float(b), 'c': float(c), 'rmse': rmse, 'R2': float(R2),
        **{f'cluster_{k}': v for k, v in cluster.items()},
        **{f'pr_{k}': v for k, v in pr_stats.items()},
    }


def save_head_csv(evals: np.ndarray, x_vals: np.ndarray, r_expect: np.ndarray, pr_vals: np.ndarray, head: int = 20) -> None:
    lines = ["k,E,r_expect,PR,x,round_n"]
    for k in range(min(head, len(evals))):
        n = int(round(x_vals[k]))
        lines.append(
            f"{k},{evals[k]:.6f},{r_expect[k]:.6f},{pr_vals[k]:.6f},{x_vals[k]:.6f},{n}"
        )
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/modes_head.csv', 'w') as f:
        f.write("\n".join(lines) + "\n")


# ---------------- Main pipeline -----------------

def main() -> None:
    cfg = QG_ANALOG_CONFIG()
    ln_lambda = cfg.derived_constants()['ln_lambda']

    # Geometry and Hamiltonian
    positions, radii, shell_idx = build_geometry(cfg)
    A = build_adjacency(positions, radii, shell_idx, cfg)
    H = build_hamiltonian(A, radii, cfg)

    # Eigensolution
    N = H.shape[0]
    k = min(100, N - 2)  # eigsh requires k < N - 1
    evals, evecs = eigsh(H, k=k, which='SA')
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    # Observables
    r_expect, x_vals, pr_vals = compute_observables(evecs, radii, ln_lambda)

    # Save raw arrays
    os.makedirs('outputs', exist_ok=True)
    np.savez('outputs/results.npz', evals=evals, x_vals=x_vals, r_expect=r_expect, pr_values=pr_vals)

    # Analysis and plots
    metrics = analyze_and_plot(evals, x_vals, pr_vals, cfg)
    save_head_csv(evals, x_vals, r_expect, pr_vals, head=20)

    # Counts per integer n
    counts = {}
    for n in np.sort(np.unique(np.round(x_vals))):
        counts[int(n)] = int(np.sum(np.round(x_vals) == n))

    # Console summary
    print(f"Computed {len(evals)} eigenpairs. Min/Max E: {evals.min():.6f}/{evals.max():.6f}")
    print("Fit E ≈ a n^2 + b n + c:", {k: round(v, 6) for k, v in metrics.items() if k in ['a','b','c','rmse','R2']})
    print("Clustering:", {k: round(v, 6) for k, v in metrics.items() if k.startswith('cluster_')})
    print("PR stats:", {k: round(v, 6) for k, v in metrics.items() if k.startswith('pr_')})
    print("Counts per n:", counts)
    print("Saved arrays to outputs/results.npz, plot to outputs/eigenvalues_vs_x_and_pr.png, and CSV to outputs/modes_head.csv")


if __name__ == '__main__':
    main()
