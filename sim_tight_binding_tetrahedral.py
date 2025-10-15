#!/usr/bin/env python3
import os
import sys
import site
import math

# Ensure user site-packages (where pip may have installed) is on sys.path
try:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
except Exception:
    pass

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')  # headless backend for non-interactive environments
import matplotlib.pyplot as plt


@dataclass
class Params:
    num_shells: int = 20
    nodes_per_shell: int = 4  # tetrahedral corners per shell
    lambda_scale: float = math.sqrt(6.0) / 2.0
    t: float = 1.0
    V0: float = 5.0
    base_radius: float = 1.0  # r_0
    between_shell_neighbor_factor: float = 0.45  # fraction of geometric mean radius used for adaptive cutoff
    within_shell_neighbor_factor: float = 1.05  # fraction of shell chord length cutoff (>=1 to include true edges)
    random_rotate_each_shell: bool = True  # avoid perfect degeneracies
    random_seed: int = 123


def tetrahedron_vertices(radius: float) -> np.ndarray:
    # Regular tetrahedron centered at origin with circumscribed sphere radius 1, then scaled by radius.
    # One canonical set of vertices (normalized to unit length):
    verts = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])  # all have same length sqrt(3), normalize to 1
    verts *= radius
    return verts


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    # Random rotation using QR decomposition (Haar measure approximation)
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    # Ensure a proper rotation (determinant +1)
    Q *= np.sign(np.linalg.det(Q))
    return Q


def build_geometry(params: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(params.random_seed)
    lambda_scale = params.lambda_scale
    num_shells = params.num_shells
    nodes_per_shell = params.nodes_per_shell

    positions = []
    radii = []
    shell_indices = []

    for n in range(num_shells):
        r_n = params.base_radius * (lambda_scale ** n)
        base_verts = tetrahedron_vertices(r_n)
        if params.random_rotate_each_shell:
            R = random_rotation_matrix(rng)
            base_verts = (R @ base_verts.T).T
        positions.append(base_verts)
        radii.extend([r_n] * nodes_per_shell)
        shell_indices.extend([n] * nodes_per_shell)

    positions = np.vstack(positions)  # (N, 3)
    radii = np.array(radii)
    shell_indices = np.array(shell_indices)
    return positions, radii, shell_indices


def build_adjacency(positions: np.ndarray, radii: np.ndarray, shell_indices: np.ndarray, params: Params) -> csr_matrix:
    N = positions.shape[0]
    rows = []
    cols = []
    data = []

    # Pre-group indices by shell for efficiency
    num_shells = params.num_shells
    nodes_per_shell = params.nodes_per_shell
    assert N == num_shells * nodes_per_shell

    shell_to_indices = [np.arange(s * nodes_per_shell, (s + 1) * nodes_per_shell) for s in range(num_shells)]

    # Within-shell connections: fully connect the 4 vertices of the tetrahedron if they are below chord cutoff
    # Chord length between any pair on circumscribed sphere of radius r is <= 2r. For regular tetrahedron, pairwise angle is arccos(-1/3), chord = 2 r sin(theta/2)
    theta = math.acos(-1.0 / 3.0)
    chord_length_factor = 2.0 * math.sin(theta / 2.0)  # ~1.633

    for s in range(num_shells):
        idx = shell_to_indices[s]
        r_s = radii[idx[0]]
        cutoff_within = params.within_shell_neighbor_factor * chord_length_factor * r_s
        for i_local in range(nodes_per_shell):
            for j_local in range(i_local + 1, nodes_per_shell):
                i = idx[i_local]
                j = idx[j_local]
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= cutoff_within:
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([1.0, 1.0])

    # Between adjacent shells: connect each node to its nearest node(s) in the adjacent shell only
    factor = params.between_shell_neighbor_factor
    for s in range(num_shells - 1):
        idx_s = shell_to_indices[s]
        idx_sp1 = shell_to_indices[s + 1]
        r_s = radii[idx_s[0]]
        r_sp1 = radii[idx_sp1[0]]
        # Adaptive cutoff based on geometric mean radius times factor
        cutoff_between = factor * math.sqrt(r_s * r_sp1)

        # Track per-node connectivity across this shell boundary
        has_forward = {int(i): False for i in idx_s}
        has_backward = {int(j): False for j in idx_sp1}

        # For each node in shell s, connect to the nearest node in shell s+1 within cutoff
        for i in idx_s:
            diffs = positions[idx_sp1] - positions[i]
            dists = np.linalg.norm(diffs, axis=1)
            j_local = int(np.argmin(dists))
            j = int(idx_sp1[j_local])
            if dists[j_local] <= cutoff_between:
                rows.extend([int(i), j])
                cols.extend([j, int(i)])
                data.extend([1.0, 1.0])
                has_forward[int(i)] = True
                has_backward[j] = True

        # Symmetrically, for each node in s+1, connect to nearest in s within cutoff
        for j in idx_sp1:
            diffs = positions[idx_s] - positions[j]
            dists = np.linalg.norm(diffs, axis=1)
            i_local = int(np.argmin(dists))
            i = int(idx_s[i_local])
            if dists[i_local] <= cutoff_between:
                rows.extend([i, int(j)])
                cols.extend([int(j), i])
                data.extend([1.0, 1.0])
                has_forward[i] = True
                has_backward[int(j)] = True

        # Fallback to ensure connectivity across shells: if any node lacks a neighbor across the boundary,
        # connect it to its nearest node regardless of cutoff (still only nearest neighbors).
        for i in idx_s:
            if not has_forward[int(i)]:
                diffs = positions[idx_sp1] - positions[i]
                dists = np.linalg.norm(diffs, axis=1)
                j_local = int(np.argmin(dists))
                j = int(idx_sp1[j_local])
                rows.extend([int(i), j])
                cols.extend([j, int(i)])
                data.extend([1.0, 1.0])
                has_forward[int(i)] = True
                has_backward[j] = True

        for j in idx_sp1:
            if not has_backward[int(j)]:
                diffs = positions[idx_s] - positions[j]
                dists = np.linalg.norm(diffs, axis=1)
                i_local = int(np.argmin(dists))
                i = int(idx_s[i_local])
                rows.extend([i, int(j)])
                cols.extend([int(j), i])
                data.extend([1.0, 1.0])
                has_forward[i] = True
                has_backward[int(j)] = True

    A = coo_matrix((data, (rows, cols)), shape=(N, N))
    # Remove potential duplicate edges by summing and binarizing
    A.sum_duplicates()
    A.data[:] = 1.0
    return A.tocsr()


def build_hamiltonian(A: csr_matrix, radii: np.ndarray, params: Params) -> csr_matrix:
    # Tight-binding: H = -t * A + diag(V(r)) with V(r) = V0 * (ln r / ln lambda)^2
    t = params.t
    V0 = params.V0
    lam = params.lambda_scale
    ln_lam = math.log(lam)

    # Avoid r=0 by construction; base_radius>0
    ln_r = np.log(radii)
    V = V0 * (ln_r / ln_lam) ** 2
    H = (-t) * A.copy()
    H = H.tocsr()
    H = H + diags(V, format='csr')
    return H


def radial_expectation(psi: np.ndarray, radii: np.ndarray) -> float:
    prob = np.abs(psi) ** 2
    prob /= prob.sum()
    return float(np.dot(prob, radii))


def participation_ratio(psi: np.ndarray) -> float:
    prob = np.abs(psi) ** 2
    norm2 = prob.sum()
    if norm2 == 0:
        return 0.0
    pr = (norm2 ** 2) / (np.sum(prob ** 2) + 1e-18)
    return float(pr)


def run_sim(params: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions, radii, shell_indices = build_geometry(params)
    A = build_adjacency(positions, radii, shell_indices, params)
    H = build_hamiltonian(A, radii, params)

    N = H.shape[0]
    k = min(100, N - 2)  # eigsh requires k < N - 1

    # Shift-invert could be used, but lowest algebraic is fine for positive V
    evals, evecs = eigsh(H, k=k, which='SA')

    # Sort ascending
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    r_expect = np.array([radial_expectation(evecs[:, i], radii) for i in range(evecs.shape[1])])
    pr_values = np.array([participation_ratio(evecs[:, i]) for i in range(evecs.shape[1])])
    x_vals = np.log(r_expect) / math.log(params.lambda_scale)
    return evals, evecs, x_vals, r_expect, pr_values


def verify_and_plot(evals: np.ndarray, x_vals: np.ndarray, pr_values: np.ndarray, params: Params) -> None:
    n_pred = np.round(x_vals)
    # A simple quadratic fit of E vs n (through origin not enforced)
    coeffs = np.polyfit(n_pred, evals, deg=2)
    n_fit = np.linspace(n_pred.min() - 0.5, n_pred.max() + 0.5, 400)
    E_fit = np.polyval(coeffs, n_fit)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sc = ax[0].scatter(x_vals, evals, c=pr_values, cmap='viridis', s=24, edgecolor='k', linewidths=0.3)
    ax[0].plot(n_fit, E_fit, 'r--', label=f'Quadratic fit: {coeffs[0]:.3f} n^2 + {coeffs[1]:.3f} n + {coeffs[2]:.3f}')
    ax[0].set_xlabel(r"$x_k = \ln \langle r \rangle_k / \ln \lambda$")
    ax[0].set_ylabel(r"$E_k$")
    ax[0].set_title("Eigenvalues vs scaled radial index")
    ax[0].legend()
    cbar = plt.colorbar(sc, ax=ax[0])
    cbar.set_label("Participation Ratio (PR)")

    ax[1].plot(np.arange(len(evals)), pr_values, 'o-', ms=3)
    ax[1].set_xlabel("Eigenstate index k (sorted)")
    ax[1].set_ylabel("Participation Ratio (PR)")
    ax[1].set_title("Participation ratios of lowest states")

    # Add vertical lines at integer n for visual clustering
    for n_int in range(int(np.floor(x_vals.min())) - 1, int(np.ceil(x_vals.max())) + 2):
        ax[0].axvline(n_int, color='gray', alpha=0.2, linewidth=1)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/eigenvalues_vs_x_and_pr.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    params = Params()
    evals, evecs, x_vals, r_expect, pr_values = run_sim(params)
    # Save numerical results
    os.makedirs('outputs', exist_ok=True)
    np.savez('outputs/results.npz', evals=evals, x_vals=x_vals, r_expect=r_expect, pr_values=pr_values)
    print(f"Computed {len(evals)} eigenpairs. Min/Max E: {evals.min():.6f}/{evals.max():.6f}")
    print("Saved results to outputs/results.npz and plots to outputs/eigenvalues_vs_x_and_pr.png")
    verify_and_plot(evals, x_vals, pr_values, params)

    # Additional Phase 1 metrics and report
    n_pred = np.round(x_vals)
    coeffs = np.polyfit(n_pred, evals, deg=2)
    E_fit = np.polyval(coeffs, n_pred)
    # R^2 for quadratic fit
    ss_res = float(np.sum((evals - E_fit) ** 2))
    ss_tot = float(np.sum((evals - np.mean(evals)) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-18))
    # Localization: mean |x - n|
    loc_dev = float(np.mean(np.abs(x_vals - n_pred)))
    # Degeneracy: count near-degenerate quartets per integer n by grouping x near n
    # A simple method: for each integer n in observed range, count states with |x-n| < 0.15
    n_min = int(np.floor(x_vals.min()))
    n_max = int(np.ceil(x_vals.max()))
    quartet_counts = {}
    for n in range(n_min, n_max + 1):
        mask = np.abs(x_vals - n) < 0.15
        quartet_counts[n] = int(np.sum(mask))

    report_lines = []
    report_lines.append("Phase 1 metrics")
    report_lines.append(f"lambda_scale = {params.lambda_scale:.6f}, V0 = {params.V0:.3f}, t = {params.t:.3f}")
    report_lines.append(f"Quadratic fit E ~ a n^2 + b n + c: a={coeffs[0]:.6f}, b={coeffs[1]:.6f}, c={coeffs[2]:.6f}")
    report_lines.append(f"R^2 = {r2:.8f}")
    report_lines.append(f"Localization mean |x - n| = {loc_dev:.6e}")
    # Summarize quartet counts
    q_summary = ", ".join([f"n={n}:{cnt}" for n, cnt in quartet_counts.items() if cnt > 0])
    report_lines.append(f"States per shell (|x-n|<0.15): {q_summary}")

    with open('outputs/phase1_report.txt', 'w') as f:
        f.write("\n".join(report_lines) + "\n")

    print("Saved Phase 1 metrics to outputs/phase1_report.txt")
