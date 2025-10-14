#!/usr/bin/env python3
import os
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, expm_multiply

# ---------- Shared geometry from Phase 1 ----------

@dataclass
class Params:
    num_shells: int = 20
    nodes_per_shell: int = 4
    lambda_scale: float = math.sqrt(6.0) / 2.0
    t: float = 1.0
    V0: float = 5.0
    base_radius: float = 1.0
    between_shell_neighbor_factor: float = 0.45
    within_shell_neighbor_factor: float = 1.05
    random_rotate_each_shell: bool = True
    random_seed: int = 123


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


def build_geometry(params: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(params.random_seed)
    positions = []
    radii = []
    shell_indices = []
    for n in range(params.num_shells):
        r_n = params.base_radius * (params.lambda_scale ** n)
        base_verts = tetrahedron_vertices(r_n)
        if params.random_rotate_each_shell:
            R = random_rotation_matrix(rng)
            base_verts = (R @ base_verts.T).T
        positions.append(base_verts)
        radii.extend([r_n] * params.nodes_per_shell)
        shell_indices.extend([n] * params.nodes_per_shell)
    return np.vstack(positions), np.array(radii), np.array(shell_indices)


def build_adjacency(positions: np.ndarray, radii: np.ndarray, shell_indices: np.ndarray, params: Params) -> csr_matrix:
    N = positions.shape[0]
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    num_shells = params.num_shells
    p = params.nodes_per_shell
    shell_to_indices = [np.arange(s * p, (s + 1) * p) for s in range(num_shells)]

    theta = math.acos(-1.0 / 3.0)
    chord_length_factor = 2.0 * math.sin(theta / 2.0)

    # within-shell
    for s in range(num_shells):
        idx = shell_to_indices[s]
        r_s = radii[idx[0]]
        cutoff_within = params.within_shell_neighbor_factor * chord_length_factor * r_s
        for i_local in range(p):
            for j_local in range(i_local + 1, p):
                i = int(idx[i_local]); j = int(idx[j_local])
                if np.linalg.norm(positions[i] - positions[j]) <= cutoff_within:
                    rows.extend([i, j]); cols.extend([j, i]); data.extend([1.0, 1.0])

    # between shells (nearest only with connectivity fallback)
    factor = params.between_shell_neighbor_factor
    for s in range(num_shells - 1):
        idx_s = shell_to_indices[s]
        idx_sp1 = shell_to_indices[s + 1]
        r_s = radii[idx_s[0]]; r_sp1 = radii[idx_sp1[0]]
        cutoff_between = factor * math.sqrt(r_s * r_sp1)
        has_f = {int(i): False for i in idx_s}
        has_b = {int(j): False for j in idx_sp1}
        for i in idx_s:
            dists = np.linalg.norm(positions[idx_sp1] - positions[int(i)], axis=1)
            jloc = int(np.argmin(dists)); j = int(idx_sp1[jloc])
            if dists[jloc] <= cutoff_between:
                rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([1.0, 1.0])
                has_f[int(i)] = True; has_b[j] = True
        for j in idx_sp1:
            dists = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
            iloc = int(np.argmin(dists)); i = int(idx_s[iloc])
            if dists[iloc] <= cutoff_between:
                rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([1.0, 1.0])
                has_f[i] = True; has_b[int(j)] = True
        for i in idx_s:
            if not has_f[int(i)]:
                dists = np.linalg.norm(positions[idx_sp1] - positions[int(i)], axis=1)
                jloc = int(np.argmin(dists)); j = int(idx_sp1[jloc])
                rows.extend([int(i), j]); cols.extend([j, int(i)]); data.extend([1.0, 1.0])
        for j in idx_sp1:
            if not has_b[int(j)]:
                dists = np.linalg.norm(positions[idx_s] - positions[int(j)], axis=1)
                iloc = int(np.argmin(dists)); i = int(idx_s[iloc])
                rows.extend([i, int(j)]); cols.extend([int(j), i]); data.extend([1.0, 1.0])

    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    A.sum_duplicates(); A.data[:] = 1.0
    return A


def build_hamiltonian(A: csr_matrix, radii: np.ndarray, params: Params) -> csr_matrix:
    lam = params.lambda_scale
    V = params.V0 * (np.log(radii) / math.log(lam)) ** 2
    return (-params.t) * A + diags(V, format='csr')


# ---------- Entanglement Utilities ----------

def mask_for_first_N_shells(N_shells: int, params: Params) -> np.ndarray:
    p = params.nodes_per_shell
    N = params.num_shells * p
    mask = np.zeros(N, dtype=bool)
    upto = min(N_shells * p, N)
    mask[:upto] = True
    return mask


def single_particle_entropy_for_cut(psi: np.ndarray, mask_A: np.ndarray) -> float:
    # Fock-space reduced entropy across bipartition A|B for a single-particle pure state
    prob = np.abs(psi) ** 2
    pA = float(prob[mask_A].sum())
    pB = 1.0 - pA
    # entropy S = -[pA ln pA + pB ln pB]
    eps = 1e-18
    terms = []
    if pA > eps:
        terms.append(-pA * math.log(pA))
    if pB > eps:
        terms.append(-pB * math.log(pB))
    return float(sum(terms))


def two_shell_entropy(psi: np.ndarray, shell_n: int, params: Params) -> float:
    # Reduce to A=shell n, B=shell n+1, trace out rest (vacuum mix)
    p = params.nodes_per_shell
    N = params.num_shells * p
    idxA = np.arange(shell_n * p, (shell_n + 1) * p)
    idxB = np.arange((shell_n + 1) * p, (shell_n + 2) * p)
    prob = np.abs(psi) ** 2
    pA = float(prob[idxA].sum())
    pB = float(prob[idxB].sum())
    pAB = pA + pB
    pOut = max(0.0, 1.0 - pAB)
    # mixture entropy + pure-state entanglement inside AB
    eps = 1e-18
    mix = 0.0
    if pOut > eps:
        mix += -pOut * math.log(pOut)
    if pAB > eps:
        mix += -pAB * math.log(pAB)
        q = pA / pAB
        if q > eps and (1.0 - q) > eps:
            mix += pAB * ( - q * math.log(q) - (1.0 - q) * math.log(1.0 - q) )
    return float(mix)


def correlation_matrix(evecs: np.ndarray, occ_indices: List[int]) -> np.ndarray:
    # C = sum_k |psi_k><psi_k|
    V = evecs[:, occ_indices]
    return V @ V.conj().T


def many_body_entropy_for_cut(C: np.ndarray, mask_A: np.ndarray) -> float:
    # Restrict correlation matrix to A
    idx = np.where(mask_A)[0]
    CA = C[np.ix_(idx, idx)]
    # Eigenvalues in [0,1]
    vals = np.clip(np.linalg.eigvalsh(CA), 0.0, 1.0)
    eps = 1e-18
    S = 0.0
    for v in vals:
        if v > eps and (1.0 - v) > eps:
            S += -v * math.log(v) - (1.0 - v) * math.log(1.0 - v)
    return float(S)


# ---------- Driving (Resonance Analog) ----------

def build_between_shell_hopping(A: csr_matrix, params: Params) -> csr_matrix:
    # Create matrix containing only between-shell edges (zero out within-shell)
    p = params.nodes_per_shell
    num_shells = params.num_shells
    rows, cols = A.nonzero()
    data = []
    r2 = []
    for i, j in zip(rows, cols):
        si = i // p
        sj = j // p
        if abs(si - sj) == 1:
            data.append(1.0)
            r2.append((i, j))
    if not data:
        return A.copy() * 0.0
    rr = np.array([i for (i, j) in r2], dtype=int)
    cc = np.array([j for (i, j) in r2], dtype=int)
    M = coo_matrix((np.array(data), (rr, cc)), shape=A.shape)
    # symmetrize
    M = (M + M.T).tocsr()
    M.sum_duplicates(); M.data[:] = 1.0
    return M


def time_evolve(H0: csr_matrix, Hdrive: csr_matrix, psi0: np.ndarray, omega: float, eps: float, T: float, dt: float) -> np.ndarray:
    psi = psi0.copy()
    t = 0.0
    while t < T - 1e-12:
        Ht = H0 + (eps * math.cos(omega * t)) * Hdrive
        # apply exp(-i Ht dt) using expm_multiply
        psi = expm_multiply((-1j * dt) * Ht, psi)
        t += dt
    return psi


# ---------- Main analysis ----------

def run_phase4():
    params = Params()
    positions, radii, shell_idx = build_geometry(params)
    A = build_adjacency(positions, radii, shell_idx, params)
    H = build_hamiltonian(A, radii, params)

    N = H.shape[0]
    k = min(120, N - 2)
    evals, evecs = eigsh(H, k=k, which='SA')
    ord = np.argsort(evals)
    evals = evals[ord]
    evecs = evecs[:, ord]

    # Identify localized representatives near integer shells using x = ln <r> / ln λ
    lam = params.lambda_scale
    x_vals = np.empty(k)
    for i in range(k):
        prob = np.abs(evecs[:, i]) ** 2
        mean_r = float(np.dot(prob, radii))
        x_vals[i] = math.log(mean_r) / math.log(lam)

    # Pick states: ground state k0 and one near n=5
    k0 = 0
    target_n = 5
    near5 = int(np.argmin(np.abs(x_vals - target_n)))

    # 1) Two-shell entanglement at boundary n for selected state
    two_shell_S = []
    for n in range(params.num_shells - 1):
        S_loc = two_shell_entropy(evecs[:, near5], n, params)
        two_shell_S.append(S_loc)

    # 2) Scaling: single-particle entanglement across cuts N=1..19
    cuts = list(range(1, params.num_shells))
    S_k0 = []
    S_k5 = []
    for Ncut in cuts:
        mask = mask_for_first_N_shells(Ncut, params)
        S_k0.append(single_particle_entropy_for_cut(evecs[:, k0], mask))
        S_k5.append(single_particle_entropy_for_cut(evecs[:, near5], mask))

    # 3) Many-body correlation entropy: occupy one localized state per shell up to n_max
    # Select representative per integer shell index by nearest x
    reps: Dict[int, int] = {}
    for i in range(k):
        n_est = int(round(x_vals[i]))
        if 0 <= n_est < params.num_shells and n_est not in reps:
            reps[n_est] = i
    occ_shells = sorted([n for n in reps.keys() if n <= 15])  # occupy first ~16 shells
    occ_indices = [reps[n] for n in occ_shells]
    C = correlation_matrix(evecs, occ_indices)
    S_many = []
    for Ncut in cuts:
        mask = mask_for_first_N_shells(Ncut, params)
        S_many.append(many_body_entropy_for_cut(C, mask))

    # Info capacity prediction (bits)
    I_bits = np.array(cuts) * math.log2(lam)
    # Convert entanglement nats -> bits
    S_k0_bits = np.array(S_k0) / math.log(2.0)
    S_k5_bits = np.array(S_k5) / math.log(2.0)
    S_many_bits = np.array(S_many) / math.log(2.0)

    os.makedirs('outputs', exist_ok=True)

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(range(params.num_shells - 1), two_shell_S, '-o', ms=3)
    ax[0].set_xlabel('n (adjacent shells n|n+1)')
    ax[0].set_ylabel('Two-shell entanglement S (nats)')
    ax[0].set_title('Local two-shell entanglement for state near n=5')

    ax[1].plot(cuts, S_k0_bits, 'o-', label='single-particle: ground')
    ax[1].plot(cuts, S_k5_bits, 'o-', label='single-particle: near n=5')
    ax[1].plot(cuts, S_many_bits, 'o-', label='many-body: 1 per shell (≤15)')
    ax[1].plot(cuts, I_bits, 'k--', label='I_N = N log2 λ')
    ax[1].set_xlabel('N shells in A')
    ax[1].set_ylabel('Entropy / Capacity (bits)')
    ax[1].set_title('Scaling vs geometric capacity')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('outputs/phase4_entanglement_scaling.png', dpi=150)
    plt.close(fig)

    # ---------- Resonance analog ----------
    # Choose state localized near n*=5 as initial
    nstar = target_n
    psi0 = evecs[:, near5]
    H0 = H
    Hdrive = build_between_shell_hopping(A, params)
    eps = 0.2
    dt = 0.05
    T = 10.0
    lam = params.lambda_scale
    w0 = 0.5
    omegas = [w0 * (lam ** j) for j in range(-3, 4)]

    def S_cut_for_state(psi: np.ndarray, Ncut: int) -> float:
        mask = mask_for_first_N_shells(Ncut, params)
        return single_particle_entropy_for_cut(psi, mask)/math.log(2.0)

    results_res = []
    for w in omegas:
        psiT = time_evolve(H0, Hdrive, psi0, w, eps, T, dt)
        S_after = S_cut_for_state(psiT, nstar)
        results_res.append((w, S_after))

    results_res = np.array(results_res)
    # Plot resonance curve
    plt.figure(figsize=(8,4))
    plt.semilogx(results_res[:,0], results_res[:,1], 'o-')
    plt.xlabel('ω_drive (log scale)')
    plt.ylabel('Entanglement across cut N=5 (bits)')
    plt.title('Driven entanglement vs ω (resonance analog)')
    plt.tight_layout()
    plt.savefig('outputs/phase4_resonance.png', dpi=150)
    plt.close()

    # Save numerical outputs
    np.savez('outputs/phase4_results.npz',
             cuts=np.array(cuts), two_shell_S=np.array(two_shell_S),
             S_k0_bits=S_k0_bits, S_k5_bits=S_k5_bits, S_many_bits=S_many_bits,
             I_bits=I_bits, resonance=results_res, evals=evals, x_vals=x_vals)

    print('Phase 4 complete. Plots saved to outputs/phase4_entanglement_scaling.png and outputs/phase4_resonance.png')


if __name__ == '__main__':
    run_phase4()
