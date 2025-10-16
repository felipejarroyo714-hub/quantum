#!/usr/bin/env python3
"""
Phase 4 Unification Enhanced Campaign
=====================================

This script implements the tetrahedral-shell tight-binding simulation
as described in the Proof Sketch for demonstrating λ-harmonic ladder behavior.

The simulation builds concentric tetrahedral shells with λ-scaling and
analyzes the resulting eigenvalue spectrum for scale-invariant properties.
"""

import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from math import sqrt, log
import csv
import os

def main():
    """Main simulation function implementing the tetrahedral-shell tight-binding model."""
    
    # Parameters
    R0 = 1.0                      # base radius (units)
    lambda_val = sqrt(6)/2        # tetrahedral kernel λ = sqrt(6)/2 ≈ 1.22474487139
    num_shells = 10               # number of concentric tetrahedral shells
    V0 = 5.0                      # potential prefactor
    t_hop = 1.0                   # hopping amplitude (positive -> will use -t in Hamiltonian)
    connect_cutoff = 1.5 * R0     # base cutoff; will scale with shell spacing

    print("Phase 4 Unification Enhanced Campaign")
    print("=" * 50)
    print(f"Parameters: R0 = {R0:.3f}, lambda = {lambda_val:.6f}, V0 = {V0:.3f}, t = {t_hop:.3f}, shells = {num_shells}")

    # Build tetrahedron vertices (regular tetrahedron centered at origin)
    # Use canonical tetrahedron vertices (normalized)
    tetra_base = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    # normalize to unit circumradius ~ sqrt(3)
    tetra_base = tetra_base / np.linalg.norm(tetra_base[0])  # make first unit length; they are symmetric

    # Create nodes: for each shell n, scale tetrahedron by R0 * lambda^n and optionally rotate slightly
    rng = np.random.default_rng(12345)  # deterministic random rotation seeds for reproducibility
    nodes = []
    radii = []
    for n in range(num_shells):
        R = R0 * (lambda_val ** n)
        # small rotation matrix via random axis-angle (keeps reproducible)
        theta = (rng.random() - 0.5) * 0.02  # tiny rotation to break degeneracies
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        ux, uy, uz = axis
        c = np.cos(theta); s = np.sin(theta)
        # Rodrigues rotation
        Rmat = np.array([
            [c + ux*ux*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
            [uy*ux*(1-c)+uz*s, c + uy*uy*(1-c), uy*uz*(1-c)-ux*s],
            [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c + uz*uz*(1-c)]
        ])
        for v in tetra_base:
            pos = (Rmat @ v) * R
            nodes.append(pos)
            radii.append(R)
    nodes = np.array(nodes)        # shape (4*num_shells, 3)
    radii = np.array(radii)        # length 4*num_shells
    N = nodes.shape[0]

    print(f"Created {N} nodes across {num_shells} tetrahedral shells")

    # Build adjacency (distance-based)
    # Use cutoff that scales with local radius spacing: approx spacing between shells = (lambda-1)*R
    # We'll use a conservative cutoff to link nearest neighbors within and between adjacent shells.
    dists = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
    # set cutoff adaptive: max distance for same-shell connectivity ~ 2*R*sin(angle/2) but we'll use fraction
    cutoff_matrix = np.zeros_like(dists)
    for i in range(N):
        for j in range(N):
            # adaptive cutoff based on mean radius
            meanR = 0.5*(radii[i]+radii[j])
            cutoff = 0.9 * (lambda_val - 1.0) * max(meanR, R0) + 0.6 * (2*R0*np.sin(1/2))  # includes small-angle base
            # ensure not too small at inner shells
            cutoff = max(cutoff, 0.6*R0)
            cutoff_matrix[i,j] = cutoff

    adj = (dists > 1e-12) & (dists <= cutoff_matrix)
    print(f"Adjacency matrix: {np.sum(adj)} connections out of {N*N} possible")
    
    # Build Hamiltonian: H = -t * A + diag(V(r))
    A = adj.astype(float)
    H = -t_hop * A.copy()
    # Add diagonal potentials
    # For r very close to zero (not happening here), guard log.
    log_lambda = np.log(lambda_val)
    scaled_log = np.log(np.maximum(radii, 1e-12)) / log_lambda  # x = ln r / ln λ
    V = V0 * (scaled_log ** 2)
    H = H + np.diag(V)

    # Make H symmetric
    H = 0.5*(H + H.T)

    # Compute eigenpairs (small N so dense solve ok)
    print("Computing eigenvalues and eigenvectors...")
    eigvals, eigvecs = sla.eigh(H)
    # Sort ascending
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Compute radial expectation <r> for each eigenstate and participation ratio
    r_expect = np.array([np.sum(np.abs(eigvecs[:, k])**2 * radii) for k in range(N)])
    PR = np.array([1.0 / np.sum(np.abs(eigvecs[:, k])**4) for k in range(N)])  # participation ratio

    # For inspection: pick lowest M eigenstates
    M = min(20, N)
    sel = np.arange(M)

    # Prepare table of results
    rows = []
    for k in sel:
        rows.append((k, eigvals[k], r_expect[k], PR[k], scaled_log[np.argmin(np.abs(radii - r_expect[k]))]))

    # Display results and a plot: eigenvalue vs scaled_log(expectation) to check ladder behavior
    print("\nLowest eigenstates (index, eigenvalue, <r>, participation_ratio, nearest_shell_scaled_log):")
    print("{:>3s}  {:>12s}  {:>10s}  {:>10s}  {:>10s}".format("k","eigval","<r>","PR","x=ln(r)/lnλ"))
    for k, eig, re, pr, x in rows:
        print(f"{k:3d}  {eig:12.6f}  {re:10.4f}  {pr:10.4f}  {x:10.4f}")

    # Plot eigenvalue vs scaled log radius expectation
    xvals = np.log(np.maximum(r_expect, 1e-12)) / log_lambda
    yvals = eigvals[:M]

    plt.figure(figsize=(10, 6))
    plt.plot(xvals[:M], yvals, marker='o', linestyle='-', markersize=6)
    plt.xlabel(r"$\ln(\langle r\rangle)/\ln(\lambda)$ (approx. shell index)")
    plt.ylabel("Eigenvalue (energy)")
    plt.title("Phase 4: Eigenvalue vs scaled-log-radius (λ-harmonic ladder)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = "outputs/phase4_lambda_harmonic_ladder.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")
    plt.show()

    # Fit linear model to check approximate quadratic relation from potential: V ∝ x^2
    # We'll fit y ≈ a x^2 + b x + c for lowest M states
    X = np.vstack([xvals[:M]**2, xvals[:M], np.ones_like(xvals[:M])]).T
    coeffs, *_ = np.linalg.lstsq(X, yvals, rcond=None)
    a,b,c = coeffs
    print("\nQuadratic fit to eigenvalue vs x (lowest M states): E ≈ a x^2 + b x + c")
    print(f"a={a:.6f}, b={b:.6f}, c={c:.6f}")

    # Also compute ratios of consecutive eigenvalues ordered by increasing x (approx shell index)
    order_by_x = np.argsort(xvals[:M])
    ordered_eigs = yvals[order_by_x]
    ratios = ordered_eigs[1:]/ordered_eigs[:-1]
    print("\nEigenvalue ratios (ordered by scaled-log expectation):")
    for i, r in enumerate(ratios):
        print(f"state {i}->{i+1}: ratio = {r:.4f}")

    # Save minimal summary to disk for review (CSV)
    outrows = [("k","eigval","r_expect","PR","x_scaled_log")]
    for k,eig,re,pr,x in rows:
        outrows.append((k,eig,re,pr,x))
    
    csv_path = "outputs/phase4_eigen_summary.csv"
    with open(csv_path,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerows(outrows)
    print(f"\nWrote summary table to {csv_path}")

    # Additional analysis: Check for λ-harmonic behavior
    print("\n" + "="*50)
    print("LAMBDA-HARMONIC LADDER ANALYSIS")
    print("="*50)
    
    # Check if eigenvalues follow λ^2 scaling
    print(f"Expected λ^2 scaling factor: {lambda_val**2:.6f}")
    
    # Analyze shell localization
    shell_assignments = np.round(scaled_log).astype(int)
    unique_shells = np.unique(shell_assignments)
    print(f"Shell indices found: {unique_shells}")
    
    # Check for degeneracy within shells
    for shell in unique_shells:
        shell_mask = shell_assignments == shell
        shell_eigs = eigvals[shell_mask]
        if len(shell_eigs) > 1:
            degeneracy = len(shell_eigs)
            energy_spread = np.max(shell_eigs) - np.min(shell_eigs)
            print(f"Shell {shell}: {degeneracy} states, energy spread = {energy_spread:.6f}")
    
    return {
        'eigenvalues': eigvals,
        'eigenvectors': eigvecs,
        'radial_expectations': r_expect,
        'participation_ratios': PR,
        'scaled_log_coords': xvals,
        'quadratic_coeffs': (a, b, c),
        'lambda_val': lambda_val,
        'num_shells': num_shells
    }

if __name__ == "__main__":
    results = main()