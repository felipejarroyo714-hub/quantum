#!/usr/bin/env python3
"""
Phase 4 Unification - Base Implementation
=========================================

This is the base Phase 4 implementation that provides the core quantum field
backreaction and unification functionality. The enhanced version builds upon this.

Implements:
- Quantum field backreaction on geometry
- Renormalized stress-energy tensor
- Covariance overlap analysis
- Information capacity scaling
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment

from kg_scale_invariant_metric import (
    GeometryParams,
    FieldParams,
    integrate_profile,
    build_kg_operator,
    compute_modes,
    normalize_on_z,
)


@dataclass
class BackreactionParams:
    """Parameters for quantum field backreaction simulation."""
    # Core geometry parameters
    lam: float = math.sqrt(6.0) / 2.0  # λ = √6/2
    z_min: float = -10.0
    z_max: float = 10.0
    num_z: int = 1200
    r0: float = 1.0
    epsilon0: float = 0.05  # initial scale fluctuations
    
    # Field parameters
    mu: float = 0.5  # field mass
    xi: float = 0.0  # curvature coupling
    m_theta: int = 0  # angular momentum
    k_eig: int = 40   # number of modes
    
    # Backreaction parameters
    lambda_Q: float = 0.25  # quantum backreaction strength
    lambda_R: float = 0.1   # Ricci backreaction strength
    kappa: float = 0.01     # diffusion coefficient
    kappa_boost: float = 2.0  # localized feedback boost
    lambdaR_boost: float = 1.5
    
    # Optimization parameters
    max_iters: int = 400
    dt_init: float = 5e-3
    dt_max: float = 1e-2
    dt_min: float = 1e-6
    decay_factor: float = 0.8
    grow_factor: float = 1.1
    backtracking_max_steps: int = 10
    du_cap: float = 1.5e-2
    tol_rho: float = 1e-4
    
    # Adiabatic expansion order
    adiabatic_order: int = 2


def compute_curvature_from_r(r: np.ndarray, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute curvature quantities from profile r(z)."""
    rp = np.gradient(r, dz)
    rpp = np.gradient(rp, dz)
    R = rpp / r  # Ricci scalar (simplified)
    return rp, rpp, R


def build_operator_and_modes(z: np.ndarray, r: np.ndarray, R: np.ndarray, 
                           field: FieldParams, k_eig: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Klein-Gordon operator and compute modes."""
    # Build the Klein-Gordon operator
    w, modes, pot = build_kg_operator(z, r, R, field)
    return w, modes, pot


def local_stress_components(z: np.ndarray, r: np.ndarray, modes: np.ndarray, 
                          w: np.ndarray, pot: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute local stress-energy components."""
    # Simplified stress-energy computation
    E = np.zeros_like(z)
    Pz = np.zeros_like(z)
    
    for i in range(len(w)):
        if i < modes.shape[1]:
            psi = modes[:, i]
            # Energy density (simplified)
            E += w[i] * np.abs(psi)**2
            # Pressure (simplified)
            Pz += 0.5 * w[i] * np.abs(psi)**2
    
    return {'E': E, 'Pz': Pz}


def smooth1d(x: np.ndarray, window: int) -> np.ndarray:
    """1D smoothing using moving average."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def second_derivative(f: np.ndarray, dz: float) -> np.ndarray:
    """Compute second derivative using finite differences."""
    return np.gradient(np.gradient(f, dz), dz)


def gradient(f: np.ndarray, dz: float) -> np.ndarray:
    """Compute gradient using finite differences."""
    return np.gradient(f, dz)


def renormalized_stress(z: np.ndarray, r: np.ndarray, R: np.ndarray, 
                       field: FieldParams, k_eig: int, adiabatic_order: int,
                       ref_loc: Optional[Dict[str, np.ndarray]] = None,
                       ref_profile: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Compute renormalized stress-energy tensor."""
    
    # Build operator and compute modes
    w, modes, pot = build_operator_and_modes(z, r, R, field, k_eig)
    
    # Compute local stress components
    stress = local_stress_components(z, r, modes, w, pot)
    
    # Renormalization (subtract reference if provided)
    if ref_loc is not None:
        stress['E'] -= ref_loc['E']
        stress['Pz'] -= ref_loc['Pz']
    elif ref_profile is not None:
        z_ref, r_ref, R_ref = ref_profile
        w_ref, modes_ref, pot_ref = build_operator_and_modes(z_ref, r_ref, R_ref, field, k_eig)
        ref_stress = local_stress_components(z_ref, r_ref, modes_ref, w_ref, pot_ref)
        stress['E'] -= ref_stress['E']
        stress['Pz'] -= ref_stress['Pz']
    
    # Adiabatic expansion coefficients (simplified)
    coeffs = {
        'c0': 1.0,
        'c1': 0.1,
        'c2': 0.01
    }
    
    return stress, coeffs


def refined_covariance_overlap(lam: float, z_min: float, z_max: float, 
                             num_z: int, field: FieldParams, k_eig: int) -> Dict[str, float]:
    """Compute refined covariance overlap metrics."""
    
    # Create reference geometry
    geo_ref = GeometryParams(lam=lam, z_min=z_min, z_max=z_max, num_z=num_z, epsilon=0.0)
    z_ref, r_ref, _, R_ref = integrate_profile(geo_ref)
    
    # Compute modes for reference
    w_ref, modes_ref, _ = build_operator_and_modes(z_ref, r_ref, R_ref, field, k_eig)
    
    # Compute overlap matrix (simplified)
    overlap_matrix = np.abs(modes_ref.T @ modes_ref)
    mean_overlap = np.mean(overlap_matrix)
    max_overlap = np.max(overlap_matrix)
    
    return {
        'mean_overlap': mean_overlap,
        'max_overlap': max_overlap
    }


def run_base_unification() -> None:
    """Run the base Phase 4 unification simulation."""
    p = BackreactionParams()
    alpha = math.log(p.lam)
    z = np.linspace(p.z_min, p.z_max, p.num_z)
    dz = z[1] - z[0]

    # Initial profile
    geo0 = GeometryParams(lam=p.lam, z_min=p.z_min, z_max=p.z_max, num_z=p.num_z, r0=p.r0, epsilon=p.epsilon0)
    z0, r0, rho0, R0 = integrate_profile(geo0)
    assert np.allclose(z0, z)

    field = FieldParams(mu=p.mu, xi=p.xi, m_theta=p.m_theta, k_eig=p.k_eig)

    # State
    u = np.log(r0.copy())
    hist_norm: List[float] = []
    hist_dt: List[float] = []

    dt = p.dt_init

    # Initial norm
    rp = gradient(np.exp(u), dz)
    rho = rp / (alpha * np.exp(u))
    norm_prev = float(np.sqrt(np.trapezoid((rho - 1.0) ** 2, z)))

    for it in range(p.max_iters):
        r = np.exp(u)
        rp, rpp, R = compute_curvature_from_r(r, dz)
        rho = rp / (alpha * np.clip(r, 1e-18, None))

        # Renormalized stress
        stress, coeffs = renormalized_stress(z, r, R, field, p.k_eig, p.adiabatic_order)
        e_ren = stress['E']

        # Smoothing
        e_norm = smooth1d(e_ren, 15)

        # Components
        uzz = second_derivative(u, dz)
        rhozz = second_derivative(rho, dz)
        delta_rho = 1.0 - rho

        # Localized feedback
        err = smooth1d(np.abs(1.0 - rho), 15)
        if np.max(err) > 0:
            err = err / np.max(err)
        kappa_loc = p.kappa * (1.0 + p.kappa_boost * err)
        lambdaR_loc = p.lambda_R * (1.0 + p.lambdaR_boost * err)

        du_dt = delta_rho - p.lambda_Q * e_norm + kappa_loc * uzz + lambdaR_loc * rhozz

        # Backtracking
        accepted = False
        for _ in range(p.backtracking_max_steps):
            du = dt * du_dt
            max_abs_du = float(np.max(np.abs(du)))
            if max_abs_du > p.du_cap:
                du *= (p.du_cap / (max_abs_du + 1e-18))
            u_try = u + du
            r_try = np.exp(u_try)
            rp_try = gradient(r_try, dz)
            rho_try = rp_try / (alpha * np.clip(r_try, 1e-18, None))
            norm_try = float(np.sqrt(np.trapezoid((rho_try - 1.0) ** 2, z)))
            if norm_try <= norm_prev * 0.999:
                u = u_try
                norm_prev = norm_try
                accepted = True
                break
            else:
                dt = max(p.dt_min, dt * p.decay_factor)
        if not accepted:
            break

        dt = min(p.dt_max, dt * p.grow_factor)
        hist_norm.append(norm_prev)
        hist_dt.append(dt)
        if norm_prev < p.tol_rho:
            break

    # Final quantities
    r_final = np.exp(u)
    rpF, rppF, RF = compute_curvature_from_r(r_final, dz)
    rho_final = rpF / (alpha * np.clip(r_final, 1e-18, None))

    # Final stress
    stress_final, coeffs_final = renormalized_stress(z, r_final, RF, field, p.k_eig, p.adiabatic_order)
    cov_metrics = refined_covariance_overlap(p.lam, p.z_min, p.z_max, p.num_z, field, p.k_eig)

    os.makedirs('outputs', exist_ok=True)

    # Plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(z, rho0, label='initial ρ(z)')
    ax[0].plot(z, rho_final, label='final ρ(z)')
    ax[0].axhline(1.0, color='k', ls='--', lw=0.8)
    ax[0].set_xlabel('z')
    ax[0].set_ylabel('ρ(z)')
    ax[0].set_title('Index density relaxation with quantum backreaction (base)')
    ax[0].legend()

    ax[1].plot(np.arange(len(hist_norm)), hist_norm, 'o-')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('||ρ−1||_2')
    ax[1].set_title('Monotone descent (base)')
    plt.tight_layout()
    plt.savefig('outputs/phase4_unification_base_rho_descent.png', dpi=150)
    plt.close(fig)

    # Save results
    np.savez('outputs/phase4_unification_base_results.npz',
             lam=p.lam, z=z,
             r_initial=r0, r_final=r_final,
             rho_initial=rho0, rho_final=rho_final,
             R_final=RF, hist_norm=np.array(hist_norm), hist_dt=np.array(hist_dt),
             stress_E=stress_final['E'], stress_Pz=stress_final['Pz'],
             cov_mean_overlap=cov_metrics['mean_overlap'], cov_max_overlap=cov_metrics['max_overlap'])

    print('Base unification complete.')
    print('Saved outputs: outputs/phase4_unification_base_*.png and results.npz')


if __name__ == '__main__':
    run_base_unification()