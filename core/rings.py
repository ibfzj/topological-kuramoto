"""
=========================================================
Ring Graphs and 1D Kuramoto Dynamics Visualization
=========================================================

This module provides specialized routines for constructing, analyzing,
and visualizing phase-locked states on 1D ring complexes (polygonal faces).

It serves as a simplified 1-dimensional case of the general topological
Kuramoto model, used to produce Figures 2 and 3 of the manuscript.

Functions
---------
Core solvers
-------------
- find_what_works_ring :
    1D reduction of the n-dimensional solver, computing all phase-locked
    states for a given ring partition.

Plotting utilities
------------------
- compute_zeta_range :
    Computes the valid range of the phase variable zeta_plus for consistent arcsin branches.
- get_plotting_data_ring :
    Evaluates f-curves along zeta_plus for given partitions.

Robust root-finding and stability
---------------------------------
- find_two_robust_brackets :
    Detects stable bracket intervals for brentq root finding.
- robust_find_phase_locked_roots :
    Finds intersections of projected f-curves with integer lines, robustly across omega_0 values.
- check_stable_jacobian_1d :
    Evaluates the 1D Jacobian for each solution and determines stability.

Figure generators
-----------------
- plot_ring_fcurves_grouped_stable :
    Generates Figure 2 – all distinct f-curves grouped by symmetry and annotated
    with stability markers determined from the Jacobian spectrum.
- plot_all_phase_locked_solutions_for_varying_w0_both :
    Produces Figure 3a – shows the evolution of f-curves for different omega_0 values,
    with stable and unstable intersections marked.
- plot_bifurcation_ring_branches :
    Produces Figure 3b – bifurcation diagram tracking stable branches of zeta as omega_0 varies.

Notes
-----
These routines are specifically optimized for the ring (1D cycle) topology.
They rely on the general Kuramoto dynamics utilities from `core.dynamics`.
"""


import numpy as np
from scipy.optimize import minimize, brentq, root_scalar, root as root_solver
from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import textwrap 

from core.dynamics import get_psi_minus_sp, build_partitioned_root_problem, generate_z_vectors_from_C, f_n_minus_one, is_jacobian_stable


def find_what_works_ring(
    S_plus, S_minus, B1, K1, C1,
    tol=1e-7, omega=None,
    z_range=None, verbose=False
):
    """
    1D Ring wrapper around the n-D solver, allowing manual z-range override
    (default z-range is adjusted for stable solution search, here we look for all solutions, including unstable ones).
    """
    global generate_z_vectors_from_C

    # (n+1) layer absent → use explicit empties
    B_np1 = np.zeros((0, B1.shape[0]))   # (0 x num_edges) for shape safety
    K_np1 = np.eye(0)
    C_np1 = np.zeros((0, 0))

    # partition mapping
    S0_np1, S2_np1 = [], []
    S0_n, S2_n = S_plus, S_minus

    # allow manual z-range override
    if z_range is not None:
        def generate_z_override(C_n, C_np1):
            z_n_vecs = [np.array([z]) for z in z_range]
            z_np1_vecs = [()]  # none on upper layer
            if verbose:
                print(f"manual z_n_vecs = {list(z_range)}")
            return z_n_vecs, z_np1_vecs
        generate_func = generate_z_override
    else:
        generate_func = generate_z_vectors_from_C

    omega_used = np.zeros(B1.shape[1]) if omega is None else omega

    # temporarily swap generator
    original_gen = generate_z_vectors_from_C
    generate_z_vectors_from_C = generate_func

    try:
        # manually compute psi_sp for ring 
        psi_n_sp = get_psi_minus_sp(B1, K1, omega_used)
        psi_np1_sp = np.zeros((0,))   # empty vector

        # now use the same residual builder directly 
        z_n_vecs, z_np1_vecs = generate_func(C1, C_np1)

        roots = []
        for z_np1 in z_np1_vecs:
            for z_n in z_n_vecs:
                func, target, zeta0, mode = build_partitioned_root_problem(
                    psi_np1_sp, psi_n_sp,
                    z_np1, z_n,
                    S0_np1, S0_n, S2_np1, S2_n,
                    K1, K_np1, C1, C_np1
                )

                if mode == "none":
                    continue

                result = root_solver(func, x0=zeta0, method="lm",
                                     options={"xtol": 1e-10, "ftol": 1e-10})
                fun_norm = np.linalg.norm(result.fun) if result.success else np.nan

                if verbose:
                    print(f"z_n={z_n}, success={result.success}, ||f||={fun_norm:.2e}")

                if result.success and not np.isnan(result.fun).any() and fun_norm < tol:
                    roots.append((np.array([]), z_n, np.array([]), result.x))

    finally:
        generate_z_vectors_from_C = original_gen  # restore original

    #print("Phase-locked solutions (ring):")
    #for zp, zm, zpz, zmz in roots:
    #    print(f"z- = {zm}, zeta- = {zmz}")

    return roots


# === Fig 2 related functions ===

# Styling the plots
def generate_multiset_label(S_plus, S_minus):
    count_plus = len(S_plus)
    count_minus = len(S_minus)
    parts = []

    # S_plus
    if count_plus > 0:
        parts.append(r"$|S_{[0]}^{\bullet}|$" + f" = {count_plus}")
    else:
        parts.append(r"$|S_{[0]}^{\bullet}| = \emptyset$")  # empty set

    # S_minus
    if count_minus > 0:
        parts.append(r"$|S_{[0]}^{\circ}|$" + f" = {count_minus}")
    else:
        parts.append(r"$|S_{[0]}^{\circ}| = \emptyset$")  # empty set

    return ", ".join(parts)

def arrays_equal(a, b, tol=1e-8):
    return np.allclose(a, b, atol=tol)

def add_2pi_lines(ax, range_z=(-1, 4)):
    for z in range(range_z[0], range_z[1] + 1):
        ax.axhline(y=z, color="black", linestyle="--", linewidth=1.2, alpha=0.3)
    ax.set_yticks(range(range_z[0], range_z[1] + 1))
    ax.set_yticklabels([str(z) for z in range(range_z[0], range_z[1] + 1)])

def wrap_labels(labels, width=60):
    return [textwrap.fill(label, width=width) for label in labels]

# Computing data for plots

def compute_zeta_range(psi_sp, K1, C1):
    C1 = C1.reshape(-1, 1) if C1.ndim == 1 else C1
    d = np.linalg.solve(K1, C1.T).flatten()
    zeta_mins, zeta_maxs = [], []
    for psi_i, d_i in zip(psi_sp, d):
        if abs(d_i) < 1e-12:
            continue
        lo = (-1 - psi_i) / d_i
        hi = ( 1 - psi_i) / d_i
        zeta_mins.append(min(lo, hi))
        zeta_maxs.append(max(lo, hi))
    if not zeta_mins or not zeta_maxs:
        return -np.pi, np.pi
    eps = 1e-9
    return np.max(zeta_mins) + eps, np.min(zeta_maxs) - eps

def get_plotting_data_ring(psi_sp, S_plus, S_minus, K1, C1, resolution=500):
    # exact zeta range
    zmin, zmax = compute_zeta_range(psi_sp, K1, C1)
    zetas = np.linspace(zmin, zmax, resolution)

    # d = K1^{-1} C1^T  → shape (n, r). For ring r=1.
    C1c = C1.reshape(-1, 1) if C1.ndim == 1 else C1
    d = np.linalg.solve(K1, C1c.T)          # (n, r)
    if d.shape[1] != 1:
        raise ValueError("Ring plotting expects rank(C1)=1.")

    d = d[:, 0]                              # (n,)
    psi_vals = psi_sp[None, :] + np.outer(zetas, d)    # (res, n)
    psi_vals[(psi_vals > 1) | (psi_vals < -1)] = np.nan

    f_vals = np.zeros_like(psi_vals)
    f_vals[:, S_plus]  = np.arcsin(psi_vals[:, S_plus])
    f_vals[:, S_minus] = np.pi - np.arcsin(psi_vals[:, S_minus])

    # project via C1
    f_projected = (f_vals @ C1c.T) / (2 * np.pi)       # (res, 1)
    return zetas, f_projected


# Final plotting function

def plot_ring_fcurves_grouped_stable(
    B1, K1, C1, all_partitions, *,
    omega=None, resolution=600, levels=(-4, 4), save_path="Fig2.pdf"
):
    """
    Group identical f-curves (by multiset permutations), plot one representative per group,
    and mark each intersection with stability from Jacobian test.
    """
    omega = np.zeros(B1.shape[1]) if omega is None else omega
    psi_sp = get_psi_minus_sp(B1, K1, omega)

    # ---- group identical curves; store a representative partition per curve ----
    unique_curves = []   # list of dicts: {'zetas','f','S_plus','S_minus'}
    label_groups = defaultdict(list)

    for S_plus, S_minus in all_partitions:
        zetas, f_vals = get_plotting_data_ring(psi_sp, S_plus, S_minus, K1, C1, resolution)
        found = False
        for j, cur in enumerate(unique_curves):
            if arrays_equal(f_vals, cur['f']):
                label_groups[j].append(generate_multiset_label(S_plus, S_minus))
                found = True
                break
        if not found:
            j = len(unique_curves)
            unique_curves.append({
                'zetas': zetas,
                'f': f_vals,
                'S_plus': S_plus,
                'S_minus': S_minus
            })
            label_groups[j].append(generate_multiset_label(S_plus, S_minus))

    # ---- plotting ----
    fig, ax = plt.subplots(figsize=(7, 6.4))
    add_2pi_lines(ax, range_z=levels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_curves)))

    # precompute d for residuals/J (rank-1 expected)
    C1c = C1.reshape(-1, 1) if C1.ndim == 1 else C1
    dir_vec = np.linalg.solve(K1, C1c.T)
    if dir_vec.shape[1] != 1:
        raise ValueError("Ring plotting expects rank(C1)=1.")
    d = dir_vec[:, 0]   # (n,)

    for i, cur in enumerate(unique_curves):
        zetas, f_vals = cur['zetas'], cur['f']
        S_plus, S_minus = cur['S_plus'], cur['S_minus']
        f_flat = f_vals.ravel()
        label = ", ".join(sorted(set(label_groups[i])))
        ax.plot(zetas, f_flat, color=colors[i], lw=2, label=label)

        # intersections & stability using Jacobian test
        for k in range(levels[0], levels[1] + 1):
            g = f_flat - k
            # find brackets where g changes sign and both endpoints are finite
            idxs = np.where(np.isfinite(g[:-1]) & np.isfinite(g[1:]) & (g[:-1] * g[1:] < 0))[0]
            for idx in idxs:
                a, b = zetas[idx], zetas[idx + 1]

                # scalar residual for this (S_plus, S_minus)
                def residual(z):
                    psi = psi_sp + d * z
                    f = f_n_minus_one(psi, S_plus, S_minus)
                    # project via C1 and normalize by 2pi
                    return (C1c @ f)[0] / (2*np.pi) - k

                # robust root in [a,b]
                try:
                    z_star = brentq(residual, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
                except ValueError:
                    continue

                # Jacobian & stability exactly
                psi_root = psi_sp + d * z_star
                theta    = f_n_minus_one(psi_root, S_plus, S_minus)
                J1       = -B1.T @ K1 @ np.diag(np.cos(theta)) @ B1
                stable   = is_jacobian_stable(J1)

                if stable:
                    ax.plot(z_star, k, "o",
                            color=colors[i], markersize=6,
                            markeredgecolor="black", markerfacecolor=colors[i], zorder=3)
                else:
                    ax.plot(z_star, k, "o",
                            color=colors[i], markersize=6,
                            markeredgecolor=colors[i], markerfacecolor="white",
                            alpha=0.95, zorder=3)

    # cosmetics
    ax.set_xlabel(r"$\zeta^{[-]}$", fontsize=16)
    ax.set_ylabel(r"$C_1^\top f_{0}(\psi^{[-]})/(2\pi)$", fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, wrap_labels(labels, 70), bbox_to_anchor=(0.5, -0.12),
              loc="upper center", ncol=2, frameon=False, fontsize=13)
    plt.xticks(fontsize=13); plt.yticks(fontsize=13)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# === Fig 3 related functions ===

# === Robust root-finding helpers ===

def find_two_robust_brackets(func, zeta_range, target, num=8000, merge_threshold=5e-4):
    """
    Find two robust bracket intervals where f(zeta) - target crosses zero.
    Groups nearby crossings into wider intervals (used for Fig. 3a).
    """
    zetas = np.linspace(zeta_range[0], zeta_range[1], num)
    f_vals = np.array([func(z) - target for z in zetas])

    small_brackets = []
    for i in range(len(f_vals) - 1):
        if np.isnan(f_vals[i]) or np.isnan(f_vals[i + 1]):
            continue
        if f_vals[i] * f_vals[i + 1] < 0:
            small_brackets.append((zetas[i], zetas[i + 1]))

    if not small_brackets:
        return []

    # Group nearby brackets into two wide ones
    grouped = []
    current = [small_brackets[0]]
    for (a, b) in small_brackets[1:]:
        if abs(a - current[-1][1]) < merge_threshold:
            current.append((a, b))
        else:
            grouped.append(current)
            current = [(a, b)]
    grouped.append(current)

    wide_brackets = []
    for group in grouped[:2]:
        start, end = group[0][0], group[-1][1]
        wide_brackets.append((start, end))

    return wide_brackets


def robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, z_vals, S_plus, S_minus, K1, C1):
    """
    Finds all phase-locked roots (z, zeta) for given partitions.
    Uses robust bracketing (brentq) across integer targets 2*pi*z.
    """
    roots = []
    for z in z_vals:
        target = 2 * np.pi * z
        func = lambda zeta: (C1 @ f_n_minus_one(psi_sp + np.linalg.solve(K1, C1.T)[:, 0] * zeta, S_plus, S_minus))[0]

        brackets = find_two_robust_brackets(func, (zeta_min, zeta_max), target)
        for a, b in brackets:
            try:
                result = root_scalar(lambda zeta: func(zeta) - target, bracket=[a, b], method='brentq')
                if result.converged:
                    roots.append((z, result.root))
            except ValueError:
                continue
    return roots


def check_stable_jacobian_1d(roots, psi_sp, B1, K1, C1, S_plus, S_minus):
    """
    Compute Jacobian and check stability for each root in 1D ring case.
    Returns list of indices of stable roots and corresponding eigenvalues.
    """
    stable_indices = []
    eigvals_all = []

    C1c = C1.reshape(-1, 1) if C1.ndim == 1 else C1
    d = np.linalg.solve(K1, C1c.T)[:, 0]

    for idx, (_, zeta_root) in enumerate(roots):
        psi_root = psi_sp + d * zeta_root
        theta = f_n_minus_one(psi_root, S_plus, S_minus)
        J = -B1.T @ K1 @ np.diag(np.cos(theta)) @ B1
        eigvals = np.linalg.eigvalsh(J)
        eigvals_all.append(eigvals)
        
        nonzero = np.abs(eigvals) > 1e-10
        if np.all(eigvals[nonzero] < 0):
            stable_indices.append(idx)

    return stable_indices, eigvals_all


# === Figure 3a plotting ===

def plot_all_phase_locked_solutions_for_varying_w0_both(
    Sp0, Sm0, Sp1, Sm1,
    B1, K1, C1,
    w0_values_main, w0_values_inset,
    *,
    resolution_main=1500, resolution_inset=3500,
    save_path="Fig3a.pdf"
):
    """
    Plot main and inset figures for varying omega_0 values (Fig. 3a).
    Shows two partitions (e.g. all-plus and one-minus) for a 1D ring.
    """
    fig, ax = plt.subplots(figsize=(7, 5.25))
    all_w0_values = list(w0_values_main) + [w for w in w0_values_inset if w not in w0_values_main]
    colors_all = cm.viridis(np.linspace(0, 1, len(all_w0_values)))
    color_map = {w0: colors_all[i] for i, w0 in enumerate(all_w0_values)}

    lines, labels = [], []

    # main curves
    for w0 in w0_values_main:
        omega = np.array([0, 0, 0, 0, +w0, -w0])
        psi_sp = get_psi_minus_sp(B1, K1, omega)
        zetas0, f_vals0 = get_plotting_data_ring(psi_sp, Sp0, Sm0, K1, C1, resolution_main)
        zetas1, f_vals1 = get_plotting_data_ring(psi_sp, Sp1, Sm1, K1, C1, resolution_main)

        color = color_map[w0]
        l1, = ax.plot(zetas0, f_vals0, color=color, lw=2)
        ax.plot(zetas1, f_vals1, '--', color=color, lw=2)

        # roots and stability
        zeta_min, zeta_max = compute_zeta_range(psi_sp, K1, C1)
        roots_0 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, [-1, 0, 1], Sp0, Sm0, K1, C1)
        roots_1 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, [-1, 0, 1], Sp1, Sm1, K1, C1)
        stable_0, _ = check_stable_jacobian_1d(roots_0, psi_sp, B1, K1, C1, Sp0, Sm0)
        stable_1, _ = check_stable_jacobian_1d(roots_1, psi_sp, B1, K1, C1, Sp1, Sm1)

        for idx, (z, zeta) in enumerate(roots_0):
            ax.plot(zeta, z, 'o',
                    markerfacecolor=color if idx in stable_0 else 'white',
                    markeredgecolor=color, markersize=6, zorder=3)
        for idx, (z, zeta) in enumerate(roots_1):
            ax.plot(zeta, z, 'o',
                    markerfacecolor=color if idx in stable_1 else 'white',
                    markeredgecolor=color, markersize=6, zorder=3)

        lines.append(l1)
        labels.append(fr"$\omega_0 = {w0:.2f}$")

    # horizontal lines
    for k in range(-1, 2):
        ax.axhline(y=k, linestyle='--', color='black', alpha=0.3)

    ax.set_xlabel(r"$\zeta^{[-]}$", fontsize=16)
    ax.set_ylabel(r"$C_1^\top f_0(\psi^{[-]})/(2\pi)$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    # === inset zoom ===
    axins = inset_axes(ax, width="45%", height="40%", loc='lower right', borderpad=2)
    for w0 in w0_values_inset:
        color = color_map[w0]
        omega = np.array([0, 0, 0, 0, +w0, -w0])
        psi_sp = get_psi_minus_sp(B1, K1, omega)
        zetas0, f_vals0 = get_plotting_data_ring(psi_sp, Sp0, Sm0, K1, C1, resolution_inset)
        zetas1, f_vals1 = get_plotting_data_ring(psi_sp, Sp1, Sm1, K1, C1, resolution_inset)
        axins.plot(zetas0, f_vals0, color=color, lw=1.5)
        axins.plot(zetas1, f_vals1, '--', color=color, lw=1.5)

        # find local roots in narrow window
        zeta_min, zeta_max = -0.2, -0.08
        roots_0 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, [0], Sp0, Sm0, K1, C1)
        roots_1 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, [0], Sp1, Sm1, K1, C1)
        stable_0, _ = check_stable_jacobian_1d(roots_0, psi_sp, B1, K1, C1, Sp0, Sm0)
        stable_1, _ = check_stable_jacobian_1d(roots_1, psi_sp, B1, K1, C1, Sp1, Sm1)

        for idx, (z, zeta) in enumerate(roots_0):
            axins.plot(zeta, z, 'o',
                       markerfacecolor=color if idx in stable_0 else 'white',
                       markeredgecolor=color, markersize=5, zorder=3)
        for idx, (z, zeta) in enumerate(roots_1):
            axins.plot(zeta, z, 'o',
                       markerfacecolor=color if idx in stable_1 else 'white',
                       markeredgecolor=color, markersize=5, zorder=3)

    axins.set_xlim([-0.175, -0.075])
    axins.set_ylim([-0.04, 0.03])
    axins.axhline(y=0, linestyle='--', color='black', alpha=0.3)
    axins.tick_params(axis='both', labelsize=10)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.7")

    inset_only_w0 = [w for w in w0_values_inset if w not in w0_values_main]
    for w0 in inset_only_w0:
        color = color_map[w0]
        dummy_line, = ax.plot([], [], color=color, lw=2)
        lines.append(dummy_line)
        labels.append(fr"$\omega_0 = {w0:.3f}$")

    # Final legend
    ax.legend(lines, labels, loc='upper left', fontsize=12, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# === Fig 3b related functions ===
def plot_bifurcation_ring_branches(
    Sp0, Sm0, Sp1, Sm1,
    B1, K1, C1,
    omega0_vals=np.linspace(1.307, 1.3275, 200),
    target_z=0,
    save_path="Fig3b.pdf"
):
    """
    Plot the bifurcation diagram (Fig. 3b) showing how the phase-locked roots
    move as omega_0 varies for two selected partitions in the 1D ring.

    Parameters
    ----------
    Sp0, Sm0 : list[int]
        Partition defining the all-plus branch (S0_plus, S0_minus = emptyset).
    Sp1, Sm1 : list[int]
        Partition defining the mixed branch (+++++- etc.).
    B1, K1, C1 : np.ndarray
        Boundary, coupling and cycle matrices for the ring.
    omega0_vals : iterable
        Range of omega_0 values to sweep.
    target_z : int
        Integer winding number used in Eq. (22).
    save_path : str
        Optional file path for saving the figure.
    """
    z_vals = [target_z]
    results_partitioned = []

    for i, w0 in enumerate(omega0_vals):
        omega = np.array([0, 0, 0, 0, +w0, -w0])
        psi_sp = get_psi_minus_sp(B1, K1, omega)

        # --- root finding for both partitions ---
        zeta_min, zeta_max = compute_zeta_range(psi_sp, K1, C1)
        roots_5_1 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, z_vals, Sp1, Sm1, K1, C1)
        roots_6_0 = robust_find_phase_locked_roots(psi_sp, zeta_min, zeta_max, z_vals, Sp0, Sm0, K1, C1)

        results_partitioned.append({
            "omega0": w0,
            "roots_5_1": [r[1] for r in roots_5_1],
            "roots_6_0": [r[1] for r in roots_6_0],
        })
        print(f"{i}", end=" ")

    # === extract branch data ===
    omega_list = [r["omega0"] for r in results_partitioned]
    root_5_1_a, root_5_1_b = [], []
    root_6_0_a, root_6_0_b = [], []

    for entry in results_partitioned:
        r51 = entry["roots_5_1"]
        r60 = entry["roots_6_0"]

        root_5_1_a.append(r51[0] if len(r51) > 0 else np.nan)
        root_5_1_b.append(r51[1] if len(r51) > 1 else np.nan)
        root_6_0_a.append(r60[0] if len(r60) > 0 else np.nan)
        root_6_0_b.append(r60[1] if len(r60) > 1 else np.nan)

    # detect discontinuities (NaN appears after branch disappears)
    nan_idx_60 = np.isnan(root_6_0_a)
    first_nan_60 = np.argmax(nan_idx_60) if np.any(nan_idx_60) else None
    nan_idx_51 = np.isnan(root_5_1_a)
    first_nan_51 = np.argmax(nan_idx_51) if np.any(nan_idx_51) else None

    # === plotting ===
    plt.figure(figsize=(7, 2.8))

    # partition S_{[0]}^{\circ} = [5,6]
    plt.plot(omega_list, root_5_1_a,
             label=r'$S_{[0]}^{\circ} = [5, 6]$', color='tab:blue', linestyle='--')
    plt.plot(omega_list, root_5_1_b,
             label=r'$S_{[0]}^{\circ} = [5, 6]$', color='tab:blue')

    # partition S_{[0]}^{\circ} = emptyset
    plt.plot(omega_list, root_6_0_a,
             label=r'$S_{[0]}^{\circ} = \emptyset$', color='tab:orange')

    # vertical dashed lines at bifurcations
    if first_nan_60 is not None and first_nan_60 > 0:
        plt.axvline(omega_list[first_nan_60 - 1], linestyle='--', color='k', lw=1.5)
    if first_nan_51 is not None:
        plt.axvline(omega_list[first_nan_51], linestyle='--', color='k', lw=1.5)

    plt.xlabel(r'$\omega_0$', fontsize=15)
    plt.ylabel(r'$\zeta^{[-]}$', fontsize=15, labelpad=-6)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.32, 0.83))
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    # nice x-ticks
    xticks = np.linspace(omega_list[0], omega_list[-10], 6)
    plt.xticks(xticks)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
