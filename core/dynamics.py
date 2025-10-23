"""
=========================================================
Topological Kuramoto Model — Core Dynamical Routines
=========================================================

This module implements the dynamical core of the topological Kuramoto model.
It defines the nonlinear equations, stationary states, winding-number
enumeration, and stability analysis used across all geometries.

Functions
---------

- get_psi_plus_sp, get_psi_minus_sp:
    Compute stationary particular solutions psi_plus and psi_minus for given boundary and
    coupling matrices (Eqs. 13–14).
- f_n_plus_one, f_n_minus_one:
    Nonlinear elementwise coupling functions implementing the arcsin / (pi–arcsin)
    partition structure (Eq. 20).
- f_zeta_partitioned_sum_nd:
    Evaluate the residual sums in Eq. 22 for general n-dimensional complexes.
- build_partitioned_root_problem:
    Construct the nonlinear residual functions f(zeta) = 0 for arbitrary partitions.
- generate_z_vectors_from_C:
    Generate integer winding-number candidates from kernel matrices.
- find_what_works_nd:
    Solve all phase-locked states for arbitrary n-dimensional complexes.
- find_all_plus_omega_zero_solutions :
    Unified high-level driver for computing all-plus stationary states (omega = 0).
- is_jacobian_stable:
    Check whether all non-zero eigenvalues of a Jacobian are < 0.
- check_stable_jacobians:
    Evaluate full Jacobian stability for each solution.
- check_cos_theta:
    Verify cosine-positivity condition (cos theta > 0) for all components.
- all_partitions :
    Generate all 2^n binary partitions of S into S_plus and S_minus.
- get_all_plus_partitions :
    Construct the all-plus partition for a given dimension.

Notes
-----
This module provides the mathematical engine underlying all figures and
analyses. It connects the algebraic-topological structures from
`core.complexes` with the geometric and graphical representations in
`core.rings`, `core.simplexes`, and `core.platonic_solids`.
"""


import numpy as np
from numpy.linalg import inv, norm
import math
import itertools
from itertools import product
from scipy.optimize import root as root_solver
from core.complexes import  generate_all_B_by_definition, compute_boundary_matrices, compute_kernel_sp, get_C_from_B, get_K_from_B
from core.simplexes import generate_simplices

# --- helper functions for dynamics ---

def get_psi_plus_sp(B_np1, K_np1, omega):
    """Eq 13"""
    L_down = B_np1.T @ B_np1
    lhs = L_down @ K_np1
    rhs = B_np1.T @ omega
    psi_sp = np.linalg.lstsq(lhs, rhs, rcond=None)[0] #solve (L_down @ K_np1 ) @ psi_sp_plus = B_np1.T @ omega.
    return psi_sp

def get_psi_minus_sp(B_n, K_n, omega):
    """Eq 14"""
    L_up = B_n @ B_n.T
    lhs = L_up @ K_n
    rhs = B_n @ omega
    psi_sp = np.linalg.lstsq(lhs, rhs, rcond=None)[0] #solve (L_up @ K_n ) @ psi_sp_minus = B_n @ omega.
    return psi_sp

def f_n_minus_one(psi_minus, S_plus, S_minus):
    """Eq 20"""
    if np.any(np.abs(psi_minus) > 1):
        return np.full_like(psi_minus, np.nan)
    
    f = np.zeros_like(psi_minus)
    f[S_plus] = np.arcsin(psi_minus[S_plus])
    f[S_minus] = np.pi - np.arcsin(psi_minus[S_minus])

    return f

def f_n_plus_one(psi_plus, S_plus, S_minus):
    """Eq 20"""
    if np.any(np.abs(psi_plus) > 1):
        return np.full_like(psi_plus, np.nan)
    
    f = np.zeros_like(psi_plus)

    f[S_plus] = np.arcsin(psi_plus[S_plus])
    f[S_minus] = np.pi - np.arcsin(psi_plus[S_minus])

    return f


def f_zeta_partitioned_sum_nd(psi_np1_sp, psi_n_sp, zeta_np1_vec, zeta_n_vec, 
                               S0_np1, S0_n, S2_np1, S2_n, 
                               K_n, K_np1, C_n, C_np1):
    """LHS of Eq 22."""
    C_n = C_n.reshape(-1, 1) if C_n.ndim == 1 else C_n
    C_np1 = C_np1.reshape(-1, 1) if C_np1.ndim == 1 else C_np1

    psi_np1 = psi_np1_sp + np.linalg.inv(K_np1) @ (C_np1 @ zeta_np1_vec)
    psi_n = psi_n_sp + np.linalg.inv(K_n) @ (C_n.T @ zeta_n_vec)

    if np.any(np.abs(psi_np1) > 1) or np.any(np.abs(psi_n) > 1):
        return np.full(C_n.shape[0] + C_np1.shape[1], np.nan)

    f_n_val = f_n_minus_one(psi_n, S0_np1, S0_n)
    f_np1_val = f_n_plus_one(psi_np1, S2_np1, S2_n)

    sum_n = C_n @ f_n_val
    sum_np1 = C_np1.T @ f_np1_val

    return np.concatenate([sum_n, sum_np1])


def compute_zeta_ranges_double(psi_plus_sp, psi_minus_sp, K1, K2, C1, C2):
    """
    Compute the admissible zeta-ranges for psi_plus and psi_minus given the constraint |psi| <= 1 (Eq. 18).

    Returns
    -------
    (zeta_plus_min, zeta_plus_max),
    (zeta_minus_min, zeta_minus_max)
    """
    psi_plus_sp = psi_plus_sp.flatten()
    psi_minus_sp = psi_minus_sp.flatten()

    A_plus  = (np.linalg.inv(K2) @ C2).flatten()
    A_minus = (np.linalg.inv(K1) @ C1.T).flatten()

    with np.errstate(divide='ignore', invalid='ignore'):
        zp_candidates = np.vstack([(1 - psi_plus_sp)/A_plus, (-1 - psi_plus_sp)/A_plus])
        zm_candidates = np.vstack([(1 - psi_minus_sp)/A_minus, (-1 - psi_minus_sp)/A_minus])

    # remove NaNs before taking global extrema
    zp_min, zp_max = np.nanmax(np.nanmin(zp_candidates, axis=0)), np.nanmin(np.nanmax(zp_candidates, axis=0))
    zm_min, zm_max = np.nanmax(np.nanmin(zm_candidates, axis=0)), np.nanmin(np.nanmax(zm_candidates, axis=0))

    return (zp_min, zp_max), (zm_min, zm_max)



def generate_z_vectors_from_C(C_n, C_np1):
    """
    Generate winding number candidate z_n and z_np1 vectors using the strict elementwise bound
    from Eq. (23): |z_i| < (1/4) * sum_j |C_ij|.
    Each z_i gets its own bound u_i, so we avoid a single global max.
    """

    def per_row_ranges(C, is_empty):
        if is_empty:
            return []
        m_per_row = np.sum(np.abs(C), axis=1)  # m_i = sum_j |C_ij|
        ranges = []
        for m in m_per_row:
            # strict bound: integers with |z_i| < m/4
            u = math.ceil(m / 4) - 1  # ceil(m/4) - 1 gives max |z_i|
            if u < 0:  # happens if m < 4
                ranges.append([0])  # only zero possible
            else:
                ranges.append(list(range(-u, u + 1)))
        return ranges

    def cartesian_from_ranges(ranges):
        if not ranges:
            return [()]
        return list(itertools.product(*ranges))

    is_Cn_empty = C_n.size == 0 or C_n.shape[0] == 0 or C_n.shape[1] == 0
    is_Cnp1_empty = C_np1.size == 0 or C_np1.shape[0] == 0 or C_np1.shape[1] == 0

    z_n_ranges   = per_row_ranges(C_n, is_Cn_empty)
    z_np1_ranges = per_row_ranges(C_np1.T, is_Cnp1_empty)

    z_n_vecs   = cartesian_from_ranges(z_n_ranges)
    z_np1_vecs = cartesian_from_ranges(z_np1_ranges)
    
    print(f'total of {len(z_n_vecs)} different z_n_vecs considered, and {len(z_np1_vecs)} different z_np1_vecs.')

    return z_n_vecs, z_np1_vecs

# --- equation solver for finding phase-locked solutions ---

def build_partitioned_root_problem(
    psi_np1_sp, psi_n_sp,
    z_np1, z_n,
    S0_np1, S0_n, S2_np1, S2_n,
    K_n, K_np1, C_n, C_np1
):
    """
    Eq22: Constructs a nonlinear residual function f(zeta)
    for identifying phase-locked states in the topological Kuramoto model.
    This function acts as a "problem builder" for nonlinear solvers (e.g. `scipy.root`),

    Depending on which cell dimension is active (n or n+1 or both), the routine builds
    a root-finding problem of the form:

        C_n f_{n-1}(psi_n)  =  2*pi*z_n
        C_{n+1}^\top f_{n+1}(psi_{n+1})  =  2*pi*z_{n+1}

    with the correct domain, orientation, and variable substitution
    \psi = \psi_sp + inv(K) C zeta.

    Parameters
    ----------
    psi_np1_sp : np.ndarray
        Stationary particular solution psi_plus at dimension n+1.
    psi_n_sp : np.ndarray
        Stationary particular solution psi_minus at dimension n.
    z_np1, z_n : np.ndarray
        Integer vectors (winding numbers) specifying multiples of 2pi
        for each independent harmonic mode.
    S0_np1, S0_n, S2_np1, S2_n : list[int]
        Index partitions defining local nonlinear interaction rules for
        psi_plus and psi_minus, i.e. which components use arcsin(psi) vs. pi – arcsin(psi).
    K_n, K_np1 : np.ndarray
        Coupling matrices on dimensions n and n+1.
    C_n, C_np1 : np.ndarray
        Cycle matrices defining the kernel of the boundary operator
        at dimensions n and n+1.

    Returns
    -------
    func : callable
        Residual function f(zeta_vec) whose root corresponds to a phase-locked state.
        The function automatically enforces |psi| ≤ 1, returning NaN outside the domain.
    target : np.ndarray
        The target vector 2pi*z containing the integer winding constraints.
    zeta0 : np.ndarray
        Initial guess for the solver (zeros).
    mode : str
        Indicates which subproblem was constructed:
        'only_n'   → only psi_minus (e.g. ring case)
        'only_np1' → only psi_plus
        'both'     → both psi_minus and psi_plus
        'none'     → no valid root problem (empty matrices)
    """

    C_n = C_n.reshape(-1, 1) if C_n.ndim == 1 else C_n
    C_np1 = C_np1.reshape(-1, 1) if C_np1.ndim == 1 else C_np1

    only_np1 = C_n.shape[0] == 0
    only_n = C_np1.shape[1] == 0

    if only_np1 and not only_n:
        target = 2 * np.pi * np.array(z_np1)
        def func(zeta_vec):
            psi_np1 = psi_np1_sp + np.linalg.solve(K_np1, C_np1 @ zeta_vec)
            if np.any(np.abs(psi_np1) > 1):
                return np.full(C_np1.shape[1], np.nan)
            f_plus = f_n_plus_one(psi_np1, S2_np1, S2_n)
            return C_np1.T @ f_plus - target
        zeta0 = np.zeros(C_np1.shape[1])
        return func, target, zeta0, 'only_np1'

    elif only_n and not only_np1:
        target = 2 * np.pi * np.array(z_n)
        def func(zeta_vec):
            psi_n = psi_n_sp + np.linalg.solve(K_n, C_n.T @ zeta_vec)
            if np.any(np.abs(psi_n) > 1):
                return np.full(C_n.shape[0], np.nan)
            f_minus = f_n_minus_one(psi_n, S0_np1, S0_n)
            return C_n @ f_minus - target
        zeta0 = np.zeros(C_n.shape[0])
        return func, target, zeta0, 'only_n'

    elif not only_n and not only_np1:
        target = 2 * np.pi * np.concatenate([z_n, z_np1])
        n_np1 = C_np1.shape[1]
        n_n = C_n.shape[0]
        def func(zeta_vec):
            zp = zeta_vec[:n_np1]
            zm = zeta_vec[n_np1:]

            psi_np1 = psi_np1_sp + np.linalg.solve(K_np1, C_np1 @ zp)
            psi_n = psi_n_sp + np.linalg.solve(K_n, C_n.T @ zm)

            if np.any(np.abs(psi_np1) > 1) or np.any(np.abs(psi_n) > 1):
                return np.full(len(target), np.nan)

            f_n_val = f_n_minus_one(psi_n, S0_np1, S0_n)
            f_np1_val = f_n_plus_one(psi_np1, S2_np1, S2_n)

            return np.concatenate([C_n @ f_n_val, C_np1.T @ f_np1_val]) - target

        zeta0 = np.zeros(n_np1 + n_n)
        return func, target, zeta0, 'both'

    else:
        return None, None, None, 'none'
    

def find_what_works_nd(S0_np1, S0_n, S2_np1, S2_n, B_n, B_np1, K_n, K_np1, C_n, C_np1, tol = 1e-7, omega=None):
    """
    Solve for all phase-locked states of the topological Kuramoto model
    on an arbitrary cell complex, by constructing and solving the appropriate
    nonlinear root-finding problems in zeta-space.

    This routine automatically builds the correct residual function(s) via
    `build_partitioned_root_problem`, applies numerical root-finding,
    and collects all physically valid stationary states (|psi| ≤ 1).

    Parameters
    ----------
    S0_np1, S0_n, S2_np1, S2_n : list[int]
        Partition index sets defining the nonlinear interaction rules
        for psi_plus (dimension n+1) and psi_minus (dimension n).
        Each pair (S0, S2) corresponds to domains where arcsin(psi)
        or pi − arcsin(psi) apply.
    B_n, B_np1 : np.ndarray
        Boundary matrices coupling n- and (n+1)-cells in the complex.
    K_n, K_np1 : np.ndarray
        Coupling matrices for n- and (n+1)-cells.
    C_n, C_np1 : np.ndarray
        Cycle matrices defining the harmonic subspace (kernel of B).
    tol : float, optional
        Numerical tolerance for root validation (default 1e-7).
    omega : np.ndarray, optional
        Frequency vector. Defaults to zero (homogeneous oscillators).

    Returns
    -------
    roots : list of tuple
        List of tuples (z_plus, z_minus, zeta_plus, zeta_minus)
        corresponding to all identified phase-locked solutions:
          - z_plus, z_minus : integer winding vectors
          - zeta_plus, zeta_minus : corresponding zeta solutions

    Notes
    -----
    The solver iterates over all integer vector combinations (z_n, z_np1)
    generated by `generate_z_vectors_from_C`, constructing for each a
    residual function f(zeta) = 0.

    For each case, the nonlinear equation is solved using the Levenberg–
      Marquardt method (`scipy.root(method='lm')`).

    Solutions are accepted only if:
        – the solver converged (`result.success`)
        – no NaNs appear in f(zeta)
        – the residual norm ||f(zeta)|| < tol
        – |psi| <= 1 for all components

    """
    roots = []

    # Ensure column-vector shape consistency
    C_n = C_n.reshape(-1, 1) if C_n.ndim == 1 else C_n
    C_np1 = C_np1.reshape(-1, 1) if C_np1.ndim == 1 else C_np1

    if omega is None:
        omega = np.zeros(B_n.shape[1])

    psi_n_sp = get_psi_minus_sp(B_n, K_n, omega)
    psi_np1_sp = get_psi_plus_sp(B_np1, K_np1, omega)

    z_n_vecs, z_np1_vecs = generate_z_vectors_from_C(C_n, C_np1)

    for z_np1 in z_np1_vecs:
        for z_n in z_n_vecs:

            # Construct the appropriate nonlinear problem
            func, target, zeta0, mode = build_partitioned_root_problem(
                psi_np1_sp, psi_n_sp,
                z_np1, z_n,
                S0_np1, S0_n, S2_np1, S2_n,
                K_n, K_np1, C_n, C_np1
            )

            if mode == 'none':
                continue

            result = root_solver(func, x0=zeta0, method='lm', options={'xtol': 1e-10, 'ftol': 1e-10})
            fun_norm = np.linalg.norm(result.fun) if result.success else np.nan

            #print(f"z_np1: {z_np1}, z_n: {z_n}, result: {result.success}, fun norm: {fun_norm}, mode: {mode}")

            # accept valid roots
            if result.success and not np.isnan(result.fun).any() and fun_norm < tol:
                if mode == 'only_np1':
                    roots.append((z_np1, np.array([]), result.x, np.array([])))
                elif mode == 'only_n':
                    roots.append((np.array([]), z_n, np.array([]), result.x))
                else:  # both
                    n_np1 = C_np1.shape[1]
                    roots.append((z_np1, z_n, result.x[:n_np1], result.x[n_np1:]))

    print("Phase-locked solutions:")
    for zp, zm, zeta_np1, zeta_n in roots:
        print(f'(z_plus, z_minus) = ({zp}, {zm})')
        print(f"zeta_plus = {zeta_np1}, zeta_minus = {zeta_n}")

    return roots

# --- stability analysis ---

def is_jacobian_stable(J, tol=1e-9):
    """
    Returns True if all eigenvalues of J are <= 0,
    treating values with absolute value <= tol as zero.
    Since harmonic perturbations do not affect the projected
    quantities, we include zero eigenvalues here.

    Eigenvalues are computed via symmetric eigendecomposition
      (`np.linalg.eigvalsh`).
    """
    eigvals = np.linalg.eigvalsh(J)  # for symmetric J
    eigvals[np.abs(eigvals) < tol] = 0.0  # round small values to zero
    return np.all(eigvals <= 0)


def check_stable_jacobians(
    roots, omega,
    B_n, B_np1, K_n, K_np1, C_n, C_np1,
    S0_plus_c, S0_minus_c, S2_plus_c, S2_minus_c):
    """
    
    Evaluate the **linear stability** of each identified phase-locked state
    by computing the Jacobian matrices of the topological Kuramoto dynamics.

    For every root (z_plus, z_minus, zeta_plus, zeta_minus) returned by `find_what_works_nd`, the
    function reconstructs psi_plus and psi_minus, forms the corresponding Jacobians J1
    and J2, and determines whether all non-zero eigenvalues are non-positive.

    Parameters
    ----------
    roots : list of tuple
        List of phase-locked solutions, each as
        (z_plus, z_minus, zeta_plus, zeta_minus).
    omega : np.ndarray
        Frequency vector used in computing the stationary particular solutions.
    B_n, B_np1 : np.ndarray
        Boundary matrices defining cell adjacencies at dimensions n and n+1.
    K_n, K_np1 : np.ndarray
        Coupling matrices for the n- and (n+1)-cells.
    C_n, C_np1 : np.ndarray
        Cycle matrices defining the harmonic basis of the complex.
    S0_plus_c, S0_minus_c, S2_plus_c, S2_minus_c : list[int]
        Index partitions specifying which psi components enter the nonlinear
        interaction as arcsin(psi) versus pi − arcsin(psi), for both psi_minus and psi_plus.

    Returns
    -------
    stable_indices : list[int]
        Indices of stable solutions (all non-zero eigenvalues ≤ 0).
    all_J1s : list[np.ndarray or None]
        Jacobians J1 = –Bn.T Kn diag(cos theta_minus) Bn for each root (or None if skipped).
    all_J2s : list[np.ndarray or None]
        Jacobians J2 = –Bnp1 Knp1 diag(cos theta_plus) Bnp1.T for each root (or None if skipped).

    Notes
    -----
    A solution is classified as stable if all non-zero eigenvalues of both
    J1 and J2 are negative (within tolerance, as checked by `is_jacobian_stable`).

    When either Cn or Cnp1 is empty, the corresponding Jacobian is omitted
    and stability is judged only on the available subsystem.

    This routine provides the final classification step following
    `find_what_works_nd`, completing the identification of
    stable and unstable phase-locked states.
    """
    stable_indices = []
    all_J1s = []
    all_J2s = []

    psi_minus_sp = get_psi_minus_sp(B_n, K_n, omega)
    psi_plus_sp = get_psi_plus_sp(B_np1, K_np1, omega)

    Cn_empty = C_n.size == 0 or C_n.shape[0] == 0 or C_n.shape[1] == 0
    Cnp1_empty = C_np1.size == 0 or C_np1.shape[0] == 0 or C_np1.shape[1] == 0

    for idx, (z_plus, z_minus, zeta_plus, zeta_minus) in enumerate(roots):
        is_stable = True

        if not Cn_empty:
            psi_minus = psi_minus_sp + (np.linalg.inv(K_n) @ C_n.T) @ zeta_minus
            theta_minus = f_n_minus_one(psi_minus, S0_plus_c, S0_minus_c)
            J1 = -B_n.T @ K_n @ np.diag(np.cos(theta_minus)) @ B_n
            all_J1s.append(J1)
            if not is_jacobian_stable(J1):
                is_stable = False
        else:
            all_J1s.append(None)

        if not Cnp1_empty:
            psi_plus = psi_plus_sp + (np.linalg.inv(K_np1) @ C_np1) @ zeta_plus
            theta_plus = f_n_plus_one(psi_plus, S2_plus_c, S2_minus_c)
            J2 = -B_np1 @ K_np1 @ np.diag(np.cos(theta_plus)) @ B_np1.T
            all_J2s.append(J2)
            if not is_jacobian_stable(J2):
                is_stable = False
        else:
            all_J2s.append(None)

        if is_stable:
            stable_indices.append(idx)
            print(f"Solution z+ = {z_plus}, z- = {z_minus}: is stable.")
        else:
            print(f"Solution z+ = {z_plus}, z- = {z_minus}: is unstable.")

    return stable_indices, all_J1s, all_J2s

def check_cos_theta(
    roots, omega,
    B_n, B_np1, K_n, K_np1, C_n, C_np1,
    S0_plus_c, S0_minus_c, S2_plus_c, S2_minus_c):
    """
    For each root, check if all cos(theta) values are positive.
    If a kernel is empty, skip checking its corresponding condition.
    """
    positive_indices = []
    all_CT1s = []
    all_CT2s = []

    psi_minus_sp = get_psi_minus_sp(B_n, K_n, omega)
    psi_plus_sp = get_psi_plus_sp(B_np1, K_np1, omega)

    Cn_empty = C_n.size == 0 or C_n.shape[0] == 0 or C_n.shape[1] == 0
    Cnp1_empty = C_np1.size == 0 or C_np1.shape[0] == 0 or C_np1.shape[1] == 0

    for idx, (z_plus, z_minus, zeta_plus, zeta_minus) in enumerate(roots):
        all_positive = True

        if not Cn_empty:
            psi_minus = psi_minus_sp + (np.linalg.inv(K_n) @ C_n.T) @ zeta_minus
            theta_minus = f_n_minus_one(psi_minus, S0_plus_c, S0_minus_c)
            CT1 = np.cos(theta_minus)
            all_CT1s.append(CT1)
            if not np.all(CT1 > 0):
                all_positive = False
        else:
            all_CT1s.append(None)

        if not Cnp1_empty:
            psi_plus = psi_plus_sp + (np.linalg.inv(K_np1) @ C_np1) @ zeta_plus
            theta_plus = f_n_plus_one(psi_plus, S2_plus_c, S2_minus_c)
            CT2 = np.cos(theta_plus)
            all_CT2s.append(CT2)
            if not np.all(CT2 > 0):
                all_positive = False
        else:
            all_CT2s.append(None)

        if all_positive:
            positive_indices.append(idx)
            print(f"Solution z+ = {z_plus}, z- = {z_minus}: all cos(theta) > 0.")
        else:
            print(f"Solution z+ = {z_plus}, z- = {z_minus}: some cos(theta) <= 0.")

    return positive_indices, all_CT1s, all_CT2s

# --- calculate partitions: all and only all-plus ---

def all_partitions(n):
    """
    Generate all 2^n partitions of [0, ..., n-1] into S_plus and S_minus.
    """
    partitions = []

    #generate all sequences of n bits (0 or 1), each sequence of bits is one possible partition.
    #create corresponding lists S_plus and S_minus
    for bits in product([0, 1], repeat=n):  # 0 -> S_plus, 1 -> S_minus
        S_plus = [i for i, b in enumerate(bits) if b == 0]
        S_minus = [i for i, b in enumerate(bits) if b == 1]
        partitions.append((S_plus, S_minus))

    return partitions

def get_all_plus_partitions(S_n_minus_one, S_n_plus_one):
    """
    Returns only the all plus partitions given a dimension n.
    The S_n_minus_one_PLUS contains all elements from S_n_minus_one.
    The S_n_minus_one_MINUS is empty, as we consider the all plus partitions only.
    Analogously, S_n_minus_one_PLUS contains all elements from S_n_plus_one,
    and the S_n_minus_one_MINUS is empty.
    """
    S_nm1_plus = [i for i in range(0, len(S_n_minus_one))]
    S_nm1_minus = []
    S_np1_plus = [i for i in range(0, len(S_n_plus_one))]
    S_np1_minus = []
    return S_nm1_plus, S_nm1_minus, S_np1_plus, S_np1_minus


# --- function that unifies it all into a comprehensive analysis of all-plus solutions ---

def find_all_plus_omega_zero_solutions(S, n, simplex = True, tol = 1e-7):
    """
    Compute all phase-locked, all-plus partition, stationary solutions for a given
    cell dimension `n` of a simplicial or general cell complex.
    It treats the omega = zero case only.

    This function performs the following steps:
      1. Constructs the boundary matrices `B_n` and `B_{n+1}`.
      2. Computes the kernel (cycle) and coupling matrices `C_n`, `C_{n+1}`, `K_n`, `K_{n+1}`.
      3. Builds and solves the nonlinear phase-locking equations via `find_what_works_nd`.
      4. Checks the stability (Jacobian eigenvalues) and cosine-positivity conditions
         for all identified roots.

    The routine can handle both simplicial complexes (where all cells are simplices)
    and general cell complexes (e.g., rings, Platonic solids, polyhedra with polygonal faces).

    Parameters
    ----------
    S : list of lists
        The cell complex structure `S = [S[0], S[1], ..., S[n]]`,
        where `S_k` is the list of k-cells (each cell is a list of vertex indices).
        For simplicial complexes, this corresponds to the full combinatorial structure.
    n : int
        Target cell dimension (0 <= n <= len(S)−1) at which to compute the roots.
    simplex : bool, optional
        If True, interpret `S` as a simplicial complex and generate boundary matrices
        via the formal combinatorial definition (`generate_all_B_by_definition`).
        If False, compute boundary matrices via the generalized geometric algorithm
        (`compute_boundary_matrices`).
    tol : float, optional
        Numerical tolerance for root acceptance. Default is 1e−7.

    Returns
    -------
    roots : list of tuple
        List of all valid phase-locked solutions, each as a tuple
        (z_plus, z_minus, zeta_plus, zeta_minus),
        along with printed diagnostic information about stability and cosine conditions.

    Notes
    -----
    For n = 0, the system corresponds to phase differences on edges (the simplest case);
    the lower boundary matrix B0 is empty, and only `psi_plus` variables are active.

    The function automatically builds the relevant partitions:
    all n−1 and n+1 cells are included in the "all-plus" configuration
    (`S_plus = all`, `S_minus = empty set`).

    Each identified solution is further analyzed for:
        – **stability**, using `check_stable_jacobians`, which verifies that all
          non-zero eigenvalues of J1 and J2 are negative..
        – **positivity**, using `check_cos_theta`, ensuring all cos(theta) > 0.

    Output includes:
        – the (z_plus, z_minus) winding vectors,
        – zeta_plus and zeta_minus,
        – stability and positivity,
        – full eigenvalue spectra of the Jacobians,
        – and cos(theta) values for each component.
    """
    if simplex == True:
        B = generate_all_B_by_definition(S)
    elif simplex == False:
        B = compute_boundary_matrices(S)

    if n < 0 or n >= len(S) - 1:
        raise ValueError("n must be in the range [0, len(S) - 2]")
    

    if n == 0:
        # n=0: vertices are the 0-cells, edges are (n+1)-cells
        B_n = np.zeros((0, len(S[0])))                # no lower boundary
        C_n = np.zeros((0, 0))  # shape is (0, number of vertices)
        K_n = np.zeros((0, 0))
        B_np1 = B[0]              # vertex->edge incidence
        C_np1 = compute_kernel_sp(B_np1)  
        K_np1 = np.eye(B_np1.T.shape[0])  # identity in edge space

        # Partition vertices into plus/minus (these are n=0 cells)
        S_nm1_plus, S_nm1_minus = [], []  # there is no S_{n-1}
        S_np1_plus = [i for i in range(0, len(S[1]))]
        S_np1_minus = []

    else:
        # normal case
        B_n = B[n-1]
        B_np1 = B[n]
        C_n, C_np1 = get_C_from_B(B_n, B_np1)
        K_n, K_np1 = get_K_from_B(B_n, B_np1)
        S_nm1_plus, S_nm1_minus, S_np1_plus, S_np1_minus = get_all_plus_partitions(S[n-1], S[n+1])

    print(f'n = {n},\nC_n = {C_n},\nC_np1 = {C_np1}')

    if n == 0:
        num_n_cells = len(S[0])   # vertices
    else:
        num_n_cells = B_n.shape[1]
    omega = np.zeros(num_n_cells)
    roots = []

    found_roots = find_what_works_nd(S_nm1_plus, S_nm1_minus, S_np1_plus, S_np1_minus, 
                                         B_n, B_np1, K_n, K_np1, C_n, C_np1, tol = tol, omega=omega)
        
    if found_roots:
        roots.extend(found_roots)
    else:
        print(f"No roots found for n = {n}.")
    
    #check stability of all roots, and cosine(theta) positivity
    stable_indices, all_J1s, all_J2s = check_stable_jacobians(
        roots, omega, B_n, B_np1, K_n, K_np1, C_n, C_np1,
        S_nm1_plus, S_nm1_minus, S_np1_plus, S_np1_minus
    )   
    positive_indices, all_CT1s, all_CT2s = check_cos_theta(roots, omega, B_n, B_np1, K_n, K_np1, C_n, C_np1,
        S_nm1_plus, S_nm1_minus, S_np1_plus, S_np1_minus)   
    
    #make something that prints for each root the eigenvalues of J1, J2, cos(theta) values and whether the root is stable and positive
    for idx, (z_plus, z_minus, zeta_plus, zeta_minus) in enumerate(roots):
        is_stable = idx in stable_indices
        is_positive = idx in positive_indices
        J1 = all_J1s[idx] if idx < len(all_J1s) else None
        eig_J1 = np.linalg.eigvalsh(J1) if J1 is not None else None
        J2 = all_J2s[idx] if idx < len(all_J2s) else None
        eig_J2 = np.linalg.eigvalsh(J2) if J2 is not None else None
        CT1 = all_CT1s[idx] if idx < len(all_CT1s) else None
        CT2 = all_CT2s[idx] if idx < len(all_CT2s) else None
        
        print(f"Root {idx+1}: z+ = {z_plus}, z- = {z_minus}, "
              f"zeta_plus = {zeta_plus}, zeta_minus = {zeta_minus}, "
              f"Stable: {is_stable}, Positive: {is_positive}")
        print(f"Cos(theta)_minus: {CT1} \n Eig J1:{eig_J1}\n Cos(theta)_plus: {CT2}\n Eig J2:{eig_J2}\n")

    return roots