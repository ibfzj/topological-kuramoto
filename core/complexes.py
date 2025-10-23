"""
=========================================================
Boundary, Kernel, and Coupling Matrix Construction
=========================================================

This module defines low-level algebraic–topological routines for constructing
boundary operators, kernel matrices, and coupling matrices for both
simplicial and general cell complexes.

It forms the algebraic backbone of the topological Kuramoto framework, ensuring
Bn @ Bn+1 = 0 and proper orientation across all dimensions.

Functions
---------
- generate_all_B_by_definition :
    Construct oriented boundary matrices for simplicial complexes via the
    standard alternating-sign combinatorial rule.

- compute_boundary_matrices :
    Construct boundary matrices for general (non-simplicial) cell complexes,
    including polygons and polyhedra, using geometric orientation inference.

- check_boundary_conditions :
    Verify the boundary condition (Bn @ Bn+1 = 0) for all n.

- compute_kernel_sp :
    Compute symbolic kernels (nullspaces) using SymPy for exact algebraic consistency.

- get_C_from_B :
    Compute cycle matrices Cn @ Cn+1 corresponding to the nullspaces of Bn.T and Bn+1.

- get_K_from_B :
    Generate diagonal coupling matrices Kn, Kn+1 as identity matrices corresponding to
    boundary operator dimensions.

Notes
-----
These utilities provide the algebraic topology layer shared across all higher modules
(`core.simplexes`, `core.rings`, `core.platonic_solids`).
All matrices produced satisfy the topological consistency relations required by
the dynamical solvers in `core.dynamics`.
"""

import numpy as np
import sympy as sp

from itertools import combinations, product


def generate_all_B_by_definition(S):
    """
    Construct all oriented boundary matrices for a simplicial complex
    directly from the combinatorial definition.

    Parameters
    ----------
    S : list[list[list[int]]]
        Ordered list of simplices by dimension:
        - S[0] : 0-simplices (vertices)
        - S[1] : 1-simplices (edges)
        - S[2] : 2-simplices (triangular faces)
        - ...
        Each simplex is represented by an ordered list of vertex indices.

    Returns
    -------
    B : list[np.ndarray]
        Sequence of integer-valued boundary matrices
        B = [B1, B2, …, Bn], where each Bk maps oriented k-simplices
        to their oriented (k–1)-dimensional faces.
        Columns correspond to k-simplices, rows to (k–1)-simplices.

    Notes
    -----
    The orientation of each simplex is defined by its vertex ordering.
    Signs alternate according to the standard algebraic-topological rule
    (–1)^i for the *i*-th omitted vertex.

    This routine is valid only for **simplicial complexes**, i.e. those
    whose k-cells are (k+1)-vertex simplices.  
    For more general cell complexes (e.g. cubes, dodecahedra, polygons),
    use `compute_boundary_matrices`, which infers orientation geometrically.

    The function verifies the chain condition for boundary operators to
    ensure topological consistency.

    """
    B = []

    for k in range(1, len(S)):
        S_k = S[k]
        S_k_minus_1 = S[k - 1]

        index_map = {tuple(cell): i for i, cell in enumerate(S_k_minus_1)}
        num_rows = len(S_k_minus_1)
        num_cols = len(S_k)
        B_k = np.zeros((num_rows, num_cols), dtype=int)

        for j, simplex in enumerate(S_k):
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i+1:]
                sign = (-1) ** i
                face_tuple = tuple(sorted(face))
                row = index_map.get(face_tuple)
                if row is not None:
                    B_k[row, j] = sign

        B.append(B_k)

    if check_boundary_conditions(B) == False:
        raise ValueError("Boundary conditions are not satisfied.")

    return B

def compute_boundary_matrices(S):
    """
    Construct oriented boundary matrices for a general cell complex, whose cells may have arbitrary polygonal
    or polyhedral structure, given the list of cells S = [S[0], S[1], S[2], S[3]].

    Parameters
    ----------
    S : list[list[list[int]]]
        Nested list of cells for each dimension:
        - S[0] : 0-cells (vertices)
        - S[1] : 1-cells (edges)
        - S[2] : 2-cells (faces, e.g. polygons)
        - S[3] : 3-cells (volumes)
        Each cell is represented by an ordered list of vertex indices.

    It is possible to use this function for complexes with dimensions lower than 3, i.e. input only S[0], S[1], and / or S[2].

    Returns
    -------
    boundary_matrices : list[np.ndarray]
        Sequence of integer-valued boundary matrices
        B = [B1, B2, ..., Bn], where each B_k maps oriented k-cells
        to their oriented (k–1)-dimensional boundaries.

    Notes
    -----
    For k = 1 (edges -> vertices):  the orientation is taken directly from
    the vertex ordering of each edge.  The signs (–1)ⁱ reproduce the usual
    alternating convention for 1-simplices.

    For k = 2 (faces -> edges):  each face is treated as a closed polygon.
    Its oriented boundary is traced cyclically along successive vertex pairs
    (a, b), assigning +1 if the edge orientation matches that of the face,
    and –1 otherwise.

    For k = 3 (volumes -> faces):  orientation is determined by matching each
    face of the volume against the global list of 2-cells.  A sign of +1 (–1)
     is assigned depending on whether the local orientation of the face agrees
     (or disagrees) with that of the volume.

    Unlike `generate_all_B_by_definition`, which uses the canonical alternating-sign
    formula valid only for simplices, this routine explicitly checks vertex and edge
    orderings to infer the proper boundary orientation for general polygonal cells.

    The function verifies the chain condition for boundary operators to
    ensure topological consistency.
    """
    boundary_matrices = []

    for k in range(1, len(S)):
        #create a mapping from each (k-1)-simplex to its row index
        lower_simplices = [tuple(s) for s in S[k - 1]]  # edges or faces
        lower_index = {s: i for i, s in enumerate(lower_simplices)}
        #boundary matrix B_k
        num_rows = len(S[k - 1])
        num_cols = len(S[k])
        B = np.zeros((num_rows, num_cols), dtype=int)
        #loop over all k-simplices
        for col, simplex in enumerate(S[k]):
            simplex = list(simplex)

            if k == 1: #special case: k=1. boundary of edges are vertices
                # B1: from edges to vertices
                for i in range(len(simplex)):
                    #remove vertex i from the edge to get a 0-simplex (i.e. a vertex)
                    face = simplex[:i] + simplex[i+1:]
                    t_face = tuple(face)
                    t_face_rev = tuple(reversed(face)) # try both directions

                    if t_face in lower_index: # if face found in S[0], set entry with alternating sign (-1)^i
                        B[lower_index[t_face], col] = (-1)**i
                    elif t_face_rev in lower_index: # if reversed face found, adjust sign accordingly
                        B[lower_index[t_face_rev], col] = (-1)**(i + 1)

            elif k == 2: # k = 2 
                # B2: from faces to edges
                for i in range(len(simplex)): #walk along the edges of the polygon
                    a = simplex[i]
                    b = simplex[(i + 1) % len(simplex)]  # wrap around to close the face
                    edge = (a, b)
                    edge_rev = (b, a)

                    if edge in lower_index: #if edge found with same orientation, +1
                        B[lower_index[edge], col] = 1
                    elif edge_rev in lower_index: #if reversed edge found, -1
                        B[lower_index[edge_rev], col] = -1
                    else: #skip edges that aren't part of the simplicial complex
                        pass 
                    
            elif k == 3:  # B3: from volumes to faces
                for col, cell in enumerate(S[k]):  # each 3-cell
                    volume = cell
                    for i, face in enumerate(S[k - 1]):  # look for each face in the list of known faces
                        face_set = set(face)
                        if face_set.issubset(volume):
                            face_tuple = tuple(face)
                            face_rev = tuple(reversed(face))
                            if face_tuple in lower_index:
                                B[lower_index[face_tuple], col] = 1
                            elif face_rev in lower_index:
                                B[lower_index[face_rev], col] = -1


        boundary_matrices.append(B)
    
    if check_boundary_conditions(boundary_matrices) == False:
        raise ValueError("Boundary conditions are not satisfied.")

    return boundary_matrices

def check_boundary_conditions(boundary_matrices, atol=1e-10):
    """
    Checks if all consecutive boundary matrix products are zero:
    B1 @ B2 == 0, B2 @ B3 == 0, ..., B_{n-1} @ B_n == 0
    
    Args:
        boundary_matrices: list of NumPy arrays [B1, B2, ..., Bn]
        atol: absolute tolerance for checking near-zero values
    
    Returns:
        True if all conditions are satisfied, else raises AssertionError
    """
    for i in range(len(boundary_matrices) - 1):
        B_n = boundary_matrices[i]
        B_np1 = boundary_matrices[i + 1]
        if not np.allclose(B_n @ B_np1, 0, atol=atol):
            raise AssertionError(
                f"Boundary condition failed at B[{i}] @ B[{i+1}]: "
                f"max residual = {np.abs(product).max()}"
            )
    return True


def compute_kernel_sp(A_np):
    """
    Computes the kernel (nullspace) of a given matrix A using sympy for symbolic computation.
    """
    A_sp = sp.Matrix(A_np.tolist()) # Convert the numpy array to a sympy matrix, to_list to avoid issues with transposing incorrectly
    n = A_np.shape[1] if A_np.ndim == 2 else A_np.shape[0]
    kernel = A_sp.nullspace() # Compute the kernel symbolically using sympy
    if not kernel: 
        return np.zeros((n, 0))  # Empty kernel as a (n, 0) zero matrix
    # Convert each vector in the nullspace to a NumPy array and stack them
    kernel_numpy = np.column_stack([
    np.array(vec).astype(np.float64).flatten()
    for vec in kernel
    ])
    return kernel_numpy

def get_C_from_B(B_n, B_np1):
    """
    Computes the kernel matrices C_n and C_np1 from the B matrices.
    Returns:
        C_n: C_n.T is the kernel matrix for B_n.T
        C_np1: Kernel matrix for C_np1
    """
    C_nT = compute_kernel_sp(B_n.T)
    C_n = C_nT.T

    C_np1 = compute_kernel_sp(B_np1)

    # Assert that B_n.T @ C_n.T is zero
    assert np.allclose(B_n.T @ C_n.T, 0), "B_n.T @ C_n.T is not zero"

    # Assert that B_np1 @ C_np1 is zero
    assert np.allclose(B_np1 @ C_np1, 0), "B_np1 @ C_np1 is not zero"
    
    return C_n, C_np1

def get_K_from_B(B_n, B_np1):
    """
    Computes the diagonal coupling constant matrices K_n and K_np1 from the B matrices, assuming that all elements are one!
    If you want to use different coupling constants, you should define K_n and K_np1 manually.
    Returns:
        K_n: Coupling matrix for n
        K_np1: Coupling matrix for n+1
    """
    K_n = np.eye(np.shape(B_n)[0])
    K_np1 = np.eye(np.shape(B_np1.T)[0])
    
    return K_n, K_np1
