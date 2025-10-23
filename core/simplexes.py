"""
=========================================================
Simplex Construction and Boundary–Cycle Matrix Utilities
=========================================================

This module provides functions for generating oriented simplicial complexes
and for inspecting their associated boundary and kernel matrices.
It implements the purely combinatorial construction of simplices and their
incidence relationships.

Functions
---------
- generate_simplices :
    Construct all simplices (vertices, edges, faces, etc.) up to a given
    dimension n, for an (n+1)-simplex.

- print_B_and_C_for_simplex :
    Generate all boundary matrices B[n] and corresponding kernel matrices C[n]
    for simplices up to a chosen dimension, printing them in formatted LaTeX
    tables suitable for supplementary documentation.
"""


import numpy as np
from core.complexes import (
    generate_all_B_by_definition,
    get_C_from_B,
)
from itertools import combinations


def generate_simplices(n):
    """
    Generate all simplices up to dimension n of the (n+1)-cell.
    Each vertex is numbered from 1 to (n+2), so we have a (n+1)-simplex.
    Returns a list of simplices or sets: S[0], S[1], ..., S[n]
    """
    vertices = list(range(1, n + 2))  # n+1 vertices
    S = []
    for k in range(n + 1):
        S_k = [list(sorted(c)) for c in combinations(vertices, k + 1)]
        S.append(S_k)
    return S

def print_B_and_C_for_simplex(max_dim=5):
    """
    Generate all simplices up to `max_dim` and print their
    boundary matrices B[n] and cycle matrices C[n].

    For each n:
        B[n] : boundary matrix mapping n-simplices -> (n-1)-simplices
        C[n], C[n+1] : kernel (cycle) matrices satisfying B[n].T @ C[n].T = 0 and B[n+1] @ C[n+1] = 0

    The output is formatted for LaTeX/SI-style supplementary tables.
    """
    S = generate_simplices(max_dim)
    B = generate_all_B_by_definition(S)

    print(f"\n=== Simplices up to {max_dim}-simplex ===")
    for i, s in enumerate(S):
        print(f"S[{i}] ({len(s)} elements): {s}")
    print("")

    # Iterate over all n-levels
    for n in range(len(B)):
        B_n = B[n]
        print(f"\n--- Boundary matrix  B[{n+1}]  (maps from S[{n+1}] to S[{n}]) ---")
        print(B_n)
        print(f"Shape: {B_n.shape}")

        # Compute kernels (C matrices)
        if n < len(B) - 1:
            C_n, C_np1 = get_C_from_B(B_n, B[n + 1])
            print(f"\nCycle matrices for n = {n}")
            print(f"C[{n}] (from B[{n}].T @ C[{n}].T = 0):")
            print(C_n)
            print(f"Shape: {C_n.shape}")

            print(f"\nC[{n+1}] (from B[{n+1}] @ C[{n+1}] = 0):")
            print(C_np1)
            print(f"Shape: {C_np1.shape}")

        print("-" * 80)
