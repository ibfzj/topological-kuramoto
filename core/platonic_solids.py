"""
=========================================================
Platonic Solid Generators for Topological Cell Complexes
=========================================================

This module provides helper routines for constructing oriented cell
complexes corresponding to the five Platonic solids (tetrahedron, cube,
octahedron, dodecahedron, icosahedron). The data is returned in a form
compatible with the Kuramoto dynamics pipeline (`find_all_plus_omega_zero_solutions`).

Functions
---------
- extract_edges_from_faces : Build unique edges from polygonal faces.
- build_face_adjacency : Compute face adjacency via shared edges.
- edges_direction_in_face : Determine local edge orientation in a face.
- orient_faces_adjacency : Propagate orientation consistency across faces.
- get_platonic_solids_full_data : Generate all five oriented Platonic solids.

"""

import numpy as np
from collections import defaultdict, deque


def extract_edges_from_faces(faces):
    """
    Extract the unique set of edges from a list of polygonal faces.

    Each face is given as a cyclic list of vertex indices.
    The function enumerates all consecutive vertex pairs (including the closing edge),
    stores them as sorted tuples (u, v), and removes duplicates.

    Parameters
    ----------
    faces : list of list[int]
        List of polygonal faces, each represented by its ordered vertex indices.

    Returns
    -------
    edges : list of list[int]
        Sorted list of unique undirected edges present in the complex.

    Notes
    -----
    Edge orientation is ignored (stored as sorted vertex pairs).
    The output is suitable for building the 1-skeleton or incidence matrix B1.
    """

    edge_set = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            u, v = face[i], face[(i + 1) % n]
            edge = tuple(sorted((u, v)))
            edge_set.add(edge)
    return sorted([list(e) for e in edge_set])


def build_face_adjacency(faces):
    """
    Construct the adjacency relation between faces via shared edges.

    Each edge shared by exactly two faces defines an adjacency link.
    The function returns a mapping from each face index to its neighboring
    faces and their common edges.

    Parameters
    ----------
    faces : list of list[int]
        List of polygonal faces (each as an ordered list of vertex indices).

    Returns
    -------
    adjacency : dict[int, list[tuple[int, tuple[int,int]]]]
        Dictionary mapping each face index to a list of tuples
        (neighbor_index, shared_edge_vertices).

    Notes
    -----
    Faces sharing an edge are considered adjacent only if that edge
    occurs exactly twice in the complex.
    This structure supports propagation of orientation across faces.
    """
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            u, v = face[i], face[(i + 1) % n]
            edge_to_faces[tuple(sorted((u, v)))].append(fi)
    adjacency = defaultdict(list)
    for edge, f_list in edge_to_faces.items():
        if len(f_list) == 2:
            f1, f2 = f_list
            adjacency[f1].append((f2, edge))
            adjacency[f2].append((f1, edge))
    return adjacency


def edges_direction_in_face(face):
    """
    Compute the orientation of each edge within a single face.

    For each edge (u, v) in the ordered vertex list of the face,
    the direction is assigned +1 if (u < v) matches the traversal order,
    and −1 otherwise. The edge is stored as a sorted tuple (min, max).

    Parameters
    ----------
    face : list[int]
        Ordered list of vertex indices representing one face.

    Returns
    -------
    dir_map : dict[tuple[int,int], int]
        Mapping from edge (min, max) → orientation (+1 or −1).

    Notes
    -----
    Used internally to compare local edge orientations between neighboring faces.
    Assumes the face vertex order is consistent (counterclockwise or clockwise).
    """
    n = len(face)
    dir_map = {}
    for i in range(n):
        u, v = face[i], face[(i + 1) % n]
        edge = tuple(sorted((u, v)))
        # Direction is +1 if edge is (u->v) matches (min->max), else -1
        dir_map[edge] = 1 if (u < v) else -1
    return dir_map

def orient_faces_adjacency(faces, vertices):
    """
    Orient all faces of a polyhedral surface consistently.

    Starting from one arbitrarily oriented face, this function propagates
    orientation to all neighboring faces so that shared edges have opposite
    traversal directions (ensuring coherent global orientation).

    Parameters
    ----------
    faces : list of list[int]
        List of polygonal faces (each as a list of vertex indices).
    vertices : list[list[float]]
        Vertex coordinates (only used to define total count; geometry unused).

    Returns
    -------
    oriented : list[list[int]]
        List of faces with vertex orders adjusted for consistent orientation.

    Notes
    -----
    Uses breadth-first traversal of the face adjacency graph.
    Ensures each face's orientation is compatible with its neighbors
    (shared edges are traversed in opposite directions).
    """
    adjacency = build_face_adjacency(faces)
    oriented = [None] * len(faces)
    oriented[0] = faces[0]  # seed orientation with first face

    queue = deque([0])
    while queue:
        f_idx = queue.popleft()
        current_face = oriented[f_idx]
        current_dir = edges_direction_in_face(current_face)

        for neighbor_idx, shared_edge in adjacency[f_idx]:
            if oriented[neighbor_idx] is not None:
                continue

            neighbor_face = faces[neighbor_idx]
            neighbor_dir = edges_direction_in_face(neighbor_face)

            # If shared edge direction same in both faces, flip neighbor face
            if current_dir[tuple(sorted(shared_edge))] == neighbor_dir[tuple(sorted(shared_edge))]:
                oriented[neighbor_idx] = list(reversed(neighbor_face))
            else:
                oriented[neighbor_idx] = neighbor_face

            queue.append(neighbor_idx)

    return oriented

def get_platonic_solids():
    """
    Construct the complete topological data for all five Platonic solids.

    Each solid is returned as a tuple (S0, S1, S2, S3), where:
        S0 — list of vertices,
        S1 — list of edges (2-cells),
        S2 — list of faces (3-cells),
        S3 — list containing the single 3D volume (4-cell equivalent).

    The orientation of all faces is made globally consistent using
    `orient_faces_adjacency`.

    Returns
    -------
    solids : dict[str, tuple[list, list, list, list]]
        Dictionary mapping solid name → (vertices, edges, faces, volume).

    All 5 Platonic Solids included: 'tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron'

    Notes
    -----
    Vertex numbering starts at 1.
    """
    solids = {}

    # Tetrahedron
    tetra_vertices = [
        [1, 1, 1], [-1, -1, 1],
        [-1, 1, -1], [1, -1, -1]
    ]
    tetra_faces = [
        [1, 2, 3], [1, 4, 2],
        [3, 2, 4], [1, 3, 4]
    ]
    tetra_faces = orient_faces_adjacency(tetra_faces, tetra_vertices)
    tetra_edges = extract_edges_from_faces(tetra_faces)
    tetra_volume = [[1, 2, 3, 4]]
    solids['tetrahedron'] = (
        [[i + 1] for i in range(len(tetra_vertices))],
        tetra_edges,
        tetra_faces,
        tetra_volume
    )

    # Cube
    cube_vertices = [
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]
    ]
    cube_faces = [
        [1, 2, 3, 4], [5, 6, 7, 8],
        [1, 5, 6, 2], [4, 3, 7, 8],
        [1, 4, 8, 5], [2, 6, 7, 3]
    ]
    cube_faces = orient_faces_adjacency(cube_faces, cube_vertices)
    cube_edges = extract_edges_from_faces(cube_faces)
    cube_volume = [[1, 2, 3, 4, 5, 6, 7, 8]]
    solids['cube'] = (
        [[i + 1] for i in range(len(cube_vertices))],
        cube_edges,
        cube_faces,
        cube_volume
    )

    # Octahedron
    octa_vertices = [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ]
    octa_faces = [
        [1, 3, 5], [3, 2, 5], [2, 4, 5], [4, 1, 5],
        [3, 1, 6], [2, 3, 6], [4, 2, 6], [1, 4, 6]
    ]
    octa_faces = orient_faces_adjacency(octa_faces, octa_vertices)
    octa_edges = extract_edges_from_faces(octa_faces)
    octa_volume = [[1, 2, 3, 4, 5, 6]]
    solids['octahedron'] = (
        [[i + 1] for i in range(len(octa_vertices))],
        octa_edges,
        octa_faces,
        octa_volume
    )

    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    # Dodecahedron
    phi_inv = 1 / phi
    dodeca_vertices = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi_inv, phi], [0, -phi_inv, phi], [0, phi_inv, -phi], [0, -phi_inv, -phi],
        [phi_inv, phi, 0], [-phi_inv, phi, 0], [phi_inv, -phi, 0], [-phi_inv, -phi, 0],
        [phi, 0, phi_inv], [-phi, 0, phi_inv], [phi, 0, -phi_inv], [-phi, 0, -phi_inv]
    ]
    dodeca_faces = [
        [1, 2, 3, 4, 5],
        [1, 6, 11, 12, 2],
        [2, 12, 17, 18, 3],
        [3, 18, 19, 7, 4],
        [4, 7, 8, 9, 5],
        [1, 5, 9, 10, 6],
        [6, 10, 14, 15, 11],
        [12, 11, 15, 16, 17],
        [18, 17, 16, 20, 19],
        [7, 19, 20, 13, 8],
        [9, 8, 13, 14, 10],
        [20, 16, 15, 14, 13]
    ]
    dodeca_faces = orient_faces_adjacency(dodeca_faces, dodeca_vertices)
    dodeca_edges = extract_edges_from_faces(dodeca_faces)
    dodeca_volume = [[i + 1 for i in range(len(dodeca_vertices))]]
    solids['dodecahedron'] = (
        [[i + 1] for i in range(len(dodeca_vertices))],
        dodeca_edges,
        dodeca_faces,
        dodeca_volume
    )

    # Icosahedron
    icosa_vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    icosa_faces = [
        [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 2],
        [2, 7, 3], [3, 8, 4], [4, 9, 5], [5, 10, 6], [6, 11, 2],
        [7, 8, 3], [8, 9, 4], [9, 10, 5], [10, 11, 6], [11, 7, 2],
        [7, 12, 8], [8, 12, 9], [9, 12, 10], [10, 12, 11], [11, 12, 7]
    ]
    icosa_faces = orient_faces_adjacency(icosa_faces, icosa_vertices)
    icosa_edges = extract_edges_from_faces(icosa_faces)
    icosa_volume = [[i + 1 for i in range(len(icosa_vertices))]]
    solids['icosahedron'] = (
        [[i + 1] for i in range(len(icosa_vertices))],
        icosa_edges,
        icosa_faces,
        icosa_volume
    )

    return solids
