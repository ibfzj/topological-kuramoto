# Algorithmic workflow

This document summarizes the computational workflow implemented in this repository for constructing complexes, solving the topological nonlinear Kirchhoff conditions, classifying phase-locked states, and reproducing the manuscript figures.

## 1. Input and data structures

A simplicial or cell complex is represented as a nested list
\[
S = [S_{[0]}, S_{[1]}, \dots, S_{[d]}],
\]
where `S[k]` is the list of \(k\)-cells. Each cell is represented by a list of node labels.

Examples:
- ring: nodes, edges, one polygonal face
- Platonic solid: nodes, edges, faces, one volume cell
- simplex-generated simplicial complex: one \(d\)-simplex together with all of its subsimplices

The main computational inputs are:
- the complex \(S\),
- the cochain degree \(n\),
- intrinsic frequencies \(\omega\),
- the partition sets \(S^\bullet\), \(S^\circ\),
- the coupling matrices \(K_{[n]}\), \(K_{[n+1]}\).

## 2. Construction of boundary, kernel, and coupling matrices

The low-level algebraic-topological objects are constructed in `core/complexes.py`.

### Pseudocode: boundary and kernel construction

```text
INPUT: complex S, flag simplex

if simplex = True:
    construct boundary operators B by the combinatorial simplicial formula
else:
    construct boundary operators B by geometric orientation rules

verify B_[k] B_[k+1] = 0 for all admissible k

for the target degree n:
    set B_n     = boundary operator from n-cells to (n-1)-cells
    set B_{n+1} = boundary operator from (n+1)-cells to n-cells

compute cycle matrices:
    C_n     from ker(B_n^T)
    C_{n+1} from ker(B_{n+1})

set default coupling matrices:
    K_n     = identity of appropriate size
    K_{n+1} = identity of appropriate size

OUTPUT: B_n, B_{n+1}, C_n, C_{n+1}, K_n, K_{n+1}
```

### File mapping
- `generate_all_B_by_definition`: simplicial boundary operators
- `compute_boundary_matrices`: general cell-complex boundary operators
- `compute_kernel_sp`: symbolic nullspace computation
- `get_C_from_B`: cycle matrices
- `get_K_from_B`: default identity couplings

## 3. General solver for phase-locked states

The main stationary-state solver is implemented in `core/dynamics.py`.

The code uses the ansatz
\[
\psi^{[+]} = \psi^{[+]}_{\mathrm p} + K_{[n+1]}^{-1} C_{[n+1]} \zeta^{[+]},
\qquad
\psi^{[-]} = \psi^{[-]}_{\mathrm p} + K_{[n]}^{-1} C_{[n]}^\top \zeta^{[-]},
\]
and enforces the topological nonlinear Kirchhoff conditions through a nonlinear root problem.

### Pseudocode: general stationary-state solver

```text
INPUT:
    B_n, B_{n+1}, C_n, C_{n+1}, K_n, K_{n+1},
    omega,
    partition sets S0_plus, S0_minus, S2_plus, S2_minus,
    tolerance tol

compute stationary particular solutions:
    psi_minus_sp = get_psi_minus_sp(B_n, K_n, omega)
    psi_plus_sp  = get_psi_plus_sp(B_{n+1}, K_{n+1}, omega)

enumerate candidate winding vectors:
    (z_n_candidates, z_{n+1}_candidates) = generate_z_vectors_from_C(C_n, C_{n+1})

initialize empty list ROOTS

for each z_{n+1} in z_{n+1}_candidates:
    for each z_n in z_n_candidates:
        build the appropriate nonlinear residual problem:
            - only lower-dimensional part active
            - only higher-dimensional part active
            - both parts active

        solve f(zeta) = 0 numerically using Levenberg-Marquardt

        if solver converged AND residual norm < tol AND |psi_i| <= 1:
            store the root as a phase-locked solution

OUTPUT: list ROOTS of tuples
    (z_plus, z_minus, zeta_plus, zeta_minus)
```

### Residual construction

The branch-dependent nonlinearities are encoded by:
- `f_n_minus_one`
- `f_n_plus_one`

These apply either
- \(\arcsin(\psi)\) on the all-normal branch, or
- \(\pi - \arcsin(\psi)\) on the complementary branch,

depending on the partition sets.

The residual builder `build_partitioned_root_problem` automatically constructs the correct nonlinear system in one of three modes:
- `only_n`
- `only_np1`
- `both`

## 4. Stability classification

Once candidate stationary states are found, the code reconstructs the corresponding projected angles and computes Jacobians.

### Pseudocode: stability check

```text
INPUT:
    roots, omega,
    B_n, B_{n+1}, C_n, C_{n+1}, K_n, K_{n+1},
    partition sets

compute psi_minus_sp and psi_plus_sp

for each root in roots:
    reconstruct psi_minus and psi_plus from zeta variables

    if lower-dimensional part exists:
        compute theta_minus from branch rule
        build Jacobian
            J1 = - B_n^T K_n diag(cos(theta_minus)) B_n
        test if all nonzero eigenvalues are <= 0

    if higher-dimensional part exists:
        compute theta_plus from branch rule
        build Jacobian
            J2 = - B_{n+1} K_{n+1} diag(cos(theta_plus)) B_{n+1}^T
        test if all nonzero eigenvalues are <= 0

    declare root stable only if all active Jacobians are stable

OUTPUT:
    stable root indices
    Jacobians and eigenvalue information
```

The cosine-positivity check is implemented separately in `check_cos_theta`.

## 5. Reduced workflow for all-normal solutions at \(\omega = 0\)

Many examples in the manuscript focus on the all-normal partition and \(\omega=0\). The function `find_all_plus_omega_zero_solutions` implements this reduced workflow.

### Pseudocode: all-normal, symmetric case

```text
INPUT: complex S, cochain degree n, simplex flag, tolerance tol

construct the boundary hierarchy B

if n = 0:
    use an empty lower-dimensional subsystem
    set B_n = 0, C_n = 0, K_n = 0
    set B_{n+1} from the edge-incidence matrix
else:
    set B_n and B_{n+1} from the boundary hierarchy
    compute C_n, C_{n+1}, K_n, K_{n+1}

set the all-normal partition:
    all cells belong to the plus branch
    minus branch is empty

set omega = 0

call the general solver find_what_works_nd(...)
classify all returned roots by stability and cosine positivity
print diagnostic information

OUTPUT: all all-normal phase-locked roots
```

This reduced workflow removes the combinatorial complexity associated with enumerating all partitions and is therefore the default mode for many examples.

## 6. Geometry-specific construction workflows

### 6.1 Rings

The ring case is treated in `core/rings.py` as a one-dimensional reduction of the general solver.

```text
INPUT:
    ring partition S_plus, S_minus,
    B1, K1, C1,
    optional omega and optional z-range

construct an empty higher-dimensional subsystem

compute psi_minus_sp for the ring

if a manual z-range is given:
    override the winding-number enumeration
else:
    use the general winding-number generator

for each candidate z:
    build the residual problem using the general builder
    solve the root problem numerically
    accept roots satisfying convergence, tolerance, and |psi| <= 1

OUTPUT: all ring phase-locked roots
```

The ring module also contains:
- projection-curve evaluation,
- robust bracket detection,
- scalar bifurcation tracking,
- figure-generation routines.

### 6.2 Platonic solids

The Platonic-solid construction is implemented in `core/platonic_solids.py`.

```text
for each Platonic solid:
    define node coordinates
    define polygonal faces
    orient faces consistently by breadth-first propagation over face adjacency
    extract the unique edge set from the faces
    assemble the complex as:
        S0 = vertices
        S1 = edges
        S2 = faces
        S3 = one volume cell
```

This produces complexes compatible with the general solver.

### 6.3 Simplices

The simplex construction is implemented in `core/simplexes.py`.

```text
INPUT: simplex dimension d

generate all subsets of vertices of size 1, 2, ..., d+1
group them by dimension
return the simplicial complex consisting of one d-simplex and all of its subsimplices
```

The helper `print_B_and_C_for_simplex` prints the corresponding boundary and cycle matrices for inspection and supplementary documentation.

## 7. Time-series workflow

The notebook `04_time_series.ipynb` extends the stationary-state analysis to explicit ODE simulations.

For the ring, the code:
- reconstructs edge phase differences from the reduced root variable,
- perturbs the stationary solution,
- integrates the ring ODE,
- and visualizes convergence to stable branches.

For the cube, the code:
- starts from projected winding numbers \((z^{[-]}, z^{[+]})\),
- constructs the corresponding projected target angles,
- introduces sparse integer branch-shift vectors,
- lifts the projected state to a full edge-phase equilibrium,
- verifies that the lifted state satisfies the full edge ODE,
- perturbs the equilibrium,
- and integrates the full \(n=1\) edge dynamics.

### Pseudocode: lifting from winding numbers to full edge phases

```text
INPUT: B1, B2, winding numbers z_minus, z_plus

construct projected target angles:
    theta_minus = (2 pi z_minus / N0) 1
    theta_plus  = (2 pi z_plus  / N2) 1

enumerate sparse integer branch-shift vectors m_minus, m_plus
with sums matching the winding numbers

for each pair (m_minus, m_plus):
    solve the linear lifting problem
        B1 theta   = theta_minus - 2 pi m_minus
        B2^T theta = theta_plus  - 2 pi m_plus

    wrap the resulting theta to (-pi, pi]
    compute the linear residual
    compute the ODE residual

choose the candidate with minimal combined residual

if linear and dynamical residuals are below tolerance:
    accept as a full edge-phase equilibrium
else:
    reject

OUTPUT: full edge-phase equilibrium theta_star
```

## 8. Notebook workflows

### 8.1 Rings (`01_rings.ipynb`)

```text
1. Define the six-edge ring complex
2. Compute boundary matrices B1, B2
3. Compute C1, C2 and K1, K2
4. Enumerate ring partitions
5. Solve for phase-locked roots for selected partitions
6. Generate grouped intersection plots and bifurcation diagrams
7. Sweep ring size and count stable all-normal states
```

### 8.2 Platonic solids (`02_platonic_solids.ipynb`)

```text
1. Generate the five Platonic solids
2. Print cell counts
3. Construct and inspect boundary matrices
4. Construct and inspect cycle matrices C1 and C2
5. Compute all-normal phase-locked states for n = 1
6. Count stable states for each solid
7. Generate the combined fixed-point figures
8. Sweep an inhomogeneous frequency parameter omega_0 for the cube
9. Plot the resulting control-of-multistability figure
```

### 8.3 Simplexes (`03_simplexes.ipynb`)

```text
1. Generate simplices up to the required dimension
2. Print the boundary and cycle matrices
3. Compute all-normal phase-locked states for all admissible n
4. Count the number of stable states
5. Assemble the counts into a simplex-versus-n table/heatmap
6. Verify the kernel-dimension formulas against binomial coefficients
```

### 8.4 Time series (`04_time_series.ipynb`)

```text
1. Construct stable stationary states for the ring
2. Simulate perturbed ring trajectories and plot convergence
3. Lift selected winding-number branches to full edge equilibria
4. Simulate full ring and cube edge dynamics
5. Plot representative time-series panels used in the manuscript
```

## 9. Mapping between files and functionality

- `core/complexes.py`
  - boundary operators
  - cycle matrices
  - default couplings

- `core/dynamics.py`
  - particular solutions
  - branch-dependent nonlinearities
  - winding-number enumeration
  - nonlinear root construction
  - phase-locked-state solver
  - Jacobian stability analysis

- `core/rings.py`
  - ring reduction
  - robust one-dimensional root finding
  - ring figure generation

- `core/platonic_solids.py`
  - oriented Platonic-solid generators

- `core/simplexes.py`
  - simplex generation
  - boundary/cycle inspection utilities

- notebooks
  - figure reproduction
  - parameter sweeps
  - time-series simulations
  - manuscript examples

## 10. Output of the code

Depending on the workflow, the code returns:
- phase-locked states labeled by winding vectors,
- reduced coordinates \((\zeta^{[+]}, \zeta^{[-]})\),
- stability labels,
- Jacobian eigenvalues,
- cosine-positivity diagnostics,
- equilibria for time-series simulations,
- and figure outputs used in the manuscript.