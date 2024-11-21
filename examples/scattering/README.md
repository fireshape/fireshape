# Inverse Acoustic Scattering Problem

This directory contains the code generating the numerical results in Chapter 5 of the thesis "Sensitivity-Guided Shape Reconstruction". The implementation is adapted from [scattering](https://github.com/gninr/scattering) for acoustic scattering.

## File description

- `scattering_PDEconstraint.py`

    This module implements the solver for the direct scattering problem that acts as PDE constraint.

- `scattering_objective.py`

    This module implements the generator for the target far field data $u_\infty$ and the evaluation of the misfit function $\mathcal{J}$.

- `scattering_main.py`

    This script sets up the problem.

- `utils.py`
    
    The function `generate_mesh` uses [Gmsh](https://gmsh.info/) to create mesh that is compatible with the PML method and far field pattern evaluation. Currently supported shapes are `"circle"`, `"kite"` and `"square"`. The other functions are implemented for various visualization purposes.

## Usage

1. Configuration

    In `scattering_main.py`:

    - Parameters of the mesh for the initial guess:
        - `a0`, `a1`, `b0`, `b1` define the size $a, a^\ast, b, b^\ast$ of the absorbing layer for the PML method.
        - `R0`, `R1` define the radius $R_0, R_1$ of the cut-off function for evaluating the far field pattern.
        - `obstacle` defines the shape of the initial guess $\Omega^0$ with parameters `"shape"`, `"shift"`, `"scale"` and `"nodes"` (for kite shape).
        - `refine` defines the refinement level of the mesh.

    - Parameters of the wavelet space
        - `bbox` defines the hold-all domain $D$.
        - `primal_orders`, `dual_orders` define the orders $d, \tilde{d}$ of the biorthogonal B-spline wavelets.
        - `levels` defines the refinement level $j$ of the wavelet basis.
        - The shape derivatives $\mathbf{dJ}$ are used as the coeffcient vector of the shape gradient if `norm_equiv = True`; otherwise the classical shape gradient $\mathbf{\nabla J}$ computed by solving the linear system is used.
        - `tol` defines the thresold $\varepsilon$ for selecting basis functions.
        - `inner` defines the inner product endowed with the control space.

    - Parameters of the incident plane waves
        - `k` defines the wavenumber of incident waves.
        - `dirs` defines the directions of incident waves.

    In `scattering_objective.py`:
    - Parameters of the mesh for the target domain (in function `target_far_field`):
        - `a0`, `a1`, `b0`, `b1` define the size $a, a^\ast, b, b^\ast$ of the absorbing layer for the PML method.
        - `R0`, `R1` define the radius $R_0, R_1$ of the cut-off function for evaluating the far field pattern.
        - `obstacle` defines the shape of the target domain $\Omega^\ast$ with parameters `"shape"`, `"shift"`, `"scale"` and `"nodes"` (for kite shape).
        - `refine` defines the refinement level of the mesh.
    - Parameters of the target far field (in function `__init__`):
        - `n` defines the number of measurement points $n_x$ for each incident wave.

2. Execution

    Type the following command in terminal:

        python3 scattering_main.py
    
3. Visualization

    Open the file `u.pvd` with [ParaView](https://www.paraview.org/).