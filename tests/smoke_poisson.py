"""Solve a manufactured Poisson problem with the pinned FEniCSx checkout."""

import argparse
import subprocess
from pathlib import Path

import basix
import dolfinx
import ffcx
import numpy as np
import ufl
from dolfinx import fem, la, mesh
from mpi4py import MPI
from scipy.sparse.linalg import spsolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path, help="FFCx JIT output directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(root / "dolfinx"), "rev-parse", "HEAD"], text=True
    ).strip()
    if dolfinx.git_commit_hash != revision:
        raise RuntimeError(
            f"DOLFINx runtime commit {dolfinx.git_commit_hash} does not match {revision}"
        )

    print(f"DOLFINx {dolfinx.__version__} ({revision})", flush=True)
    print(f"Basix {basix.__version__}, UFL {ufl.__version__}, FFCx {ffcx.__version__}", flush=True)
    print(f"MPI {MPI.Get_library_version().splitlines()[0]}", flush=True)

    domain = mesh.create_unit_square(MPI.COMM_SELF, 4, 4, cell_type=mesh.CellType.quadrilateral)
    space = fem.functionspace(domain, ("Lagrange", 2))
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    x, y = ufl.SpatialCoordinate(domain)
    # The exact solution x(1-x)y(1-y) lies in the Q2 space.
    source = 2 * (x * (1 - x) + y * (1 - y))
    forms = [
        ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx,
        ufl.inner(source, test) * ufl.dx,
    ]
    a, L = fem.form(forms, jit_options={"cache_dir": args.cache_dir.resolve()})

    facet_dim = domain.topology.dim - 1
    domain.topology.create_connectivity(facet_dim, domain.topology.dim)
    boundary = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(space, facet_dim, boundary)
    bc = fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, space)

    assembled = fem.assemble_matrix(a, bcs=[bc])
    assembled.scatter_reverse()
    matrix = assembled.to_scipy()
    rhs = fem.assemble_vector(L)
    rhs.scatter_reverse(la.InsertMode.add)
    bc.set(rhs.array)
    numerical = spsolve(matrix, rhs.array)

    exact = fem.Function(space)
    exact.interpolate(lambda p: p[0] * (1 - p[0]) * p[1] * (1 - p[1]))
    error = np.linalg.norm(numerical - exact.x.array, ord=np.inf)
    residual = np.linalg.norm(matrix @ numerical - rhs.array, ord=np.inf)
    print(f"Poisson, 4x4 Q2 cells, {matrix.shape[0]} DoFs", flush=True)
    print(f"Max nodal error {error:.3e}, residual infinity norm {residual:.3e}", flush=True)
    np.testing.assert_allclose(error, 0, atol=1e-12)
    np.testing.assert_allclose(residual, 0, atol=1e-12)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
