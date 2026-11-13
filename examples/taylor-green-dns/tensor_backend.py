"""Tensor contraction backend for the uniform hexahedral Q2 momentum form."""

from pathlib import Path

import basix
import cupy as cp
import numpy as np


def _tensor_indices(points, width):
    indices = np.rint(np.asarray(points) * (width - 1)).astype(np.int32)
    return np.ravel_multi_index(indices.T, (width,) * 3)


def tensor_data(element, geometry, cell_width):
    """Validate the uniform mesh and factor the original basis and quadrature."""
    interval = basix.CellType.interval
    hexahedron = basix.CellType.hexahedron
    q1, w1 = basix.make_quadrature(interval, 6)
    q3, w3 = basix.make_quadrature(hexahedron, 6)
    nq = len(q1)
    if nq != 4 or len(q3) != nq**3:
        raise ValueError("The tensor backend requires the original 4 x 4 x 4 quadrature.")
    order = np.argsort(q1[:, 0])
    q1, w1 = q1[order, 0], w1[order]
    q_indices = np.argmin(np.abs(q3[:, :, None] - q1), axis=2)
    np.testing.assert_allclose(q3, q1[q_indices], rtol=0, atol=1e-14)
    q_ids = np.ravel_multi_index(q_indices.T, (nq,) * 3)
    if np.unique(q_ids).size != nq**3:
        raise ValueError("The original quadrature does not cover the tensor grid.")
    weights = np.multiply.outer(np.multiply.outer(w1, w1), w1).reshape(-1)
    np.testing.assert_allclose(w3, weights[q_ids], rtol=1e-13, atol=1e-15)

    hex_element = basix.finite_element.FiniteElement(element)
    dof_ids = _tensor_indices(hex_element.points, 3)
    if hex_element.dim != 27 or np.unique(dof_ids).size != 27:
        raise ValueError("The tensor backend requires nodal Q2 hexahedra.")
    dof_order = np.argsort(dof_ids)
    one_dimensional = basix.create_element(
        basix.ElementFamily.P, interval, 2, hex_element.lagrange_variant
    )
    one_order = np.argsort(one_dimensional.points[:, 0])
    b, d = one_dimensional.tabulate(1, q1[:, None])[:, :, one_order, 0]
    reference = hex_element.tabulate(1, q3)[:, :, dof_order, 0]
    for axis in (None, 0, 1, 2):
        factors = [(d if direction == axis else b)[q_indices[:, direction]]
                   for direction in range(3)]
        tensor = np.einsum("qi,qj,qk->qijk", *factors).reshape(-1, 27)
        row = 0 if axis is None else basix.index(*(int(j == axis) for j in range(3)))
        np.testing.assert_allclose(tensor, reference[row], rtol=0, atol=2e-13)

    reference_geometry = basix.cell.geometry(hexahedron)
    expected = cell_width * (reference_geometry - reference_geometry[0])
    for axis in range(3):
        offsets = geometry[:, :, axis] - geometry[:, :1, axis]
        np.testing.assert_allclose(
            offsets, np.broadcast_to(expected[:, axis], offsets.shape), rtol=0, atol=2e-13
        )
    return dof_order, b, d, weights * cell_width**3


class TensorMomentum:
    """Evaluate one element per CUDA block using cached tensor basis tables."""

    threads = 128

    def __init__(self, data, cell_width):
        _, basis, derivative, weights = data

        def initializer(values):
            if values.ndim == 1:
                return "{" + ", ".join(format(value, ".17g") for value in values) + "}"
            return "{" + ", ".join(initializer(row) for row in values) + "}"

        constants = (
            f"__constant__ double basis[4][3] = {initializer(basis)};\n"
            f"__constant__ double derivative[4][3] = {initializer(derivative)};\n"
            f"__constant__ double weighted_volume[64] = {initializer(weights)};\n"
            f"constexpr double inverse_width = {1.0 / cell_width:.17g};\n"
        )
        source = constants + Path(__file__).with_name("tensor_momentum.cu").read_text(
            encoding="utf-8"
        )
        self.kernel = cp.RawModule(code=source, options=("--std=c++17", "--fmad=true")).get_function(
            "assemble_momentum"
        )

    def __call__(self, cells, dofs, geometry, velocity, viscosity, residual):
        self.kernel(
            (cells,), (self.threads,),
            (np.int32(cells), dofs, geometry, velocity, np.float64(viscosity), residual),
        )
