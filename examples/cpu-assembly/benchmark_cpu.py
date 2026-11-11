"""Measure fixed-discretization FEniCSx assembly paths, not full CFD solves."""

import argparse
import cProfile
import gc
import io
import json
import os
import platform
import pstats
import statistics
import subprocess
import time
from pathlib import Path

import basix
import dolfinx
import ffcx
import numpy as np
import ufl
from dolfinx import fem, la, mesh
from mpi4py import MPI


def summary(samples):
    return {
        "median_ms": 1000 * statistics.median(samples),
        "min_ms": 1000 * min(samples),
        "max_ms": 1000 * max(samples),
        "samples_ms": [1000 * s for s in samples],
    }


def measure(fn, repeat=9, inner=1):
    for _ in range(2):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - start) / inner)
    return summary(samples)


def run_case(cell, n, degree, out):
    tag = f"{cell}_n{n}_p{degree}"
    print(f"START {tag}", flush=True)
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, n, n, n, cell_type=getattr(mesh.CellType, cell)
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    W = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    velocity = fem.Function(W)
    velocity.interpolate(
        lambda x: np.vstack((1.0 + 0.2 * x[1], 0.5 + 0.1 * x[2], 0.25 + 0.1 * x[0]))
    )
    velocity_base = velocity.x.array.copy()
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt_inverse = fem.Constant(domain, dolfinx.default_scalar_type(10.0))
    nu = fem.Constant(domain, dolfinx.default_scalar_type(0.01))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree + 2})
    static = (dt_inverse * u * v + nu * ufl.inner(ufl.grad(u), ufl.grad(v))) * dx
    changing = ufl.dot(velocity, ufl.grad(u)) * v * dx
    complete = static + changing
    input_function = fem.Function(V)
    input_function.x.array[:] = np.random.default_rng(71).normal(size=input_function.x.array.size)
    action = ufl.action(complete, input_function)
    symbolic = [complete, static, changing, action]
    start = time.perf_counter()
    forms = fem.form(symbolic, jit_options={"cache_dir": out / "jit", "timeout": 120})
    cold_seconds = time.perf_counter() - start
    warm = measure(lambda: fem.form(symbolic, jit_options={"cache_dir": out / "jit"}), repeat=5)
    a, a_static, a_changing, a_action = forms
    constants = fem.pack_constants(a)
    coeffs = fem.pack_coefficients(a)
    A = fem.assemble_matrix(a)
    A.scatter_reverse()
    reference = A.data.copy()
    M = A.to_scipy()
    expected_action = M @ input_function.x.array
    output_vector = fem.assemble_vector(a_action)
    output_vector.scatter_reverse(la.InsertMode.add)
    action_error = np.linalg.norm(output_vector.array - expected_action, np.inf)
    action_scale = max(np.linalg.norm(expected_action, np.inf), 1.0)
    if action_error > 1e-11 * action_scale:
        raise AssertionError(f"Action mismatch {action_error} vs scale {action_scale}")

    def new_matrix():
        B = fem.assemble_matrix(a)
        B.scatter_reverse()
        return B

    def reused_matrix():
        A.data[:] = 0
        fem.assemble_matrix(A, a)
        A.scatter_reverse()

    def reused_prepacked():
        A.data[:] = 0
        fem.assemble_matrix(A, a, constants=constants, coeffs=coeffs)
        A.scatter_reverse()

    # Interleave the alternatives to reduce ordering and temperature bias.
    alternatives = {
        "new_matrix_and_assemble": new_matrix,
        "reuse_matrix_auto_pack": reused_matrix,
        "reuse_matrix_prepacked_unchanged": reused_prepacked,
    }
    for fn in alternatives.values():
        fn()
        fn()
    samples = {name: [] for name in alternatives}
    rng = np.random.default_rng(42)
    for _ in range(9):
        for name in rng.permutation(list(alternatives)):
            start = time.perf_counter()
            alternatives[name]()
            samples[name].append(time.perf_counter() - start)
    timings = {name: summary(values) for name, values in samples.items()}
    reused_prepacked()
    np.testing.assert_allclose(A.data, reference, rtol=1e-12, atol=1e-13)
    timings["pack_coefficients"] = measure(lambda: fem.pack_coefficients(a))
    timings["pack_constants"] = measure(lambda: fem.pack_constants(a), inner=100)
    timings["create_sparsity_and_matrix"] = measure(lambda: fem.create_matrix(a), repeat=5)
    timings["csr_matvec"] = measure(lambda: M @ input_function.x.array, inner=10)

    def matrix_free():
        output_vector.array[:] = 0.0
        fem.assemble_vector(output_vector.array, a_action)
        output_vector.scatter_reverse(la.InsertMode.add)

    timings["matrix_free_action_auto_pack"] = measure(matrix_free)
    S = fem.assemble_matrix(a_static)
    S.scatter_reverse()
    C = fem.create_matrix(a_changing)
    for other in (S, C):
        np.testing.assert_array_equal(A.indices, other.indices)
        np.testing.assert_array_equal(A.indptr, other.indptr)
    changed_constants = fem.pack_constants(a_changing)
    merged = np.empty_like(A.data)

    def full_dynamic(step):
        velocity.x.array[:] = velocity_base * (1.0 + 0.03 * np.sin(step))
        reused_matrix()

    def split_dynamic(step):
        velocity.x.array[:] = velocity_base * (1.0 + 0.03 * np.sin(step))
        C.data[:] = 0.0
        fem.assemble_matrix(
            C, a_changing, constants=changed_constants, coeffs=fem.pack_coefficients(a_changing)
        )
        C.scatter_reverse()
        np.add(S.data, C.data, out=merged)

    full_dynamic(1)
    split_dynamic(1)
    split_error = float(np.max(np.abs(A.data - merged)))
    np.testing.assert_allclose(A.data, merged, rtol=1e-11, atol=1e-12)
    dynamic_samples = {"full_dynamic": [], "split_constant_plus_dynamic": []}
    dynamic_fns = {"full_dynamic": full_dynamic, "split_constant_plus_dynamic": split_dynamic}
    for step in range(2, 11):
        for name in rng.permutation(list(dynamic_fns)):
            start = time.perf_counter()
            dynamic_fns[name](step)
            dynamic_samples[name].append(time.perf_counter() - start)
        np.testing.assert_allclose(A.data, merged, rtol=1e-11, atol=1e-12)
    timings.update({name: summary(values) for name, values in dynamic_samples.items()})
    # This profile identifies Python/native-call time for this assembly path only.
    profiler = cProfile.Profile()
    profiler.enable()
    for step in range(5):
        full_dynamic(step)
    profiler.disable()
    profile_text = io.StringIO()
    pstats.Stats(profiler, stream=profile_text).sort_stats("tottime").print_stats(18)
    (out / f"{tag}.profile.txt").write_text(profile_text.getvalue(), encoding="utf-8")
    profiler.dump_stats(out / f"{tag}.pstats")
    result = {
        "case": tag,
        "operator": "scalar transient convection-diffusion velocity-component proxy",
        "full_cfd_solve": False,
        "dtype": "float64",
        "quadrature_degree": 2 * degree + 2,
        "cells": domain.topology.index_map(3).size_global,
        "dofs": V.dofmap.index_map.size_global,
        "nnz": len(A.data),
        "csr_bytes": A.data.nbytes + A.indices.nbytes + A.indptr.nbytes,
        "packed_coefficient_bytes": sum(x.nbytes for x in coeffs.values()),
        "first_form_setup_seconds": cold_seconds,
        "jit_cache_policy": "Shared cache across cases; the second hexahedral case reuses kernels",
        "warm_form_creation_all_four": warm,
        "matrix_free_max_abs_error": float(action_error),
        "split_matrix_max_abs_error": split_error,
        "timings": timings,
        "boundary_conditions": "No BCs in this operator microbenchmark",
        "profile": str(out / f"{tag}.profile.txt"),
    }
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "-C", str(source / "dolfinx"), "rev-parse", "HEAD"], text=True).strip()
    assert sha == dolfinx.git_commit_hash
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dolfinx": dolfinx.__version__,
        "dolfinx_commit": sha,
        "basix": basix.__version__,
        "ufl": ufl.__version__,
        "ffcx": ffcx.__version__,
        "has_petsc": dolfinx.has_petsc,
        "ranks": 1,
        "blas_threads": os.getenv("OPENBLAS_NUM_THREADS"),
        "ffcx_options": ffcx.get_options(),
    }
    print(json.dumps(metadata, indent=2), flush=True)
    cases = [("hexahedron", 12, 2), ("hexahedron", 24, 2), ("tetrahedron", 16, 2)]
    results = [run_case(*case, args.out) for case in cases]
    (args.out / "results.json").write_text(json.dumps({"metadata": metadata, "cases": results}, indent=2), encoding="utf-8")
    print("PASS all operator equivalence checks", flush=True)


if __name__ == "__main__":
    main()
