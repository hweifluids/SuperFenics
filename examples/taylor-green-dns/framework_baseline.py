"""Run the periodic incompressible solver on the 3D Taylor-Green vortex."""

import argparse
import csv
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

from mpi4py import MPI

import numpy as np

import dolfinx
import ffcx
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import periodic

LENGTH = 2 * math.pi
VOLUME = LENGTH**3


def initial_velocity(x):
    """Evaluate the analytic Taylor-Green initial condition."""
    sx, sy, sz = np.sin(x)
    cx, cy, cz = np.cos(x)
    return np.array((sx * cy * cz, -cx * sy * cz, np.zeros_like(sz)))


def diagnostics(flow, velocity, step, dt, cells, viscosity, xp):
    """Measure the periodic discrete flow invariants and resolution scales."""
    parts = velocity if isinstance(velocity, tuple) else (velocity,)
    masses = flow.mass if isinstance(flow.mass, tuple) else (flow.mass,)
    residual = flow.residual(velocity)
    loads = residual if isinstance(residual, tuple) else (residual,)
    energy = spanwise_energy = dissipation = maximum = 0.0
    for mass, values, load in zip(masses, parts, loads, strict=True):
        with values.device if xp is not np else nullcontext():
            weighted_square = mass * values**2
            energy += float(xp.sum(weighted_square))
            spanwise_energy += float(xp.sum(weighted_square[2::3]))
            dissipation += float(xp.dot(values, load))
            maximum = max(maximum, float(xp.max(xp.sum(xp.abs(values.reshape(-1, 3)), axis=1))))
    dissipation /= VOLUME
    divergence = flow.weak_divergence(velocity) * (cells**3 / VOLUME)
    result = {
        "step": step,
        "time": step * dt,
        "kinetic_energy": energy / (2 * VOLUME),
        "spanwise_kinetic_energy": spanwise_energy / (2 * VOLUME),
        "viscous_dissipation": dissipation,
        "weak_divergence_rms": float(xp.sqrt(xp.mean(divergence**2))),
        "advective_cfl": maximum * dt * (2 * cells) / LENGTH,
    }
    if (
        not all(map(math.isfinite, result.values()))
        or result["kinetic_energy"] <= 0
        or dissipation <= 0
        or result["weak_divergence_rms"] > 1e-10
        or result["advective_cfl"] > 0.8
    ):
        raise RuntimeError(f"Invalid flow at step {step}: {result}")
    result["kolmogorov_length"] = (viscosity**3 / dissipation) ** 0.25
    return result


def verify_assembly(flow, form, state):
    """Compare a periodic residual against DOLFINx CPU assembly."""
    probe = np.random.default_rng(29).standard_normal(flow.velocity_dofs.size)
    state.x.array[:] = flow.velocity_dofs.expand(probe)
    expected = flow.velocity_dofs.accumulate(fem.assemble_vector(form).array)
    actual = flow.gather(flow.residual(flow.distribute(probe)))
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12 * scale)
    return float(np.max(np.abs(actual - expected)) / scale)


def main(args, xp, device):
    """Build the forms, advance the flow, and record the baseline."""
    args.out.mkdir(parents=True, exist_ok=True)
    n = args.cells
    lower, upper = (0.0,) * 3, (LENGTH,) * 3
    jit = {"cache_dir": args.out / "jit", "timeout": 180}
    viscosity = 1 / args.reynolds
    print(
        f"Preparing {n}^3 HEX cells, Re={args.reynolds:g}, dt={args.dt:g}, backend={args.backend}.",
        flush=True,
    )
    setup_start = time.perf_counter()
    domain = mesh.create_box(
        MPI.COMM_SELF,
        (lower, upper),
        (n,) * 3,
        cell_type=mesh.CellType.hexahedron,
        dtype=np.float64,
    )
    velocity_space = fem.functionspace(domain, ("Lagrange", 2, (3,)))
    pressure_space = fem.functionspace(domain, ("Lagrange", 1))
    state = fem.Function(velocity_space, dtype=np.float64)
    test = ufl.TestFunction(velocity_space)
    nu = fem.Constant(domain, np.float64(viscosity))
    convection = 0.5 * (
        ufl.inner(ufl.dot(ufl.grad(state), state), test)
        - ufl.inner(ufl.dot(ufl.grad(test), state), state)
    )
    diffusion = nu * ufl.inner(ufl.grad(state), ufl.grad(test))
    load = fem.form(
        (convection + diffusion) * ufl.dx(metadata={"quadrature_degree": 6}),
        dtype=np.float64,
        jit_options=jit,
    )
    flow = periodic.PeriodicIncompressibleProblem(
        load,
        state,
        pressure_space,
        shape=(n,) * 3,
        lower=lower,
        upper=upper,
        backend=args.backend,
        cuda_devices=args.cuda_devices,
        jit_options=jit,
    )

    assembly_error = verify_assembly(flow, load, state) if args.backend == "cuda" else None
    state.interpolate(initial_velocity)
    velocity = flow.project(flow.distribute(flow.velocity_dofs.restrict(state.x.array)))
    first = diagnostics(flow, velocity, 0, args.dt, n, viscosity, xp)
    np.testing.assert_allclose(first["kinetic_energy"], 0.125, rtol=0, atol=1e-9)
    setup_seconds = time.perf_counter() - setup_start
    print(f"Setup complete in {setup_seconds:.3f} s, assembly error={assembly_error}.", flush=True)

    rows, timings = [first], []
    loop_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        velocity = flow.step(velocity, args.dt)
        flow.synchronize()
        elapsed = time.perf_counter() - start
        if step > args.warmup:
            timings.append(elapsed)
        if step == args.steps or step % args.diagnostics_every == 0:
            row = diagnostics(flow, velocity, step, args.dt, n, viscosity, xp)
            rows.append(row)
            print(
                f"step={step}/{args.steps} t={row['time']:.4f} "
                f"E={row['kinetic_energy']:.9f} eps={row['viscous_dissipation']:.9f} "
                f"div={row['weak_divergence_rms']:.2e} wall={elapsed:.4f} s",
                flush=True,
            )
    loop_seconds = time.perf_counter() - loop_start
    pressure = flow.pressure(velocity)
    velocity = flow.gather(velocity)
    if device is not None:
        pressure = pressure.get()
    if not np.isfinite(velocity).all() or not np.isfinite(pressure).all():
        raise RuntimeError("The final fields contain non-finite values.")
    np.testing.assert_allclose(pressure.mean(), 0, rtol=0, atol=1e-12)
    np.save(args.out / "velocity.npy", velocity.reshape((2 * n,) * 3 + (3,)))
    np.save(args.out / "pressure.npy", pressure.reshape((n,) * 3))
    with (args.out / "diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "case": "3D incompressible Taylor-Green vortex",
        "discretization": "periodic hexahedral Q2/Q1 with lumped velocity mass",
        "time_integrator": "classical RK4, discrete pressure projection per stage",
        "precision": "float64",
        "assembly": "dolfinx.fem.cuda.VectorAssembler" if device else "dolfinx.fem.assemble_vector",
        "momentum_backend": "ffcx",
        "backend": args.backend,
        "device": (
            xp.cuda.runtime.getDeviceProperties(device.id)["name"].decode() if device else "CPU"
        ),
        "device_index": device.id if device else None,
        "pci_bus_id": device.pci_bus_id if device else None,
        "dolfinx": dolfinx.__version__,
        "dolfinx_commit": dolfinx.git_commit_hash,
        "ffcx": ffcx.__version__,
        "solver_module": periodic.__file__,
        "reynolds": args.reynolds,
        "dt": args.dt,
        "cells_per_axis": n,
        "hexahedral_cells": n**3,
        "velocity_dofs": flow.velocity_dofs.size,
        "pressure_dofs": flow.pressure_dofs.size,
        "steps": args.steps,
        "warmup_steps": args.warmup,
        "diagnostics_every": args.diagnostics_every,
        "setup_seconds": setup_seconds,
        "loop_wall_seconds": loop_seconds,
        "timed_steps": len(timings),
        "steps_per_second": len(timings) / sum(timings),
        "median_step_seconds": statistics.median(timings),
        "step_wall_seconds": timings,
        "relative_assembly_max_error": assembly_error,
        "maximum_weak_divergence_rms_sampled": max(row["weak_divergence_rms"] for row in rows),
        "maximum_advective_cfl_sampled": max(row["advective_cfl"] for row in rows),
        "initial": first,
        "final": rows[-1],
    }
    if device is not None:
        results.update(
            cupy=xp.__version__, cuda_runtime_version=xp.cuda.runtime.runtimeGetVersion()
        )
    if args.cuda_devices is not None:
        results["cuda_devices"] = list(args.cuda_devices)
        results["communication"] = flow.communication_profile()
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"{results['steps_per_second']:.3f} full steps/s "
        f"({len(timings)} timed steps), loop={loop_seconds:.3f} s.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--secondary-gpu", type=int, help="Second CUDA device index")
    parser.add_argument("--cells", type=int, default=48)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--reynolds", type=float, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--diagnostics-every", type=int, default=10)
    args = parser.parse_args()
    args.cuda_devices = None
    if args.secondary_gpu is not None:
        if args.backend != "cuda" or args.secondary_gpu < 0 or args.secondary_gpu == args.gpu:
            parser.error("Select a second, distinct CUDA device for the CUDA backend.")
        args.cuda_devices = (args.gpu, args.secondary_gpu)
    if MPI.COMM_WORLD.size != 1 or args.cells < 4:
        parser.error("Use one MPI rank and at least four cells per direction.")
    if not 0 <= args.warmup < args.steps or args.diagnostics_every < 1:
        parser.error("Require 0 <= warmup < steps and a positive diagnostics interval.")
    if not all(math.isfinite(x) and x > 0 for x in (args.reynolds, args.dt)):
        parser.error("Reynolds number and timestep must be finite and positive.")
    if 96 * args.dt / args.reynolds * (args.cells / LENGTH) ** 2 > 2.4:
        parser.error("The timestep exceeds the conservative RK4 viscous stability bound.")
    if args.backend == "cuda":
        import cupy as cp

        device = cp.cuda.Device(args.gpu)
        with device:
            main(args, cp, device)
    else:
        main(args, np, None)
