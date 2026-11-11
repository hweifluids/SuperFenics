"""Run a single-GPU finite element Taylor-Green vortex benchmark."""

import argparse
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path

import basix
import cupy as cp
import dolfinx
import ffcx
import numpy as np
import ufl
from cupyx.scipy.sparse import csr_matrix as gpu_csr
from dolfinx import fem, la, mesh
from mpi4py import MPI
from scipy.sparse import csr_matrix


LENGTH = 2 * math.pi
VOLUME = LENGTH**3


def periodic_indices(space, points):
    """Map DOLFINx nodes to the periodic Cartesian lattice."""
    coordinates = space.tabulate_dof_coordinates() * (points / LENGTH)
    indices = np.rint(coordinates).astype(np.int32)
    if np.max(np.abs(coordinates - indices)) > 1e-9:
        raise ValueError("Expected nodes on a uniform Cartesian lattice.")
    ids = np.ravel_multi_index((indices % points).T, (points,) * 3)
    if np.unique(ids).size != points**3:
        raise ValueError("The periodic lattice is incomplete.")
    return ids


def pressure_inverse(divergence, gradient, mass_inverse, cells):
    """Diagonalize the periodic finite element pressure Schur complement."""
    shape = (cells,) * 3
    impulse = np.zeros(cells**3)
    impulse[0] = 1
    column = divergence @ (mass_inverse * (gradient @ impulse))
    symbol = np.fft.rfftn(column.reshape(shape))
    scale = np.max(np.abs(symbol))
    if np.max(np.abs(symbol.imag)) > 1e-12 * scale or abs(symbol.flat[0]) > 1e-12 * scale:
        raise ValueError("Expected a symmetric pressure operator with a constant nullspace.")
    spectrum = symbol.real.copy()
    spectrum.flat[0] = 1
    if np.min(spectrum) <= 1e-13 * scale:
        raise ValueError("The pressure operator has an unexpected null or negative mode.")

    # Check that the impulse response also represents the assembled operator away from the origin.
    sample = np.random.default_rng(71).standard_normal(cells**3)
    direct = divergence @ (mass_inverse * (gradient @ sample))
    transformed = np.fft.irfftn(
        symbol * np.fft.rfftn(sample.reshape(shape)), s=shape, axes=(0, 1, 2)
    )
    np.testing.assert_allclose(transformed.ravel(), direct, rtol=1e-10, atol=1e-11 * scale)
    inverse = 1 / spectrum
    inverse.flat[0] = 0
    return cp.asarray(inverse)


def compile_momentum(form, out):
    """Wrap the FFCx volume kernel with CUDA cell assembly."""
    pieces = (form.code,) if isinstance(form.code, str) else form.code or ()
    code = "\n".join(piece for piece in pieces if isinstance(piece, str))
    if not code:
        code = (out / "jit" / f"{form.module.__name__}.c").read_text(encoding="utf-8")
    kernels = list(re.finditer(r"\bvoid (tabulate_tensor_\w+)\s*\([^{};]*\)\s*\{", code))
    if len(kernels) != 1:
        raise ValueError(f"Expected one FFCx volume kernel, found {len(kernels)}.")
    match = kernels[0]
    end = match.end()
    depth = 1
    while depth:
        depth += (code[end] == "{") - (code[end] == "}")
        end += 1

    kernel = code[match.start():end]
    kernel = re.sub(r"\brestrict\b", "__restrict__", kernel)
    kernel = kernel.replace(f"void {match[1]}", "__device__ void cell_momentum", 1)
    wrapper = Path(__file__).with_name("assemble_momentum.cu").read_text(encoding="utf-8")
    # NVRTC supplies the CUDA builtins. Use the fixed-width alias required by UFCx.
    source = "using uint8_t = unsigned char;\n" + kernel + "\n" + wrapper
    (out / "generated_momentum.cu").write_text(source, encoding="utf-8")
    return cp.RawModule(code=source, options=("--std=c++17", "--fmad=false")).get_function(
        "assemble_momentum"
    )


class TaylorGreen:
    """Periodic Q2/Q1 spatial discretization and projected RK4 time integration."""

    def __init__(self, cells, reynolds, out):
        self.cells = cells
        self.viscosity = 1.0 / reynolds
        self.pressure_shape = (cells,) * 3
        self.velocity_shape = (2 * cells,) * 3 + (3,)
        self.jit = {"cache_dir": out / "jit", "timeout": 180}
        self.domain = mesh.create_box(
            MPI.COMM_SELF, ((0, 0, 0), (LENGTH,) * 3), (cells,) * 3,
            cell_type=mesh.CellType.hexahedron, dtype=np.float64,
        )
        space = fem.functionspace(self.domain, ("Lagrange", 2, (3,)))
        pressure = fem.functionspace(self.domain, ("Lagrange", 1))
        self.coefficient = fem.Function(space, dtype=np.float64)
        self.velocity_ids = periodic_indices(space, 2 * cells)
        self.pressure_ids = periodic_indices(pressure, cells)
        vector_ids = (3 * self.velocity_ids[:, None] + np.arange(3)).ravel()
        v = ufl.TestFunction(space)
        q = ufl.TestFunction(pressure)
        u = self.coefficient
        nu = fem.Constant(self.domain, np.float64(self.viscosity))
        dx = ufl.Measure("dx", domain=self.domain)

        weights = fem.assemble_vector(fem.form(
            ufl.inner(ufl.as_vector((1.0, 1.0, 1.0)), v) * dx, jit_options=self.jit
        ))
        weights.scatter_reverse(la.InsertMode.add)
        mass = np.bincount(vector_ids, weights=weights.array, minlength=math.prod(self.velocity_shape))
        if np.any(mass <= 0) or not np.isclose(mass.sum(), 3 * VOLUME):
            raise ValueError("Invalid periodic lumped mass.")
        mass_inverse = 1 / mass

        # Assemble scalar blocks because the native Python CSR assembler requires
        # equal row and column block sizes.
        scalar_space = fem.functionspace(self.domain, ("Lagrange", 2))
        scalar_ids = periodic_indices(scalar_space, 2 * cells)
        trial = ufl.TrialFunction(scalar_space)
        rows, columns, values = [], [], []
        for axis in range(3):
            div_form = fem.form(trial.dx(axis) * q * dx, jit_options=self.jit)
            matrix = fem.assemble_matrix(div_form)
            matrix.scatter_reverse()
            entries = matrix.to_scipy().tocoo()
            rows.append(self.pressure_ids[entries.row])
            columns.append(3 * scalar_ids[entries.col] + axis)
            values.append(entries.data)
        divergence = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(cells**3, mass.size),
        )
        divergence.eliminate_zeros()
        gradient = divergence.T.tocsr()
        self.inverse_symbol = pressure_inverse(divergence, gradient, mass_inverse, cells)
        self.divergence = gpu_csr(divergence)
        self.gradient = gpu_csr(gradient)
        self.mass = cp.asarray(mass)
        self.mass_inverse = cp.asarray(mass_inverse)

        advection = 0.5 * (
            ufl.inner(ufl.dot(ufl.grad(u), u), v) - ufl.inner(ufl.dot(ufl.grad(v), u), u)
        )
        diffusion = nu * ufl.inner(ufl.grad(u), ufl.grad(v))
        self.momentum = fem.form(
            (advection + diffusion) * dx(metadata={"quadrature_degree": 6}),
            jit_options=self.jit,
        )
        cell_dofs = space.dofmap.list
        geometry = self.domain.geometry.x[self.domain.geometry.dofmaps[0]]
        if (
            space.dofmap.bs != 3
            or cell_dofs.shape != (cells**3, 27)
            or geometry.shape != (cells**3, 8, 3)
            or self.momentum.ufcx_form.num_coefficients != 1
            or self.momentum.ufcx_form.num_constants != 1
        ):
            raise ValueError("Unexpected layout for the generated momentum kernel.")
        self.cell_dofs = cp.asarray(self.velocity_ids[cell_dofs], dtype=cp.int32)
        self.geometry = cp.asarray(geometry)
        self.assemble = compile_momentum(self.momentum, out)

    def residual(self, velocity):
        result = cp.zeros_like(velocity)
        threads = 128
        self.assemble(
            ((self.cells**3 + threads - 1) // threads,), (threads,),
            (np.int32(self.cells**3), self.cell_dofs, self.geometry, velocity,
             np.float64(self.viscosity), result),
        )
        return result

    def pressure_correction(self, velocity):
        divergence = (self.divergence @ velocity).reshape(self.pressure_shape)
        potential = cp.fft.irfftn(
            cp.fft.rfftn(divergence) * self.inverse_symbol, s=self.pressure_shape, axes=(0, 1, 2)
        ).ravel()
        return self.mass_inverse * (self.gradient @ potential), potential

    def derivative(self, velocity):
        acceleration = -self.mass_inverse * self.residual(velocity)
        correction, _ = self.pressure_correction(acceleration)
        return acceleration - correction

    def step(self, velocity, dt):
        k1 = self.derivative(velocity)
        k2 = self.derivative(velocity + 0.5 * dt * k1)
        k3 = self.derivative(velocity + 0.5 * dt * k2)
        k4 = self.derivative(velocity + dt * k3)
        return velocity + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def initial_velocity(self):
        x = cp.arange(2 * self.cells, dtype=cp.float64) * (LENGTH / (2 * self.cells))
        s, c = cp.sin(x), cp.cos(x)
        velocity = cp.zeros(self.velocity_shape, dtype=cp.float64)
        velocity[..., 0] = s[:, None, None] * c[None, :, None] * c[None, None, :]
        velocity[..., 1] = -c[:, None, None] * s[None, :, None] * c[None, None, :]
        velocity = velocity.ravel()
        correction, _ = self.pressure_correction(velocity)
        return velocity - correction

    def check_assembly(self):
        """Compare the complete GPU gather, cell evaluation, and scatter to DOLFINx."""
        velocity = np.random.default_rng(29).standard_normal(self.mass.size)
        self.coefficient.x.array[:] = velocity.reshape(-1, 3)[self.velocity_ids].ravel()
        reference = fem.assemble_vector(self.momentum)
        reference.scatter_reverse(la.InsertMode.add)
        folded = np.zeros(velocity.shape).reshape(-1, 3)
        np.add.at(folded, self.velocity_ids, reference.array.reshape(-1, 3))
        actual = cp.asnumpy(self.residual(cp.asarray(velocity)))
        expected = folded.ravel()
        error = float(np.max(np.abs(actual - expected)))
        scale = float(np.max(np.abs(expected)))
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12 * scale)
        return error / scale

    def diagnostics(self, velocity, step, dt):
        residual = self.residual(velocity)
        weighted_square = self.mass * velocity**2
        energy = float(cp.sum(weighted_square)) / (2 * VOLUME)
        spanwise_energy = float(cp.sum(weighted_square[2::3])) / (2 * VOLUME)
        # The skew convection contributes zero to u^T R. The pressure term also
        # does no work for a velocity in the discrete divergence-free space.
        dissipation = float(cp.dot(velocity, residual)) / VOLUME
        divergence = (self.divergence @ velocity) * (self.cells**3 / VOLUME)
        divergence_rms = float(cp.sqrt(cp.mean(divergence**2)))
        maximum = float(cp.max(cp.sum(cp.abs(velocity.reshape(-1, 3)), axis=1)))
        cfl = maximum * dt * (2 * self.cells) / LENGTH
        if not all(map(math.isfinite, (energy, spanwise_energy, dissipation, divergence_rms, cfl))):
            raise RuntimeError(f"Nonfinite flow diagnostics at step {step}.")
        if energy <= 0 or dissipation <= 0 or divergence_rms > 1e-10 or cfl > 0.8:
            raise RuntimeError(
                f"Invalid flow at step {step}: E={energy}, eps={dissipation}, "
                f"weak divergence={divergence_rms}, CFL={cfl}. Reduce dt if needed."
            )
        return {
            "step": step, "time": step * dt, "kinetic_energy": energy,
            "spanwise_kinetic_energy": spanwise_energy,
            "viscous_dissipation": dissipation, "weak_divergence_rms": divergence_rms,
            "advective_cfl": cfl, "kolmogorov_length": (self.viscosity**3 / dissipation)**0.25,
        }

    def save_fields(self, velocity, out):
        _, potential = self.pressure_correction(-self.mass_inverse * self.residual(velocity))
        np.save(out / "velocity.npy", cp.asnumpy(velocity.reshape(self.velocity_shape)))
        np.save(out / "pressure.npy", -cp.asnumpy(potential.reshape(self.pressure_shape)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cells", type=int, default=48, help="Hexahedral cells per axis")
    parser.add_argument("--reynolds", type=float, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=500, help="Total physical RK4 steps")
    parser.add_argument("--warmup", type=int, default=5, help="Initial steps excluded from timing")
    parser.add_argument("--diagnostics-every", type=int, default=10)
    parser.add_argument("--check-assembly", action="store_true")
    parser.add_argument("--save-fields", action="store_true")
    args = parser.parse_args()
    if args.cells < 4 or not 0 <= args.warmup < args.steps or args.diagnostics_every < 1:
        parser.error("Require cells >= 4, 0 <= warmup < steps, and a positive diagnostics interval.")
    if not all(math.isfinite(x) and x > 0 for x in (args.reynolds, args.dt)):
        parser.error("Reynolds number and dt must be finite and positive.")
    # A conservative bound from the lumped Q2 diffusion operator and the RK4
    # stability interval on the negative real axis.
    if 96 * args.dt / args.reynolds * (args.cells / LENGTH)**2 > 2.4:
        parser.error("dt exceeds the conservative explicit viscous stability limit.")
    return args


def main():
    args = parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run the single-device benchmark with one MPI rank.")
    args.out.mkdir(parents=True, exist_ok=True)
    device = cp.cuda.Device(args.gpu)
    with device:
        print(
            f"Preparing {args.cells}^3 cells at Re={args.reynolds:g}, dt={args.dt:g} "
            f"on CUDA device {args.gpu} ({device.pci_bus_id}).", flush=True,
        )
        start = time.perf_counter()
        available_start, _ = device.mem_info
        flow = TaylorGreen(args.cells, args.reynolds, args.out)
        assembly_error = flow.check_assembly() if args.check_assembly else None
        velocity = flow.initial_velocity()
        first = flow.diagnostics(velocity, 0, args.dt)
        np.testing.assert_allclose(first["kinetic_energy"], 0.125, atol=1e-9, rtol=0)
        setup_seconds = time.perf_counter() - start
        print(f"Setup complete in {setup_seconds:.2f} s. Initial E={first['kinetic_energy']:.10f}.",
              flush=True)
        rows = [first]
        timings = []
        loop_start = time.perf_counter()
        for step in range(1, args.steps + 1):
            start = time.perf_counter()
            velocity = flow.step(velocity, args.dt)
            device.synchronize()
            elapsed = time.perf_counter() - start
            if step > args.warmup:
                timings.append(elapsed)
            if step == args.steps or step % args.diagnostics_every == 0:
                row = flow.diagnostics(velocity, step, args.dt)
                rows.append(row)
                print(
                    f"step={step}/{args.steps} t={row['time']:.4f} "
                    f"E={row['kinetic_energy']:.9f} eps={row['viscous_dissipation']:.9f} "
                    f"weak_div={row['weak_divergence_rms']:.2e} "
                    f"CFL={row['advective_cfl']:.3f} wall={elapsed:.3f} s", flush=True,
                )
        loop_seconds = time.perf_counter() - loop_start
        energy_lost = first["kinetic_energy"] - rows[-1]["kinetic_energy"]
        integrated_dissipation = float(np.trapezoid(
            [row["viscous_dissipation"] for row in rows], [row["time"] for row in rows]
        ))
        if args.save_fields:
            flow.save_fields(velocity, args.out)
        free, total = device.mem_info
        properties = cp.cuda.runtime.getDeviceProperties(args.gpu)
        results = {
            "case": "3D incompressible Taylor-Green vortex",
            "discretization": "periodic hexahedral Q2/Q1 with lumped velocity mass",
            "time_integrator": "classical RK4, discrete pressure projection per stage",
            "precision": "float64",
            "device": properties["name"].decode(),
            "device_index": args.gpu,
            "pci_bus_id": device.pci_bus_id,
            "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "cupy": cp.__version__, "dolfinx": dolfinx.__version__,
            "dolfinx_commit": dolfinx.git_commit_hash,
            "basix": basix.__version__, "ufl": ufl.__version__, "ffcx": ffcx.__version__,
            "reynolds": args.reynolds, "kinematic_viscosity": flow.viscosity,
            "dt": args.dt, "steps": args.steps, "warmup_steps": args.warmup,
            "cells_per_axis": args.cells, "hexahedral_cells": args.cells**3,
            "velocity_dofs": int(velocity.size), "pressure_dofs": args.cells**3,
            "setup_seconds": setup_seconds, "loop_wall_seconds": loop_seconds,
            "timed_steps": len(timings), "steps_per_second": len(timings) / sum(timings),
            "median_step_seconds": statistics.median(timings),
            "step_wall_seconds": timings,
            "gpu_total_bytes": total, "gpu_free_bytes_before": available_start,
            "gpu_free_bytes_after": free,
            "cupy_pool_bytes_after": cp.get_default_memory_pool().total_bytes(),
            "relative_assembly_max_error": assembly_error,
            "energy_balance_residual": integrated_dissipation - energy_lost,
            "minimum_kolmogorov_length_sampled": min(row["kolmogorov_length"] for row in rows),
            "maximum_weak_divergence_rms_sampled": max(row["weak_divergence_rms"] for row in rows),
            "maximum_advective_cfl_sampled": max(row["advective_cfl"] for row in rows),
            "initial": first, "final": rows[-1],
        }
        (args.out / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        with (args.out / "diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(first))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"{results['steps_per_second']:.3f} full steps/s, "
            f"median {results['median_step_seconds']:.4f} s/step "
            f"({len(timings)} timed steps).", flush=True,
        )


if __name__ == "__main__":
    main()
