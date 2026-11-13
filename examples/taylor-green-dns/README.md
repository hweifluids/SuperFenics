# 3D Taylor–Green vortex DNS benchmark

Solve the unforced incompressible Navier–Stokes equations in a cube of side $2\pi$, periodic in all three directions. The nondimensional initial velocity is

$$
\boldsymbol{u}(x,y,z,0)=
(\sin x\cos y\cos z,\ -\cos x\sin y\cos z,\ 0).
$$

The velocity amplitude and the reference length in the trigonometric arguments are both 1. The default Reynolds number is 200, with kinematic viscosity $\nu=1/200$. This is a direct simulation with physical viscosity and no turbulence closure. Initial kinetic energy per unit mass is $1/8$. The corresponding continuum initial dissipation rate is $3\nu/4$. For the flow and its evolution, see [Brachet et al. (1983)](https://doi.org/10.1017/S0022112083001159).

## Discretization and execution

- DOLFINx builds a uniform hexahedral mesh and a Taylor–Hood space with continuous Q2 vector velocity and continuous Q1 pressure. Degrees of freedom on opposite faces are identified periodically. Basix provides the basis functions, UFL describes the forms, and FFCx generates the cell kernels.
- The method uses row-sum lumped velocity mass, skew-symmetric convection, and viscous diffusion. The default GPU momentum backend evaluates the same Q2 form by successive one-dimensional contractions. It uses the original 64 quadrature points, FP64 arithmetic, the DOLFINx cell maps, and Basix basis tables. Basis values, derivatives, and uniform geometry factors are prepared once. Use --backend ffcx to select the original generated CUDA cell kernel.
- Each stage of classical RK4 applies the discrete pressure projection. With divergence matrix $B$ and lumped mass $M$, the pressure operator is $S=B M^{-1}B^{T}$. On the periodic uniform pressure lattice, its convolution symbol comes from the assembled operator applied to a unit impulse. A setup check compares its FFT action to a direct sparse application on a second vector. The constant pressure mode is set to zero.
- All timed stages use one CUDA device in float64 for momentum assembly, the sparse divergence and gradient, the pressure FFT solve, and the state update. The CPU prepares the static operators and launches GPU work.

## Run on Windows

Use the FEniCSx environment matching the submodule revisions. The setup uses SciPy and the same C compiler for FFCx JIT as the CPU assembly example. The timed solver uses [CuPy for CUDA 13](https://docs.cupy.dev/en/stable/install.html) and the installed CUDA Toolkit. The commands below install the verified CuPy stack into a local dependency directory without modifying the FEniCSx environment.

~~~powershell
$python = 'F:/SuperFenics/.superfenics-build/venv/Scripts/python.exe'
$env:UV_CACHE_DIR = Join-Path $PWD '.tests/cuda-python/cache'
uv pip install --python $python --target .tests/cuda-python/site-packages 'cupy-cuda13x==14.2.0' 'numpy==2.5.4'
$env:PYTHONPATH = Join-Path $PWD '.tests/cuda-python/site-packages'

nvidia-smi
& $python -B ./examples/run.py taylor-green-dns --gpu 1 --save-fields
~~~

Select an available GPU by its current CUDA index. The launcher orders devices by PCI bus ID and binds the process tree to one logical CPU in Windows processor group 0. It defaults to logical CPU 0.

The default uses $48^3$ hexahedra with $96^3$ periodic velocity nodes. It advances 500 steps at $\Delta t=0.02$ to $t=10$. For a shorter timing run, add --steps 30. To compare the GPU momentum assembly with the original DOLFINx CPU assembly on the same discrete input, add --check-assembly. The options --cells, --reynolds, --dt, --warmup, and --diagnostics-every control the mesh and run parameters.

The launcher writes the console log, generated kernel, caches, results.json, and diagnostics.csv to the run directory under .tests/examples/taylor-green-dns/. With --save-fields it writes velocity.npy with shape (2N, 2N, 2N, 3) and pressure.npy with shape (N, N, N), where N is the hexahedral cell count per axis. Both grids start at zero and omit the repeated periodic endpoint. The velocity components are ordered x, y, z.

## Measurement

The step timer includes the four RK4 stages, all pressure projections, and device synchronization. It excludes mesh setup, JIT, the first five warmup steps, diagnostics, and field output. The JSON result records all timed step durations and the wall time of the complete time loop. The CSV records kinetic energy, spanwise kinetic energy, viscous dissipation, the pressure-space divergence residual, the advective CFL estimate, and the inferred Kolmogorov length. The divergence residual measures the discrete weak constraint.

## Workstation measurements

The default mesh has 110,592 cells, 2,654,208 velocity degrees of freedom, and 110,592 pressure coefficients with a zero-mean constraint. On one RTX 2080 Ti with 22 GiB of memory, the original FFCx GPU backend measured 3.71 full steps/s after warmup. The default tensor backend measured 16.16 steps/s under the same conditions. The complete 500-step loop, including warmup and diagnostics, took 138.39 s with FFCx and 32.04 s with the tensor backend. The final velocity and pressure differed by about 1.37e-13 in relative L2 norm.

For the original FFCx run, at 51 matching times between $t=0$ and $t=10$, an independent, dealiased Fourier reference using $80^3$ collocation points gave maximum relative differences of 0.38% in kinetic energy and 1.23% in viscous dissipation. The relative velocity-field error at $t=10$ was 2.04% after interpolating the reference to the finite element velocity nodes. The maximum sampled RMS weak divergence was $7.76 \times 10^{-14}$. The measured throughput and accuracy apply to this mesh and Reynolds number.
