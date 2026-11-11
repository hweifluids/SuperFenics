# SuperFenics Examples

| Example | Description |
| --- | --- |
| [3D CPU operator assembly](cpu-assembly/benchmark_cpu.py) | Measures assembly and application of a scalar convection-diffusion operator on two hexahedral meshes and one tetrahedral mesh, and checks equivalence between computation paths. |
| [GPU tetrahedral cell kernel](gpu-cell-kernel/probe.cu) | Generates a cell kernel with FFCx, executes the same kernel on the CPU and one CUDA GPU, and compares results and timings for 65,536 independent cells. The [generator](gpu-cell-kernel/generate_gpu_probe.py) writes the generated sources into the run directory. |
| [3D Taylor–Green DNS](taylor-green-dns/README.md) | Advances an incompressible flow in a periodic cube on one CUDA GPU, including momentum assembly and pressure projection at every RK4 stage. |

Official PDE solver demos and component feature demos are stored in their original submodule demo directories.

## Run the microbenchmarks on the local workstation

Use a Python environment containing the FEniCSx versions corresponding to this repository. The CPU example also uses mpi4py, SciPy, and a C compiler for JIT compilation. The GPU example requires the CUDA Toolkit and the Visual Studio x64 C++ build environment. Run it from an x64 Developer PowerShell. Its CUDA build uses the RTX 2080 Ti sm_75 architecture and the original probe compiler options.

Run the following commands from the repository root, adjusting the Python path for your environment. Before the GPU run, use nvidia-smi to check device indices and utilization, then pass the appropriate index for the current CUDA-visible device list.

~~~powershell
$python = 'F:/SuperFenics/.superfenics-build/venv/Scripts/python.exe'
& $python -B ./examples/run.py cpu-assembly

nvidia-smi
& $python -B ./examples/run.py gpu-cell-kernel --gpu 1
~~~

The launcher locates sources relative to its own path. It restricts the entire process tree to one logical CPU in Windows processor group 0, using CPU 0 by default. Select another logical CPU in that group with --cpu. The CPU example uses one MPI rank, with OpenMP and BLAS thread counts set to 1.

Each run writes its console log, JIT cache, CUDA-generated sources, executable, and JSON results to a separate run directory under .tests/examples/ for the selected example. The launcher prints that directory when the run starts.
