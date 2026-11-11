# SuperFenics 示例

| 示例 | 内容 |
| --- | --- |
| [CPU 三维算子装配](cpu-assembly/benchmark_cpu.py) | 在两组六面体网格和一组四面体网格上测量标量对流扩散算子的装配与作用，核对不同计算路径的结果。 |
| [GPU 四面体单元核](gpu-cell-kernel/probe.cu) | 用 [FFCx 生成单元核](gpu-cell-kernel/generate_gpu_probe.py)，在 CPU 和单张 CUDA GPU 上运行同一核，对 65,536 个独立单元做结果对照和计时。 |

两者都是针对局部计算路径的微基准。官方完整求解示例及各组件的功能示例随子模块保留在原来的 demo 目录。

## 本机运行

使用已安装仓库对应版本的 FEniCSx 的 Python。CPU 示例还使用 mpi4py、SciPy 和用于 JIT 的 C 编译器。GPU 示例需要 CUDA Toolkit 与 Visual Studio 的 x64 C++ 编译环境，请在 x64 Developer PowerShell 中运行。GPU 入口沿用 RTX 2080 Ti 的 sm_75 架构和原有编译选项。

从仓库根目录运行以下命令，将 Python 路径换成实际环境。启动 GPU 示例前用 nvidia-smi 核对设备和占用，按当前 CUDA 可见设备列表传入编号。

~~~powershell
$python = 'F:/SuperFenics/.superfenics-build/venv/Scripts/python.exe'
& $python -B ./examples/run.py cpu-assembly

nvidia-smi
& $python -B ./examples/run.py gpu-cell-kernel --gpu 1
~~~

运行入口按脚本位置定位源码。它将 Windows 处理器组 0 的指定逻辑 CPU 应用于整个进程树，默认用 CPU 0，也可以通过 --cpu 选择组内其他逻辑 CPU。CPU 示例使用单 MPI rank，并把 OpenMP 和 BLAS 的线程数设为 1。每次运行的控制台日志、JIT 缓存、CUDA 生成代码和可执行文件以及 JSON 结果写入 .tests/examples/ 下对应示例的独立运行目录。终端会显示运行目录。
