# SuperFenics

FEniCSx 开发工作区。主仓库将四个 GitHub fork 固定为 Git 子模块，四个源码目录都位于主仓库根目录。

| 相对目录 | 组件 |
| --- | --- |
| dolfinx/ | C++ 求解核心、Python 接口和有限元装配 |
| basix/ | 有限元类型和基函数 |
| ufl/ | 弱式的符号表达 |
| ffcx/ | 弱式编译和 UFCx 接口 |

目录名沿用 FEniCSx 的源码构建布局。C++ 入口位于 basix/cpp 和 dolfinx/cpp，Python 绑定分别位于 basix/python 和 dolfinx/python。UFL 和 FFCx 的 Python 包从各自仓库根目录构建，UFCx 的 C 接口从 ffcx/cmake 安装。 [DOLFINx 官方源码构建说明](https://docs.fenicsproject.org/dolfinx/main/python/installation.html)

克隆和恢复主仓库指定的子模块版本。

~~~sh
git clone --recurse-submodules https://github.com/hweifluids/SuperFenics.git
cd SuperFenics
git submodule status
~~~

已有检出可以执行以下命令，使四个子模块跟随主仓库固定的提交。

~~~sh
git pull --ff-only
git submodule update --init --recursive
~~~

从上述源码构建并安装四个组件后，用同一个 Python 环境运行串行 Poisson 测试。环境需要 MPI、C 编译器和 SciPy。测试核对 DOLFINx 二进制的 Git 提交，再用 UFL、Basix、FFCx 和 DOLFINx 完成二次有限元装配并求解，检查节点误差和线性残差。

~~~sh
python tests/smoke_poisson.py --cache-dir ../superfenics-jit
~~~
