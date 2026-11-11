# SuperFenics

FEniCSx 开发工作区。主仓库将四个 GitHub fork 固定为 Git 子模块，核心源码目录位于仓库根目录。

| 相对目录 | 内容 |
| --- | --- |
| dolfinx/ | C++ 求解核心、Python 接口和有限元装配 |
| basix/ | 有限元类型和基函数 |
| ufl/ | 弱式的符号表达 |
| ffcx/ | 弱式编译和 UFCx 接口 |
| examples/ | SuperFenics 自建的 CPU 装配与 GPU 单元核示例 |
| .tests/ | 本地测试、构建、缓存、运行数据和报告，不跟踪 |
| .plans/、.reports/ | 本地保留目录，不跟踪 |

C++ 构建入口位于 basix/cpp 和 dolfinx/cpp，Python 绑定分别位于 basix/python 和 dolfinx/python。UFL 和 FFCx 的 Python 包从各自仓库根目录构建，UFCx 的 C 接口从 ffcx/cmake 安装。[DOLFINx 官方源码构建说明](https://docs.fenicsproject.org/dolfinx/main/python/installation.html)

## 克隆与同步

~~~sh
git clone --recurse-submodules https://github.com/hweifluids/SuperFenics.git
cd SuperFenics
git submodule status
~~~

已有检出可以执行以下命令恢复主仓库固定的子模块版本。

~~~sh
git pull --ff-only
git submodule update --init --recursive
~~~

修改或更新子模块前先检查工作树。主仓库通过子模块提交固定各组件的源码和官方算例版本。

## 算例

官方 demo 随各子模块检出。DOLFINx 的 Python 和 C++ 示例分别在 dolfinx/python/demo/ 和 dolfinx/cpp/demo/，Basix、UFL、FFCx 的示例分别在 basix/demo/、ufl/demo/ 和 ffcx/demo/。

根目录 [examples/](examples/README.md) 提供 SuperFenics 的两类微基准示例：三维 CPU 算子装配和 GPU 四面体单元核。运行说明及适用范围见该目录。

## 本地工作文件

根目录 AGENTS.md、.plans/、.tests/、.reports/ 及三个目录内的内容均不纳入版本控制。新克隆按本机需要建立这些本地内容。

本地测试和示例运行产物使用 .tests/。原本机 Poisson 检查脚本保留在 .tests/smoke_poisson.py，因此不随 Git 克隆分发。

报告按 .tests/data-{module}-{做了什么} 命名。同一功能点持续更新同一份报告，详细工作规则保存在本机 AGENTS.md。
