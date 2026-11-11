# SuperFenics

FEniCSx 开发工作区。主仓库将四个 GitHub fork 固定为 Git 子模块，核心源码目录位于仓库根目录。

| 相对目录 | 内容 |
| --- | --- |
| dolfinx/ | C++ 求解核心、Python 接口和有限元装配 |
| basix/ | 有限元类型和基函数 |
| ufl/ | 弱式的符号表达 |
| ffcx/ | 弱式编译和 UFCx 接口 |
| examples/ | 四个组件的官方算例，按来源组织 |
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

修改子模块或集中算例前先检查工作树。更新上游或切换提交后，复查 examples/ 与模块原 demo 入口的一致性，不覆盖未提交内容。

## 官方算例

[examples/README.md](examples/README.md) 列出了 124 个官方算例入口的分类、来源和运行要求。当前本机已将官方 demo 实体移入 examples/，原路径保留兼容目录联接。Windows 的重新整理入口为以下命令。

~~~powershell
pwsh -File examples/organize.ps1
~~~

具体处理及重复文件保护方式见算例说明。此次迁移保留了全部 158 个上游 demo 文件及相应许可证。

## 本地工作文件

根目录 AGENTS.md、.plans/、.tests/、.reports/ 及三个目录内的内容均不纳入版本控制。新克隆按本机需要建立这些本地内容。

测试使用 .tests/，根目录 tests/ 已移除。原本机 Poisson 检查脚本保留在 .tests/smoke_poisson.py，因此不随 Git 克隆分发。

报告按 .tests/data-{module}-{做了什么} 命名。同一功能点持续更新同一份报告，详细工作规则保存在本机 AGENTS.md。
