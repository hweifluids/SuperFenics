# SuperFenics

A FEniCSx development workspace. The parent repository pins four GitHub forks as Git submodules, with the component source directories at the repository root. Project code, comments, and documentation use English.

| Relative directory | Contents |
| --- | --- |
| dolfinx/ | C++ solver core, Python interface, and finite element assembly |
| basix/ | Finite element definitions and basis functions |
| ufl/ | Symbolic representation of variational forms |
| ffcx/ | Form compiler and UFCx interface |
| examples/ | SuperFenics CPU assembly and GPU cell-kernel examples |
| .tests/ | Local tests, builds, caches, runtime data, and reports, excluded from Git |
| .plans/, .reports/ | Reserved local directories, excluded from Git |

The C++ build entry points are basix/cpp and dolfinx/cpp. Their Python bindings are built from basix/python and dolfinx/python. The UFL and FFCx Python packages are built from their component roots, and the UFCx C interface is installed from ffcx/cmake. See the [official DOLFINx source-build instructions](https://docs.fenicsproject.org/dolfinx/main/python/installation.html).

## Clone and synchronize

~~~sh
git clone --recurse-submodules https://github.com/hweifluids/SuperFenics.git
cd SuperFenics
git submodule status
~~~

For an existing checkout, restore the component revisions pinned by the parent repository with the following commands.

~~~sh
git pull --ff-only
git submodule update --init --recursive
~~~

Check the working trees before modifying or updating a submodule. The parent repository pins component sources and official examples through the submodule commits.

## Examples

Official demos are included in each submodule. DOLFINx Python and C++ examples are in dolfinx/python/demo/ and dolfinx/cpp/demo/. Basix, UFL, and FFCx examples are in basix/demo/, ufl/demo/, and ffcx/demo/.

The root examples/ directory provides two SuperFenics microbenchmarks covering 3D CPU operator assembly and a GPU tetrahedral cell kernel. Their scope and execution instructions are documented in the [examples README](examples/README.md).

## Local workspace files

The root AGENTS.md, .plans/, .tests/, .reports/, and all contents of the three directories are excluded from version control. Create these local files and directories as needed in a fresh checkout.

Store local tests and example runtime artifacts under .tests/. The local Poisson installation check is stored in .tests/smoke_poisson.py and is excluded from Git.

Name reports .tests/data-{module}-{action}. Update the same report for work on the same feature or task area. Detailed local working rules are stored in AGENTS.md.
