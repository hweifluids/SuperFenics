"""Export one unchanged FFCx cell-action kernel for a CUDA feasibility probe."""

import argparse
import json
import re
from pathlib import Path

import basix.ufl
import ffcx.compiler
import ffcx.options
import ufl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
out = parser.parse_args().out.resolve()
out.mkdir(parents=True, exist_ok=True)
coordinate = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
domain = ufl.Mesh(coordinate)
V = ufl.FunctionSpace(domain, basix.ufl.element("Lagrange", "tetrahedron", 2))
W = ufl.FunctionSpace(domain, basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
state = ufl.Coefficient(V)
velocity = ufl.Coefficient(W)
test = ufl.TestFunction(V)
dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
form = (10.0 * state * test + 0.01 * ufl.inner(ufl.grad(state), ufl.grad(test))
        + ufl.dot(velocity, ufl.grad(state)) * test) * dx
sources, suffixes = ffcx.compiler.compile_ufl_objects([form], ffcx.options.get_options())
for text, suffix in zip(sources, suffixes):
    (out / f"probe_original{suffix}").write_text(text, encoding="utf-8")
code = next(text for text, suffix in zip(sources, suffixes) if suffix == ".c")
match = re.search(r"void (tabulate_tensor_\w+)\s*\(", code)
assert match is not None
begin = match.start()
body = code.index("{", match.end())
level = 1
end = body + 1
while level:
    level += (code[end] == "{") - (code[end] == "}")
    end += 1
kernel = code[begin:end]
kernel = re.sub(r"\brestrict\b", "__restrict__", kernel)
kernel = kernel.replace("void " + match.group(1), "__host__ __device__ inline void generated_action", 1)
(out / "probe_kernel.inc").write_text(kernel + "\n", encoding="utf-8")
meta = {
    "method": "3D tetrahedral P2 scalar transient convection-diffusion cell action",
    "dtype": "float64",
    "quadrature_degree": 6,
    "local_state_dofs": 10,
    "packed_coefficient_dofs": sum(c.ufl_function_space().ufl_element().dim for c in form.coefficients()),
    "coordinate_dofs": 12,
    "arithmetic_changes": "none; restrict spelling and host/device function annotation only",
    "omissions": ["global gather/scatter", "boundary conditions", "linear solve", "MPI"],
}
assert meta["packed_coefficient_dofs"] == 22
(out / "probe_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
