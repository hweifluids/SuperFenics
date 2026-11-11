extern "C" __global__ void assemble_momentum(
    int cells, const int* dofmap, const double* geometry,
    const double* velocity, double viscosity, double* residual)
{
    constexpr int nodes = 27;
    constexpr int width = 3;
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cells)
        return;

    const int* dofs = dofmap + nodes * cell;
    double w[nodes * width];
    double local[nodes * width] = {};
    for (int i = 0; i < nodes; ++i)
        for (int d = 0; d < width; ++d)
            w[width * i + d] = velocity[width * dofs[i] + d];

    cell_momentum(local, w, &viscosity, geometry + 24 * cell, nullptr, nullptr, nullptr);

    for (int i = 0; i < nodes; ++i)
        for (int d = 0; d < width; ++d)
            atomicAdd(residual + width * dofs[i] + d, local[width * i + d]);
}
