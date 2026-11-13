extern "C" __global__ void assemble_momentum(
    int cells, const int* dofmap, const double*,
    const double* velocity, double viscosity, double* residual)
{
    // The map uses tensor order (x, y, z), with three components per node.
    constexpr int components = 3;
    constexpr int nodal_size = 3 * 3 * 3 * components;
    constexpr int x_size = 4 * 3 * 3 * components;
    constexpr int xy_size = 4 * 4 * 3 * components;
    constexpr int q_size = 4 * 4 * 4 * components;
    __shared__ double nodal[nodal_size];
    __shared__ double x[2][x_size];
    __shared__ double xy[3][xy_size];
    __shared__ double quad[4][q_size];

    const int cell = blockIdx.x;
    if (cell >= cells)
        return;
    const int* dofs = dofmap + 27 * cell;
    const int lane = threadIdx.x;
    for (int t = lane; t < nodal_size; t += blockDim.x)
        nodal[t] = velocity[components * dofs[t / components] + t % components];
    __syncthreads();

    // Forward interpolation. Carry one reference derivative per axis.
    for (int t = lane; t < x_size; t += blockDim.x)
    {
        const int qx = t / (3 * 3 * components);
        const int tail = t % (3 * 3 * components);
        double value = 0, dx = 0;
        #pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            const double u = nodal[i * 3 * 3 * components + tail];
            value += basis[qx][i] * u;
            dx += derivative[qx][i] * u;
        }
        x[0][t] = value;
        x[1][t] = dx;
    }
    __syncthreads();

    for (int t = lane; t < xy_size; t += blockDim.x)
    {
        const int qx = t / (4 * 3 * components);
        const int qy = t / (3 * components) % 4;
        const int tail = t % (3 * components);
        double value = 0, dx = 0, dy = 0;
        #pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            const int from = (qx * 3 + i) * 3 * components + tail;
            value += basis[qy][i] * x[0][from];
            dx += basis[qy][i] * x[1][from];
            dy += derivative[qy][i] * x[0][from];
        }
        xy[0][t] = value;
        xy[1][t] = dx;
        xy[2][t] = dy;
    }
    __syncthreads();

    for (int t = lane; t < q_size; t += blockDim.x)
    {
        const int qxy = t / (4 * components);
        const int qz = t / components % 4;
        const int component = t % components;
        double value = 0, dx = 0, dy = 0, dz = 0;
        #pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            const int from = (qxy * 3 + i) * components + component;
            value += basis[qz][i] * xy[0][from];
            dx += basis[qz][i] * xy[1][from];
            dy += basis[qz][i] * xy[2][from];
            dz += derivative[qz][i] * xy[0][from];
        }
        quad[0][t] = value;
        quad[1][t] = dx;
        quad[2][t] = dy;
        quad[3][t] = dz;
    }
    __syncthreads();

    // Form the same skew convection and diffusion integrands as the UFL form.
    for (int q = lane; q < 64; q += blockDim.x)
    {
        const int offset = components * q;
        const double ux = quad[0][offset], uy = quad[0][offset + 1], uz = quad[0][offset + 2];
        const double weight = weighted_volume[q];
        #pragma unroll
        for (int c = 0; c < components; ++c)
        {
            const int t = offset + c;
            const double u = quad[0][t];
            const double dx = inverse_width * quad[1][t];
            const double dy = inverse_width * quad[2][t];
            const double dz = inverse_width * quad[3][t];
            quad[0][t] = 0.5 * weight * (ux * dx + uy * dy + uz * dz);
            quad[1][t] = weight * inverse_width * (viscosity * dx - 0.5 * u * ux);
            quad[2][t] = weight * inverse_width * (viscosity * dy - 0.5 * u * uy);
            quad[3][t] = weight * inverse_width * (viscosity * dz - 0.5 * u * uz);
        }
    }
    __syncthreads();

    // Transpose interpolation. Accumulate a flux when integrating along its axis.
    for (int t = lane; t < xy_size; t += blockDim.x)
    {
        const int qxy = t / (3 * components);
        const int iz = t / components % 3;
        const int component = t % components;
        double value = 0, dx = 0, dy = 0;
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const int from = (qxy * 4 + q) * components + component;
            value += basis[q][iz] * quad[0][from] + derivative[q][iz] * quad[3][from];
            dx += basis[q][iz] * quad[1][from];
            dy += basis[q][iz] * quad[2][from];
        }
        xy[0][t] = value;
        xy[1][t] = dx;
        xy[2][t] = dy;
    }
    __syncthreads();

    for (int t = lane; t < x_size; t += blockDim.x)
    {
        const int qx = t / (3 * 3 * components);
        const int iy = t / (3 * components) % 3;
        const int tail = t % (3 * components);
        double value = 0, dx = 0;
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const int from = (qx * 4 + q) * 3 * components + tail;
            value += basis[q][iy] * xy[0][from] + derivative[q][iy] * xy[2][from];
            dx += basis[q][iy] * xy[1][from];
        }
        x[0][t] = value;
        x[1][t] = dx;
    }
    __syncthreads();

    for (int t = lane; t < nodal_size; t += blockDim.x)
    {
        const int ix = t / (3 * 3 * components);
        const int tail = t % (3 * 3 * components);
        double value = 0;
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const int from = q * 3 * 3 * components + tail;
            value += basis[q][ix] * x[0][from] + derivative[q][ix] * x[1][from];
        }
        atomicAdd(residual + components * dofs[t / components] + t % components, value);
    }
}
