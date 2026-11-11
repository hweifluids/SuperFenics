// Small audit probe. The included arithmetic is generated verbatim by FFCx.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "probe_kernel.inc"

void check(cudaError_t e, const char* operation)
{
  if (e != cudaSuccess)
  {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(e));
    std::exit(2);
  }
}

__global__ void cell_action(int count, const double* w, const double* x, double* out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  double local[10] = {};
  generated_action(local, w + 22 * i, nullptr, x + 12 * i, nullptr, nullptr, nullptr);
  for (int j = 0; j < 10; ++j)
    out[10 * i + j] = local[j];
}

void cpu_action(int count, const double* w, const double* x, double* out)
{
  for (int i = 0; i < count; ++i)
  {
    double local[10] = {};
    generated_action(local, w + 22 * i, nullptr, x + 12 * i, nullptr, nullptr, nullptr);
    std::copy_n(local, 10, out + 10 * i);
  }
}

double median(std::vector<double> times)
{
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

int main(int argc, char** argv)
{
  const int count = argc > 1 ? std::atoi(argv[1]) : 65536;
  const int device = argc > 2 ? std::atoi(argv[2]) : 1;
  const int repeats = argc > 3 ? std::atoi(argv[3]) : 7;
  if (count <= 0 || repeats <= 0)
    return 2;
  check(cudaSetDevice(device), "set device");
  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, device), "device properties");
  std::vector<double> w(22 * count), x(12 * count, 0.0), reference(10 * count), result(10 * count);
  for (int i = 0; i < count; ++i)
  {
    const double scale = 0.02 + 0.001 * (i % 7);
    x[12 * i + 3] = scale;
    x[12 * i + 7] = scale;
    x[12 * i + 11] = scale;
    for (int j = 0; j < 22; ++j)
      w[22 * i + j] = 0.1 + static_cast<double>((i * 7 + j * 3) % 31) / 31.0;
  }
  double *dw, *dx, *dy;
  check(cudaMalloc(&dw, w.size() * sizeof(double)), "allocate coefficients");
  check(cudaMalloc(&dx, x.size() * sizeof(double)), "allocate geometry");
  check(cudaMalloc(&dy, result.size() * sizeof(double)), "allocate result");
  check(cudaMemcpy(dw, w.data(), w.size() * sizeof(double), cudaMemcpyHostToDevice), "upload coefficients");
  check(cudaMemcpy(dx, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice), "upload geometry");
  for (int i = 0; i < 3; ++i)
    cell_action<<<(count + 127) / 128, 128>>>(count, dw, dx, dy);
  check(cudaDeviceSynchronize(), "warmup");
  check(cudaMemcpy(result.data(), dy, result.size() * sizeof(double), cudaMemcpyDeviceToHost), "download");
  cpu_action(count, w.data(), x.data(), reference.data());
  double max_error = 0.0, max_value = 0.0;
  for (std::size_t j = 0; j < result.size(); ++j)
  {
    if (!std::isfinite(result[j]) || !std::isfinite(reference[j]))
      return 3;
    max_error = std::max(max_error, std::abs(result[j] - reference[j]));
    max_value = std::max(max_value, std::abs(reference[j]));
  }
  std::vector<double> host_times, gpu_times, roundtrip_times;
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "create start event");
  check(cudaEventCreate(&stop), "create stop event");
  for (int r = 0; r < repeats; ++r)
  {
    auto t0 = std::chrono::steady_clock::now();
    cpu_action(count, w.data(), x.data(), reference.data());
    auto t1 = std::chrono::steady_clock::now();
    host_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    check(cudaEventRecord(start), "record start");
    for (int k = 0; k < 10; ++k)
      cell_action<<<(count + 127) / 128, 128>>>(count, dw, dx, dy);
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "synchronize stop");
    float milliseconds;
    check(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed");
    gpu_times.push_back(milliseconds / 10.0);
    t0 = std::chrono::steady_clock::now();
    check(cudaMemcpy(dw, w.data(), w.size() * sizeof(double), cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(dx, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice), "upload geometry");
    cell_action<<<(count + 127) / 128, 128>>>(count, dw, dx, dy);
    check(cudaMemcpy(result.data(), dy, result.size() * sizeof(double), cudaMemcpyDeviceToHost), "download");
    t1 = std::chrono::steady_clock::now();
    roundtrip_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  check(cudaGetLastError(), "last error");
  const bool pass = max_error <= 1e-11 * std::max(max_value, 1.0);
  std::printf("{\"device\":\"%s\",\"device_index\":%d,\"cells\":%d,\"repetitions\":%d,"
              "\"cpu_single_thread_median_ms\":%.9g,\"gpu_resident_median_ms\":%.9g,"
              "\"gpu_with_pageable_host_roundtrip_median_ms\":%.9g,"
              "\"max_abs_error\":%.12g,\"max_reference\":%.12g,\"pass\":%s}\n",
              prop.name, device, count, repeats, median(host_times), median(gpu_times),
              median(roundtrip_times), max_error, max_value, pass ? "true" : "false");
  check(cudaFree(dw), "free coefficients");
  check(cudaFree(dx), "free geometry");
  check(cudaFree(dy), "free result");
  check(cudaEventDestroy(start), "destroy start");
  check(cudaEventDestroy(stop), "destroy stop");
  return pass ? 0 : 1;
}
