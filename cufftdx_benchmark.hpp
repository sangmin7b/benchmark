#pragma once
#include "cufftdx_kernels.hpp"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "arch_configs.hpp"

// Allocate device buffers, generate input data, and compute the cufft
// reference output. Caller is responsible for freeing d_input, d_output,
// and h_ref_output.
template <unsigned N, typename T>
void alloc_benchmark_buffers(int batch,
                             T** d_input, T** d_output, T** h_ref_output) {
  checkCuda(cudaMalloc(d_input, sizeof(T) * N * batch), "malloc input");
  checkCuda(cudaMalloc(d_output, sizeof(T) * N * batch), "malloc output");

  T *h_input = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
  generate_input_data(h_input, N, batch);
  checkCuda(cudaMemcpy(*d_input, h_input, sizeof(T) * N * batch,
                       cudaMemcpyHostToDevice),
            "upload input");
  free(h_input);

  T *d_ref_output;
  checkCuda(cudaMalloc(&d_ref_output, sizeof(T) * N * batch),
            "malloc ref output");
  cufftHandle plan;
  if constexpr (std::is_same_v<T, float2>) {
    checkCUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, batch), "create plan");
    cufftExecC2C(plan, (cufftComplex *)*d_input,
                 (cufftComplex *)d_ref_output, CUFFT_FORWARD);
  } else if constexpr (std::is_same_v<T, double2>) {
    checkCUFFT(cufftPlan1d(&plan, N, CUFFT_Z2Z, batch), "create plan");
    cufftExecZ2Z(plan, (cufftDoubleComplex *)*d_input,
                 (cufftDoubleComplex *)d_ref_output, CUFFT_FORWARD);
  } else if constexpr (std::is_same_v<T, __half2>) {
    long long n = N;
    long long inembed[1] = {N};
    long long onembed[1] = {N};
    long long istride = 1;
    long long ostride = 1;
    long long idist = N;
    long long odist = N;
    size_t workSize = 0;
    checkCUFFT(cufftCreate(&plan), "create plan");
    checkCUFFT(cufftXtMakePlanMany(plan, 1, &n, inembed, istride, idist,
                                   CUDA_C_16F, onembed, ostride, odist,
                                   CUDA_C_16F, batch, &workSize, CUDA_C_16F),
               "plan1d batch");
    cufftXtExec(plan, (cufftComplex *)*d_input, (cufftComplex *)d_ref_output,
                CUFFT_FORWARD);
  }
  *h_ref_output = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
  checkCuda(cudaMemcpy(*h_ref_output, d_ref_output, sizeof(T) * N * batch,
                       cudaMemcpyDeviceToHost),
            "copy ref output to host");
  cudaDeviceSynchronize();
  cufftDestroy(plan);
  cudaFree(d_ref_output);
}

template <unsigned N, typename T, unsigned SMNUM>
BenchmarkResult benchmark_cufftdx_1d_batch(int batch, bool smem = false,
                                           bool e2e = false,
                                           bool default_only = false) {
  BenchmarkResult best;
  auto upd = [&](BenchmarkResult r) { if (r < best) best = r; };

  int comp_repeats = 10000;
  int eval_repeats = 5;

  if (e2e) {
    comp_repeats = 1;
    eval_repeats = 10000;
  }

  T *d_input, *d_output, *h_ref_output;
  alloc_benchmark_buffers<N, T>(batch, &d_input, &d_output, &h_ref_output);

  printf("cuFFTDx with default FPB, EPT\n");
  if (smem) {
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 0, 0, SMNUM, true>(
        batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  } else {
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 0, 0, SMNUM, false>(
        batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  }
  if (!default_only) {
    run_arch_configs<N, T, SMNUM>(batch, smem, comp_repeats, eval_repeats,
                                  best, d_input, d_output, h_ref_output);
  }

  free(h_ref_output);
  cudaFree(d_input);
  cudaFree(d_output);

  return best;
}
