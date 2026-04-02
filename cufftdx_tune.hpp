#include "benchmark_utils.hpp"
#include "cufftdx_benchmark.hpp"
#include <cstdio>
#include <cufftdx.hpp>

template <unsigned N, typename T, unsigned SMNUM, bool SMEM>
BenchmarkResult search_parameters(int batch) {
  BenchmarkResult best;
  auto upd = [&](BenchmarkResult r) { if (r < best) best = r; };
  // Test all configurations from (FFT Per Block in [1, 32], Elements Per Thread
  // in [2, 32])
  // https://docs.nvidia.com/cuda/cufftdx/api/operators.html#ept-operator-label

#ifdef SEARCH
  int comp_repeats = 150;
  int eval_repeats = 5;

  // Allocate shared device buffers, generate input, and compute the cufft
  // reference once for all parameter combinations to avoid repeated
  // cudaMalloc/generate/cufft/cudaFree overhead.
  T *d_input, *d_output;
  checkCuda(cudaMalloc(&d_input, sizeof(T) * N * batch), "search malloc input");
  checkCuda(cudaMalloc(&d_output, sizeof(T) * N * batch), "search malloc output");
  {
    T *h_input = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    generate_input_data(h_input, N, batch);
    checkCuda(cudaMemcpy(d_input, h_input, sizeof(T) * N * batch,
                         cudaMemcpyHostToDevice),
              "search upload input");
    free(h_input);
  }

  // Compute cufft reference output once.
  T *h_ref_output = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
  {
    T *d_ref_output;
    checkCuda(cudaMalloc(&d_ref_output, sizeof(T) * N * batch),
              "search malloc ref output");
    cufftHandle plan;
    if constexpr (std::is_same_v<T, float2>) {
      checkCUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, batch), "create plan");
      cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_ref_output,
                   CUFFT_FORWARD);
    } else if constexpr (std::is_same_v<T, double2>) {
      checkCUFFT(cufftPlan1d(&plan, N, CUFFT_Z2Z, batch), "create plan");
      cufftExecZ2Z(plan, (cufftDoubleComplex *)d_input,
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
      cufftXtExec(plan, (cufftComplex *)d_input, (cufftComplex *)d_ref_output,
                  CUFFT_FORWARD);
    }
    checkCuda(cudaMemcpy(h_ref_output, d_ref_output, sizeof(T) * N * batch,
                         cudaMemcpyDeviceToHost),
              "search copy ref output");
    cudaDeviceSynchronize();
    cufftDestroy(plan);
    cudaFree(d_ref_output);
  }

  if constexpr (!std::is_same_v<T, __half2>) {
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  }

  if constexpr (!std::is_same_v<T, __half2> && !std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats, d_input, d_output, h_ref_output));

  free(h_ref_output);
  cudaFree(d_input);
  cudaFree(d_output);
#endif
  return best;
}
