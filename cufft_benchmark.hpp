#include "benchmark_utils.hpp"
#include <cstdio>

template <typename T> struct Cufft1DBatchContext {
  cufftHandle plan;
  T *input;
  T *output;
};

template <typename T> struct Cufft1DContext {
  cufftHandle plan;
  T *input;
  T *output;
};

template <typename T> void cufft_1d_run(void *user_data);

template <> void cufft_1d_run<float2>(void *user_data) {
  auto *ctx = (Cufft1DContext<float2> *)user_data;
  cufftExecC2C(ctx->plan, (cufftComplex *)ctx->input, (cufftComplex *)ctx->output, CUFFT_FORWARD);
}

template <> void cufft_1d_run<double2>(void *user_data) {
  auto *ctx = (Cufft1DContext<double2> *)user_data;
  cufftExecZ2Z(ctx->plan, (cufftDoubleComplex *)ctx->input, (cufftDoubleComplex *)ctx->output,
               CUFFT_FORWARD);
}

template <typename T> float benchmark_cufft_1d(int N) {
  printf("[cuFFT] 1D Single FFT: N = %d\n", N);
  Cufft1DContext<T> ctx;
  size_t total = N;
  checkCuda(cudaMalloc(&ctx.input, sizeof(T) * total), "malloc input");
  checkCuda(cudaMalloc(&ctx.output, sizeof(T) * total), "malloc output");
  // C2C: single precision
  // Z2Z: double precision
  checkCUFFT(cufftPlan1d(&ctx.plan, N, DataTypeToCufftType<T>::type, 1), "plan1d");
  float avg = measure_kernel_time(cufft_1d_run<T>, &ctx);
  printf("Average time: %.6f ms\n", avg);

  cufftDestroy(ctx.plan);
  cudaFree(ctx.input);
  cudaFree(ctx.output);
  return avg;
}

template <typename T> void cufft_1d_batch_run(void *user_data);

template <> void cufft_1d_batch_run<float2>(void *user_data) {
  auto *ctx = (Cufft1DBatchContext<float2> *)user_data;
  checkCUFFT(cufftExecC2C(ctx->plan, (cufftComplex *)ctx->input, (cufftComplex *)ctx->output,
                          CUFFT_FORWARD),
             "cufft 1d float2");
}

template <> void cufft_1d_batch_run<double2>(void *user_data) {
  auto *ctx = (Cufft1DBatchContext<double2> *)user_data;
  checkCUFFT(cufftExecZ2Z(ctx->plan, (cufftDoubleComplex *)ctx->input,
                          (cufftDoubleComplex *)ctx->output, CUFFT_FORWARD),
             "cufft 1d double2");
}

template <> void cufft_1d_batch_run<__half2>(void *user_data) {
  auto *ctx = (Cufft1DBatchContext<__half2> *)user_data;
  checkCUFFT(cufftXtExec(ctx->plan, ctx->input, ctx->output, CUFFT_FORWARD), "cufft 1d half2");
}

template <typename T> float benchmark_cufft_1d_batch(int N, int batch) {
  printf("[cuFFT] 1D Batch FFT: N = %d, batch = %d\n", N, batch);
  size_t total = size_t(N) * batch;
  Cufft1DBatchContext<T> ctx;
  checkCuda(cudaMalloc(&ctx.input, sizeof(T) * total), "malloc input");
  checkCuda(cudaMalloc(&ctx.output, sizeof(T) * total), "malloc output");

  checkCUFFT(cufftPlan1d(&ctx.plan, N, DataTypeToCufftType<T>::type, batch), "plan1d batch");
  float avg = measure_kernel_time(cufft_1d_batch_run<T>, &ctx);
  printf("Average time: %.6f ms\n", avg);

  cufftDestroy(ctx.plan);
  cudaFree(ctx.input);
  cudaFree(ctx.output);
  return avg;
}

template <> float benchmark_cufft_1d_batch<__half2>(int N, int batch) {
  printf("[cuFFT] 1D Batch FFT half: N = %d, batch = %d\n", N, batch);
  size_t total = size_t(N) * batch;
  Cufft1DBatchContext<__half2> ctx;
  checkCuda(cudaMalloc(&ctx.input, sizeof(__half2) * total), "malloc input");
  checkCuda(cudaMalloc(&ctx.output, sizeof(__half2) * total), "malloc output");

  long long n = N; // size per transform
  long long inembed[1] = {N};
  long long onembed[1] = {N};
  long long istride = 1; // contiguous
  long long ostride = 1;
  long long idist = N; // distance between consecutive FFTs
  long long odist = N;
  size_t workSize = 0;
  checkCUFFT(cufftCreate(&ctx.plan), "create plan");
  checkCUFFT(cufftXtMakePlanMany(ctx.plan, 1, &n, inembed, istride, idist, CUDA_C_16F, onembed,
                                 ostride, odist, CUDA_C_16F, batch, &workSize, CUDA_C_16F),
             "plan1d batch");
  float avg = measure_kernel_time(cufft_1d_batch_run<__half2>, &ctx);
  printf("Average time: %.6f ms\n", avg);

  cufftDestroy(ctx.plan);
  cudaFree(ctx.input);
  cudaFree(ctx.output);
  return avg;
}

template <typename T> struct Cufft2DContext {
  cufftHandle plan;
  T *input;
  T *output;
};

template <typename T> void cufft_2d_run(void *user_data);

template <> void cufft_2d_run<float2>(void *user_data) {
  Cufft2DContext<float2> *ctx = (Cufft2DContext<float2> *)user_data;
  cufftExecC2C(ctx->plan, (cufftComplex *)ctx->input, (cufftComplex *)ctx->output, CUFFT_FORWARD);
}

template <> void cufft_2d_run<double2>(void *user_data) {
  Cufft2DContext<double2> *ctx = (Cufft2DContext<double2> *)user_data;
  cufftExecZ2Z(ctx->plan, (cufftDoubleComplex *)ctx->input, (cufftDoubleComplex *)ctx->output,
               CUFFT_FORWARD);
}

template <typename T> float benchmark_cufft_2d(int N) {
  printf("[cuFFT] 2D FFT: N = %d x %d\n", N, N);
  Cufft2DContext<T> ctx;
  size_t total = size_t(N) * N;
  checkCuda(cudaMalloc(&ctx.input, sizeof(T) * total), "malloc input");
  checkCuda(cudaMalloc(&ctx.output, sizeof(T) * total), "malloc output");

  checkCUFFT(cufftPlan2d(&ctx.plan, N, N, DataTypeToCufftType<T>::type), "plan2d");
  float avg = measure_kernel_time(cufft_2d_run<T>, &ctx);
  printf("Average time: %.6f ms\n", avg);

  cufftDestroy(ctx.plan);
  cudaFree(ctx.input);
  cudaFree(ctx.output);
  return avg;
}