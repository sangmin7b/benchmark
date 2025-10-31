#include "benchmark_utils.hpp"
#include <cstdio>

struct Cufft2DContext {
  cufftHandle plan;
  float2 *input;
  float2 *output;
};

void cufft_2d_run(void *user_data) {
  Cufft2DContext *ctx = (Cufft2DContext *)user_data;
  cufftExecC2C(ctx->plan, (cufftComplex *)ctx->input, (cufftComplex *)ctx->output, CUFFT_FORWARD);
}

void benchmark_cufft_2d(int N) {
  printf("[cuFFT] 2D FFT: N = %d x %d\n", N, N);
  Cufft2DContext ctx;
  size_t total = size_t(N) * N;
  checkCuda(cudaMalloc(&ctx.input, sizeof(float2) * total), "malloc input");
  checkCuda(cudaMalloc(&ctx.output, sizeof(float2) * total), "malloc output");

  checkCUFFT(cufftPlan2d(&ctx.plan, N, N, CUFFT_C2C), "plan2d");
  float avg = measure_kernel_time(cufft_2d_run, &ctx);
  printf("Average time: %.3f ms\n", avg);

  cufftDestroy(ctx.plan);
  cudaFree(ctx.input);
  cudaFree(ctx.output);
}
