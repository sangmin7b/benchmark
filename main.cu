#include <vector>
#include <cstdio>

#include "cufftdx_benchmark.hpp"
#include "cufft_benchmark.hpp"

#ifndef ARCHNUM
#define ARCHNUM 890
#endif

int main(int argc, char **argv) {
  constexpr unsigned int Arch = ARCHNUM;
  printf("arch: %d\n", Arch);

  // parse args -N for size
  int N = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-N" && i + 1 < argc) {
      N = atoi(argv[i + 1]);
    }
  }
  // check regsPerBlock
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("regsPerBlock limit = %d\n", p.regsPerBlock);

  std::vector<int> sizes;
  sizes.push_back(64);
  // sizes.push_back(256);
  // sizes.push_back(1024);
  sizes.push_back(4096);
  
  int batch = 65536;
  // printf("\n======== cuFFT 1D (batch) ========\n");
  std::vector<float> cufft_1d_batched_results;
  std::vector<float> cufft_1d_batched_d_results;
  std::vector<float> cufft_1d_batched_h_results;

  // for (int N : sizes) {
  //   cufft_1d_batched_results.push_back(benchmark_cufft_1d_batch<float2>(N, batch));
  //   cufft_1d_batched_h_results.push_back(benchmark_cufft_1d_batch<__half2>(N, batch));
  //   if (N < 16384)
  //     cufft_1d_batched_d_results.push_back(benchmark_cufft_1d_batch<double2>(N, batch));
  //   else
  //     cufft_1d_batched_d_results.push_back(0.0f);
  // }

  // printf("\n======== cuFFT 2D (NxN) ========\n");
  std::vector<float> cufft_2d_batched_results;
  // for (int N : sizes)
  //     cufft_2d_batched_results.push_back(benchmark_cufft_2d<float2>(N));

  // Supported Functionality
  // C2C
  // float: N [2, 24389] (70, 86, 89)
  // float: N [2, 32768] (90)
  // double: N [2, 12167] (70, 86, 89)
  // double: N [2, 16384] (90)
  // https://docs.nvidia.com/cuda/cufftdx/requirements_func.html#functionality-label
  printf("\n======== cuFFTDx 1D (batch) ========\n");

  std::vector<float> cufftdx_1d_batched_results;
  if (N == 64) {
    // cufftdx_1d_batched_results.push_back(benchmark_cufftdx_1d_batch<64, float2, Arch>(batch));
  }
  // cufftdx_1d_batched_results.push_back(benchmark_cufftdx_1d_batch<128, float2, Arch>(batch));
  if (N == 256){
    // cufftdx_1d_batched_results.push_back(benchmark_cufftdx_1d_batch<256, float2, Arch>(batch));
  }
  // cufftdx_1d_batched_results.push_back(benchmark_cufftdx_1d_batch<512, float2, Arch>(batch));
  if (N == 1024){
    // cufftdx_1d_batched_results.push_back(benchmark_cufftdx_1d_batch<1024, float2, Arch>(batch));
  }
  // cufftdx_1d_batched_results.push_back(benchmark_cufftdx_1d_batch<2048, float2, Arch>(batch));
  if (N == 4096){
    // cufftdx_1d_batched_results.push_back(benchmark_cufftdx_1d_batch<4096, float2, Arch>(batch));
  }
  // cufftdx_1d_batched_results.push_back(benchmark_cufftdx_1d_batch<8192, float2, Arch>(batch));
  // cufftdx_1d_batched_results.push_back(benchmark_cufftdx_1d_batch<16384, float2, Arch>(batch));

  std::vector<float> cufftdx_1d_batched_d_results;
  // cufftdx_1d_batched_d_results.push_back(benchmark_cufftdx_1d_batch<64, double2, Arch>(batch));
  // cufftdx_1d_batched_d_results.push_back(benchmark_cufftdx_1d_batch<128, double2, Arch>(batch));
  // cufftdx_1d_batched_d_results.push_back(benchmark_cufftdx_1d_batch<256, double2, Arch>(batch));
  // cufftdx_1d_batched_d_results.push_back(benchmark_cufftdx_1d_batch<512, double2, Arch>(batch));
  // cufftdx_1d_batched_d_results.push_back(benchmark_cufftdx_1d_batch<1024, double2, Arch>(batch));
  // cufftdx_1d_batched_d_results.push_back(benchmark_cufftdx_1d_batch<2048, double2, Arch>(batch));
  // cufftdx_1d_batched_d_results.push_back(benchmark_cufftdx_1d_batch<4096, double2, Arch>(batch));
  // cufftdx_1d_batched_d_results.push_back(benchmark_cufftdx_1d_batch<8192, double2, Arch>(batch));
  // if constexpr (Arch != 900) {
  //   cufft_1d_batched_d_results.push_back(0.0f);
  // } else {
  //   cufftdx_1d_batched_d_results.push_back(benchmark_cufftdx_1d_batch<16384, double2, Arch>(batch));
  // }

  std::vector<float> cufftdx_1d_batched_h_results;
  if (N == 64){
    cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<64, __half2, Arch>(batch));
  }
  // cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<128, __half2, Arch>(batch));
  if (N == 256){
    cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<256, __half2, Arch>(batch));
  }
  // cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<256, __half2, Arch>(batch));
  // cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<512, __half2, Arch>(batch));
  if (N == 1024){
    cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<1024, __half2, Arch>(batch));
  }
  // cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<1024, __half2, Arch>(batch));
  // cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<2048, __half2, Arch>(batch));
  if (N == 4096){
    cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<4096, __half2, Arch>(batch));
  }
  // cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<8192, __half2, Arch>(batch));
  // cufftdx_1d_batched_h_results.push_back(benchmark_cufftdx_1d_batch<16384, __half2, Arch>(batch));

  for (size_t i = 0; i < sizes.size(); ++i) {
    if (cufft_1d_batched_h_results.size() < sizes.size()) {
      cufft_1d_batched_h_results.push_back(0.0f);
    }
    if (cufft_1d_batched_results.size() < sizes.size()) {
      cufft_1d_batched_results.push_back(0.0f);
    }
    if (cufft_1d_batched_d_results.size() < sizes.size()) {
      cufft_1d_batched_d_results.push_back(0.0f);
    }
    if (cufftdx_1d_batched_h_results.size() < sizes.size()) {
      cufftdx_1d_batched_h_results.push_back(0.0f);
    }
    if (cufftdx_1d_batched_results.size() < sizes.size()) {
      cufftdx_1d_batched_results.push_back(0.0f);
    }
    if (cufftdx_1d_batched_d_results.size() < sizes.size()) {
      cufftdx_1d_batched_d_results.push_back(0.0f);
    }
  }

  printf("Batch, N, cuFFT_1D_half, cuFFT_1D_single, cuFFT_1D_double,  cuFFTDx_1D_half, cuFFTDx_1D_single, "
         "cuFFTDx_1D_double\n");
  for (size_t i = 0; i < sizes.size(); ++i) {
    printf("%d, %d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n", batch, sizes[i],
           cufft_1d_batched_h_results[i], cufft_1d_batched_results[i], cufft_1d_batched_d_results[i], cufftdx_1d_batched_h_results[i],
            cufftdx_1d_batched_results[i],
           cufftdx_1d_batched_d_results[i]);
  }

  return 0;
}
