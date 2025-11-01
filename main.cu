#include <cstdio>
#include <map>
#include <vector>

#include "cufft_benchmark.hpp"
#include "cufftdx_benchmark.hpp"

#ifndef ARCHNUM
#define ARCHNUM 890
#endif

int main(int argc, char **argv) {
  constexpr unsigned int Arch = ARCHNUM;
  printf("arch: %d\n", Arch);

  // parse args -N for size
  int N = 0;
  bool search = false;
  bool e2e = false;
  int type = 0; // 0: single, 1: double, 2: half
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-N" && i + 1 < argc) {
      N = atoi(argv[i + 1]);
    }
    if (std::string(argv[i]) == "--search") {
      search = true;
    }
    if (std::string(argv[i]) == "--e2e") {
      e2e = true;
    }
    if (std::string(argv[i]) == "--type" && i + 1 < argc) {
      type = atoi(argv[i + 1]);
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
  std::map<int, float> cufft_1d_batched_results;
  std::map<int, float> cufft_1d_batched_d_results;
  std::map<int, float> cufft_1d_batched_h_results;
  std::map<int, float> cufftdx_1d_batched_results;
  std::map<int, float> cufftdx_1d_batched_d_results;
  std::map<int, float> cufftdx_1d_batched_h_results;

  for (int n : sizes) {
    cufft_1d_batched_results[n] = 0.0f;
    cufft_1d_batched_d_results[n] = 0.0f;
    cufft_1d_batched_h_results[n] = 0.0f;
    cufftdx_1d_batched_results[n] = 0.0f;
    cufftdx_1d_batched_d_results[n] = 0.0f;
    cufftdx_1d_batched_h_results[n] = 0.0f;
  }

  // Supported Functionality
  // C2C
  // float: N [2, 24389] (70, 86, 89)
  // float: N [2, 32768] (90)
  // double: N [2, 12167] (70, 86, 89)
  // double: N [2, 16384] (90)
  // https://docs.nvidia.com/cuda/cufftdx/requirements_func.html#functionality-label
  printf("\n======== cuFFTDx 1D (batch) ========\n");
  if (search) {
    if (type == 0) {
      switch (N) {
      case 64:
        cufftdx_1d_batched_results[64] =
            search_parameters<64, float2, Arch>(batch);
        break;
      case 128:
        cufftdx_1d_batched_results[128] =
            search_parameters<128, float2, Arch>(batch);
        break;
      case 256:
        cufftdx_1d_batched_results[256] =
            search_parameters<256, float2, Arch>(batch);
        break;
      case 512:
        cufftdx_1d_batched_results[512] =
            search_parameters<512, float2, Arch>(batch);
        break;
      case 1024:
        cufftdx_1d_batched_results[1024] =
            search_parameters<1024, float2, Arch>(batch);
        break;
      case 2048:
        cufftdx_1d_batched_results[2048] =
            search_parameters<2048, float2, Arch>(batch);
        break;
      case 4096:
        cufftdx_1d_batched_results[4096] =
            search_parameters<4096, float2, Arch>(batch);
        break;
      }
    } else if (type == 1) {
switch (N) {
      case 64:
        cufftdx_1d_batched_h_results[64] =
            search_parameters<64, __half2, Arch>(batch);
        break;
      case 128:
        cufftdx_1d_batched_h_results[128] =
            search_parameters<128, __half2, Arch>(batch);
        break;
      case 256:
        cufftdx_1d_batched_h_results[256] =
            search_parameters<256, __half2, Arch>(batch);
        break;
      case 512:
        cufftdx_1d_batched_h_results[512] =
            search_parameters<512, __half2, Arch>(batch);
        break;
      case 1024:
        cufftdx_1d_batched_h_results[1024] =
            search_parameters<1024, __half2, Arch>(batch);
        break;
      case 2048:
        cufftdx_1d_batched_h_results[2048] =
            search_parameters<2048, __half2, Arch>(batch);
        break;
      case 4096:
        cufftdx_1d_batched_h_results[4096] =
            search_parameters<4096, __half2, Arch>(batch);
        break;
      }
    }
  } else {
    if (type == 0) {
      switch (N) {
      case 64:
        cufftdx_1d_batched_results[64] =
            benchmark_cufftdx_1d_batch<64, float2, Arch>(batch, e2e);
        break;
      case 128:
        cufftdx_1d_batched_results[128] =
            benchmark_cufftdx_1d_batch<128, float2, Arch>(batch, e2e);
        break;
      case 256:
        cufftdx_1d_batched_results[256] =
            benchmark_cufftdx_1d_batch<256, float2, Arch>(batch, e2e);
        break;
      case 512:
        cufftdx_1d_batched_results[512] =
            benchmark_cufftdx_1d_batch<512, float2, Arch>(batch, e2e);
        break;
      case 1024:
        cufftdx_1d_batched_results[1024] =
            benchmark_cufftdx_1d_batch<1024, float2, Arch>(batch, e2e);
        break;
      case 2048:
        cufftdx_1d_batched_results[2048] =
            benchmark_cufftdx_1d_batch<2048, float2, Arch>(batch, e2e);
        break;
      case 4096:
        cufftdx_1d_batched_results[4096] =
            benchmark_cufftdx_1d_batch<4096, float2, Arch>(batch, e2e);
        break;
      }
    }
    if (type == 1) {
      switch (N) {
      case 64:
        cufftdx_1d_batched_h_results[64] =
            benchmark_cufftdx_1d_batch<64, __half2, Arch>(batch, e2e);
        break;
      case 128:
        cufftdx_1d_batched_h_results[128] =
            benchmark_cufftdx_1d_batch<128, __half2, Arch>(batch, e2e);
        break;
      case 256:
        cufftdx_1d_batched_h_results[256] =
            benchmark_cufftdx_1d_batch<256, __half2, Arch>(batch, e2e);
        break;
      case 512:
        cufftdx_1d_batched_h_results[512] =
            benchmark_cufftdx_1d_batch<512, __half2, Arch>(batch, e2e);
        break;
      case 1024:
        cufftdx_1d_batched_h_results[1024] =
            benchmark_cufftdx_1d_batch<1024, __half2, Arch>(batch, e2e);
        break;
      case 2048:
        cufftdx_1d_batched_h_results[2048] =
            benchmark_cufftdx_1d_batch<2048, __half2, Arch>(batch, e2e);
        break;
      case 4096:
        cufftdx_1d_batched_h_results[4096] =
            benchmark_cufftdx_1d_batch<4096, __half2, Arch>(batch, e2e);
        break;
      }
    }
  }

  printf("Batch, N, cuFFT_1D_half, cuFFT_1D_single, cuFFT_1D_double,  "
         "cuFFTDx_1D_half, cuFFTDx_1D_single, "
         "cuFFTDx_1D_double\n");
  for (size_t i = 0; i < sizes.size(); ++i) {
    printf("%d, %d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n", batch, sizes[i],
           cufft_1d_batched_h_results[sizes[i]],
           cufft_1d_batched_results[sizes[i]],
           cufft_1d_batched_d_results[sizes[i]],
           cufftdx_1d_batched_h_results[sizes[i]],
           cufftdx_1d_batched_results[sizes[i]],
           cufftdx_1d_batched_d_results[sizes[i]]);
  }

  return 0;
}
