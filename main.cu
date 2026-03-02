#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "cufft_benchmark.hpp"
#include "cufftdx_benchmark.hpp"

#ifndef ARCHNUM
#define ARCHNUM 800
#endif

template <unsigned Arch, typename T>
float run_cufftdx_search_for_size(int n, int batch, bool smem) {
  switch (n) {
  case 64:
    return smem ? search_parameters<64, T, Arch, true>(batch)
                : search_parameters<64, T, Arch, false>(batch);
  case 128:
    return smem ? search_parameters<128, T, Arch, true>(batch)
                : search_parameters<128, T, Arch, false>(batch);
  case 256:
    return smem ? search_parameters<256, T, Arch, true>(batch)
                : search_parameters<256, T, Arch, false>(batch);
  case 512:
    return smem ? search_parameters<512, T, Arch, true>(batch)
                : search_parameters<512, T, Arch, false>(batch);
  case 1024:
    return smem ? search_parameters<1024, T, Arch, true>(batch)
                : search_parameters<1024, T, Arch, false>(batch);
  case 2048:
    return smem ? search_parameters<2048, T, Arch, true>(batch)
                : search_parameters<2048, T, Arch, false>(batch);
  case 4096:
    return smem ? search_parameters<4096, T, Arch, true>(batch)
                : search_parameters<4096, T, Arch, false>(batch);
  default:
    return 0.0f;
  }
}

template <unsigned Arch, typename T>
float run_cufftdx_benchmark_for_size(int n, int batch, bool smem, bool e2e,
                                     bool default_only) {
  switch (n) {
  case 64:
    return benchmark_cufftdx_1d_batch<64, T, Arch>(batch, smem, e2e,
                                                   default_only);
  case 128:
    return benchmark_cufftdx_1d_batch<128, T, Arch>(batch, smem, e2e,
                                                    default_only);
  case 256:
    return benchmark_cufftdx_1d_batch<256, T, Arch>(batch, smem, e2e,
                                                    default_only);
  case 512:
    return benchmark_cufftdx_1d_batch<512, T, Arch>(batch, smem, e2e,
                                                    default_only);
  case 1024:
    return benchmark_cufftdx_1d_batch<1024, T, Arch>(batch, smem, e2e,
                                                     default_only);
  case 2048:
    return benchmark_cufftdx_1d_batch<2048, T, Arch>(batch, smem, e2e,
                                                     default_only);
  case 4096:
    return benchmark_cufftdx_1d_batch<4096, T, Arch>(batch, smem, e2e,
                                                     default_only);
  default:
    return 0.0f;
  }
}

int main(int argc, char **argv) {
  constexpr unsigned int Arch = ARCHNUM;
  printf("arch: %d\n", Arch);

  enum class RunType { single, half, both };

  // parse args -N for size
  int N = 0;
  bool search = false;
  bool e2e = false;
  bool smem = false;
  bool default_only = false;
  RunType run_type = RunType::both;
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
    if (std::string(argv[i]) == "--smem") {
      smem = true;
    }
    if (std::string(argv[i]) == "--default") {
      default_only = true;
    }
    if (std::string(argv[i]) == "--type" && i + 1 < argc) {
      const std::string type_arg = argv[i + 1];
      if (type_arg == "single" || type_arg == "float" || type_arg == "fp32" ||
          type_arg == "0") {
        run_type = RunType::single;
      } else if (type_arg == "half" || type_arg == "fp16" || type_arg == "1") {
        run_type = RunType::half;
      } else if (type_arg == "all" || type_arg == "both" || type_arg == "2") {
        run_type = RunType::both;
      } else {
        printf("Invalid --type: %s (use: single, half, all)\n",
               type_arg.c_str());
        return 1;
      }
    }
  }

  std::vector<int> sizes;
  for (int i = 64; i <= 4096; i *= 2) {
    sizes.push_back(i);
  }

  std::vector<int> run_sizes;
  if (N > 0) {
    run_sizes.push_back(N);
  } else {
    run_sizes = sizes;
  }

  int batch = 65536;
  std::map<int, float> cufftdx_1d_batched_results;
  std::map<int, float> cufftdx_1d_batched_h_results;

  for (int n : sizes) {
    cufftdx_1d_batched_results[n] = 0.0f;
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
  const bool run_single = run_type == RunType::single || run_type == RunType::both;
  const bool run_half = run_type == RunType::half || run_type == RunType::both;
  for (int n : run_sizes) {
    printf("\n======== N %d ========\n", n);
    if (search) {
      if (run_single) {
        cufftdx_1d_batched_results[n] =
            run_cufftdx_search_for_size<Arch, float2>(n, batch, smem);
      }
      if (run_half) {
        cufftdx_1d_batched_h_results[n] =
            run_cufftdx_search_for_size<Arch, __half2>(n, batch, smem);
      }
    } else {
      if (run_single) {
        cufftdx_1d_batched_results[n] =
            run_cufftdx_benchmark_for_size<Arch, float2>(n, batch, smem, e2e,
                                                         default_only);
      }
      if (run_half) {
        cufftdx_1d_batched_h_results[n] =
            run_cufftdx_benchmark_for_size<Arch, __half2>(n, batch, smem, e2e,
                                                          default_only);
      }
    }
  }

  printf("Batch, N, "
         "cuFFTDx_1D_single, cuFFTDx_1D_half\n");
  for (size_t i = 0; i < run_sizes.size(); ++i) {
    const int n = run_sizes[i];
    printf("%d, %d, %.6f, %.6f\n", batch, n, cufftdx_1d_batched_results[n],
           cufftdx_1d_batched_h_results[n]);
  }

  return 0;
}
