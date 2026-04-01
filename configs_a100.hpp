#pragma once
#include "cufftdx_benchmark.hpp"

// A100 (SM 80) optimized launch configurations

template <unsigned N, typename T, unsigned SMNUM>
void run_a100_configs(int batch, bool smem, int comp_repeats, int eval_repeats,
                      BenchmarkResult &best) {
  auto upd = [&](BenchmarkResult r) { if (r < best) best = r; };

  // single precision
  if constexpr (N == 64 && std::is_same_v<T, float2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 16, SMNUM>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 32, SMNUM>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 64, SMNUM>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 128, SMNUM>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 256, SMNUM>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 128 && std::is_same_v<T, float2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 8, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 256 && std::is_same_v<T, float2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 8, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 512 && std::is_same_v<T, float2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 8, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 8, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 1024 && std::is_same_v<T, float2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 4, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 32, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 2048 && std::is_same_v<T, float2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 4, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 4096 && std::is_same_v<T, float2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 32, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  // half precision
  if constexpr (N == 64 && std::is_same_v<T, __half2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 8, SMNUM, true>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 8, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 16, SMNUM>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 32, SMNUM>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 64, SMNUM>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 128, SMNUM>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_thread_impl<N, T, 256, SMNUM>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 128 && std::is_same_v<T, __half2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 16, SMNUM, true>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 16, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 32, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 256 && std::is_same_v<T, __half2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 8, SMNUM, true>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 512 && std::is_same_v<T, __half2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, true>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 8, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 1024 && std::is_same_v<T, __half2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, true>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 2048 && std::is_same_v<T, __half2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 32, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 4096 && std::is_same_v<T, __half2>) {
    if (smem) {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 32, SMNUM, true>(batch, comp_repeats, eval_repeats));
    } else {
      upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, false>(batch, comp_repeats, eval_repeats));
    }
  }
}
