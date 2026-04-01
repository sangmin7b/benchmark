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
  if constexpr (!std::is_same_v<T, __half2>) {
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  }

  if constexpr (!std::is_same_v<T, __half2> && !std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 1, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 2, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 4, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 8, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 16, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));

  upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 2, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 4, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 8, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 16, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
  if constexpr (!std::is_same_v<T, double2>)
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 32, 32, SMNUM, SMEM>(batch, comp_repeats, eval_repeats));
#endif
  return best;
}
