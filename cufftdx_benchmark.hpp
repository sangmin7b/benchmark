#pragma once
#include "cufftdx_kernels.hpp"
#include <algorithm>
#include <cstdio>
#include "arch_configs.hpp"

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

  printf("cuFFTDx with default FPB, EPT\n");
  if (smem) {
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 0, 0, SMNUM, true>(
        batch, comp_repeats, eval_repeats));
  } else {
    upd(benchmark_cufftdx_1d_batch_impl<N, T, 0, 0, SMNUM, false>(
        batch, comp_repeats, eval_repeats));
  }
  if (default_only)
    return best;

  run_arch_configs<N, T, SMNUM>(batch, smem, comp_repeats, eval_repeats, best);

  return best;
}
