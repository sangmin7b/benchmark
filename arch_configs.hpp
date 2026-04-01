#pragma once
#include "configs_a100.hpp"
#include "configs_h100.hpp"
#include "configs_b200.hpp"

template <unsigned N, typename T, unsigned SMNUM>
void run_arch_configs(int batch, bool smem, int comp_repeats, int eval_repeats,
                      BenchmarkResult &best) {
  if constexpr (SMNUM == 800) {
    run_a100_configs<N, T, SMNUM>(batch, smem, comp_repeats, eval_repeats,
                                  best);
  }
  else if constexpr (SMNUM == 900) {
    run_h100_configs<N, T, SMNUM>(batch, smem, comp_repeats, eval_repeats, best);
  }
  else if constexpr (SMNUM == 1000) {
    run_b200_configs<N, T, SMNUM>(batch, smem, comp_repeats, eval_repeats, best);
  }
}
