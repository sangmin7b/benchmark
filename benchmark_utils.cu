// benchmark_utils.cu
#include "benchmark_utils.hpp"
#include <cstdio>
#include <stdexcept>

void checkCuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    printf("CUDA Error: %s (%s)\n", msg, cudaGetErrorString(err));
    throw std::runtime_error("CUDA operation failed");
  }
}

void checkCUFFT(cufftResult res, const char *msg) {
  if (res != CUFFT_SUCCESS) {
    printf("CUFFT Error: %s (code %d)\n", msg, res);
    throw std::runtime_error("CUFFT operation failed");
  }
}

float measure_kernel_time(BenchmarkFunc func, void *user_data, int repeat) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float warmup = 0;
  for (int i = 0; i < repeat && warmup < 100; ++i) {
    cudaEventRecord(start);
    func(user_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    warmup += ms;
  }

  float total = 0;
  for (int i = 0; i < repeat; ++i) {
    cudaEventRecord(start);
    func(user_data); // 함수 호출
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    total += ms;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return total / repeat;
}

bool validate_output(float2 *ref, float2 *target, int size, const char *tag, float atol) {
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    float diff_x = fabs(ref[i].x - target[i].x);
    float diff_y = fabs(ref[i].y - target[i].y);
    if (diff_x > atol || diff_y > atol) {
      if (++errors <= 5)
        printf("[Mismatch:%s] idx=%d ref=(%.3f, %.3f) target=(%.3f, %.3f)\n", tag, i, ref[i].x, ref[i].y, target[i].x, target[i].y);
    }
  }
  if (errors == 0) {
    printf("[%s] Validation PASSED\n", tag);
    return true;
  } else {
    printf("[%s] Validation FAILED with %d mismatches\n", tag, errors);
    return false;
  }
}

bool validate_output(double2 *ref, double2 *target, int size, const char *tag, float atol) {
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    float diff_x = fabs(ref[i].x - target[i].x);
    float diff_y = fabs(ref[i].y - target[i].y);
    if (diff_x > atol || diff_y > atol) {
      if (++errors <= 5)
        printf("[Mismatch:%s] idx=%d ref=(%.3f, %.3f) target=(%.3f, %.3f)\n", tag, i, ref[i].x, ref[i].y, target[i].x, target[i].y);
    }
  }
  if (errors == 0) {
    printf("[%s] Validation PASSED\n", tag);
    return true;
  } else {
    printf("[%s] Validation FAILED with %d mismatches\n", tag, errors);
    return false;
  }
}


bool validate_output(__half2 *ref, __half2 *target, int size, const char *tag, float atol) {
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    float diff_x = fabs(__half2float(ref[i].x) - __half2float(target[i].x));
    float diff_y = fabs(__half2float(ref[i].y) - __half2float(target[i].y));
    if (diff_x > atol || diff_y > atol) {
      if (++errors <= 5)
        printf("[Mismatch:%s] idx=%d ref=(%.3f, %.3f) target=(%.3f, %.3f)\n", tag, i, __half2float(ref[i].x), __half2float(ref[i].y), __half2float(target[i].x), __half2float(target[i].y));
    }
  }
  if (errors == 0) {
    printf("[%s] Validation PASSED\n", tag);
    return true;
  } else {
    printf("[%s] Validation FAILED with %d mismatches\n", tag, errors);
    return false;
  }
}

void generate_input_data(float2 *data, int size, int batch, int seed) {
  srand(seed);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < size; ++i) {
      data[b * size + i] = make_float2(rand() * 1.0 / RAND_MAX, rand() * 1.0 / RAND_MAX);
    }
  }
}

void generate_input_data(double2 *data, int size, int batch, int seed) {
  srand(seed);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < size; ++i) {
      data[b * size + i] = make_double2(rand() * 1.0 / RAND_MAX, rand() * 1.0 / RAND_MAX);
    }
  }
}

void generate_input_data(__half2 *data, int size, int batch, int seed) {
  srand(seed);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < size; ++i) {
      data[b * size + i] = make_half2( rand() * 1.0 / RAND_MAX, rand() * 1.0 / RAND_MAX);
    }
  }
}

