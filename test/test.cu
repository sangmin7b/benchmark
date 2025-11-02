#include <vector>
#include <cstdio>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cufftdx.hpp>

using namespace cufftdx;

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

bool validate_output(float *ref, float *target, int size, const char *tag, float atol = 1e-3f) {
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    float diff = fabs(ref[i] - target[i]);
    if (diff > atol) {
      if (++errors <= 5)
        printf("[Mismatch:%s] idx=%d ref=%.3f target=%.3f\n", tag, i, ref[i], target[i]);
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

template <class FFT, typename T>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void cufftdx_batch_kernel(T *input, T *output, int batch) {
  using complex_type = typename FFT::value_type;

  constexpr int FFTsPerBlk = FFT::ffts_per_block;
  constexpr int FFTSize = cufftdx::size_of<FFT>::value;
  constexpr int Stride = FFT::stride;

  int global_fft_id = (blockIdx.x * FFTsPerBlk + +threadIdx.y) / FFT::implicit_type_batching;
  if (global_fft_id >= batch)
    return;
  extern __shared__ __align__(alignof(float4)) void *smem;
  complex_type thread_data[FFT::elements_per_thread];

  // Custom load
  unsigned int offset = global_fft_id * FFTSize;
  unsigned int index = offset + threadIdx.x;

  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      thread_data[i] = reinterpret_cast<complex_type *>(input)[index];
    }
    index += Stride;
  }

  // Execute
  // for (int i = 0; i < 10000; i++){
  FFT().execute(thread_data, smem);
  // }

  // Custom store
  index = offset + threadIdx.x;
  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      reinterpret_cast<complex_type *>(output)[index] = thread_data[i];
    }
    index += Stride;
  }
}


int main() {
    printf("FFT Benchmark Test\n");
    // float version cufft and cufftdx 
    const int N = 1024;
    const int batch = 1024 * 64;
    const size_t total_size = N * batch;
    cufftComplex *d_input, *d_output;
    checkCuda(cudaMalloc(&d_input, sizeof(cufftComplex) * total_size), "malloc ref input");
    checkCuda(cudaMalloc(&d_output, sizeof(cufftComplex) * total_size), "malloc ref output");

    // Generate input data
    cufftComplex *h_ref_input, *h_ref_output;
    h_ref_input = (cufftComplex *)malloc(sizeof(cufftComplex) * total_size);
    h_ref_output = (cufftComplex *)malloc(sizeof(cufftComplex) * total_size);
    for (int i = 0; i < total_size; ++i) {
        h_ref_input[i].x = (static_cast<float>(i % N)) / N;
        h_ref_input[i].y = (static_cast<float>((i + N/2) % N)) / N;
    }
    checkCuda(cudaMemcpy(d_input, h_ref_input, sizeof(cufftComplex) * total_size, cudaMemcpyHostToDevice), "copy input to device");
    printf("start CUFFT\n");
    // CUFFT plan
    cufftHandle plan;
    checkCUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, batch), "cufftPlan1d");
    // Execute CUFFT
    printf("cufft plan created\n");
    checkCUFFT(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD), "cufftExecC2C");
    printf("CUFFT execution completed\n");
    // Copy output data back to host
    checkCuda(cudaMemcpy(h_ref_output, d_output, sizeof(cufftComplex) * total_size, cudaMemcpyDeviceToHost), "copy output to host");
    // print first 10 outputs
    for (int i = 0; i < 10; ++i) {
        printf("CUFFT Output[%d]: (%.5f, %.5f)\n", i, h_ref_output[i].x, h_ref_output[i].y);
    }
    // Run cuFFTDx
    float2 *cufftdx_input, *cufftdx_output;
    checkCuda(cudaMalloc(&cufftdx_input, sizeof(float2) * N * batch), "malloc cufftdx input");
    checkCuda(cudaMalloc(&cufftdx_output, sizeof(float2) * N * batch), "malloc cufftdx output");
    checkCuda(cudaMemcpy(cufftdx_input, h_ref_input, sizeof(float2) * N * batch, cudaMemcpyHostToDevice), "copy input to device for cufftdx");

    using FFT = decltype(Size<1024>() + Precision<float>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() + FFTsPerBlock<2>() + ElementsPerThread<8>() + SM<700>() + Block());

    constexpr int FFTsPerBlk = FFT::ffts_per_block;
    const int grid_size = (batch + FFTsPerBlk - 1) / FFTsPerBlk;
    cufftdx_batch_kernel<FFT, float2><<<grid_size, FFT::block_dim, FFT::shared_memory_size>>>(cufftdx_input, cufftdx_output, batch);
    checkCuda(cudaDeviceSynchronize(), "cufftdx kernel sync");

    // Copy cufftdx output back to host
    float2* h_cufftdx_output = (float2 *)malloc(sizeof(float2) * N * batch);
    checkCuda(cudaMemcpy(h_cufftdx_output, cufftdx_output, sizeof(float2) * N * batch, cudaMemcpyDeviceToHost), "copy cufftdx output to host");
    // Validate outputs
    validate_output(reinterpret_cast<float *>(h_ref_output), reinterpret_cast<float *>(h_cufftdx_output), N * batch, "CUFFT vs cuFFTDx", 1e-4f);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(cufftdx_input);
    cudaFree(cufftdx_output);
    free(h_cufftdx_output);
    free(h_ref_input);
    free(h_ref_output);
    return 0;
}