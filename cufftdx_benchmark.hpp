#include "benchmark_utils.hpp"
#include <cstdio>
#include <cufftdx.hpp>

using namespace cufftdx;

template <class FFT, typename T>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void cufftdx_batch_kernel(T *input, T *output, int batch, int repeat) {
  using complex_type = typename FFT::value_type;

  constexpr int FFTsPerBlk = FFT::ffts_per_block;
  constexpr int FFTSize = cufftdx::size_of<FFT>::value;
  constexpr int Stride = FFT::stride;

  int global_fft_id = (blockIdx.x * FFTsPerBlk + threadIdx.y);
  if (global_fft_id >= batch)
    return;
  extern __shared__ __align__(alignof(float4)) unsigned char smem[];
  complex_type thread_data[FFT::elements_per_thread];

  // Custom load
  unsigned int offset = global_fft_id * FFTSize / FFT::implicit_type_batching;
  unsigned int index = offset + threadIdx.x;

  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      thread_data[i] = reinterpret_cast<complex_type *>(input)[index];
    }
    index += Stride;
  }

// Execute
#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(thread_data, reinterpret_cast<void *>(smem));
  }

  // Custom store
  index = offset + threadIdx.x;
  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      reinterpret_cast<complex_type *>(output)[index] = thread_data[i];
    }
    index += Stride;
  }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void cufftdx_batch_kernel_half(__half2 *input, __half2 *output, int batch,
                                   int repeat) {
  using complex_type = typename FFT::value_type;

  constexpr int FFTsPerBlk = FFT::ffts_per_block;
  constexpr int FFTSize = cufftdx::size_of<FFT>::value;
  constexpr int Stride = FFT::stride;

  int global_fft_id = (blockIdx.x * FFTsPerBlk + threadIdx.y);
  if (global_fft_id >= batch)
    return;
  extern __shared__ __align__(alignof(float4)) unsigned char smem[];
  complex_type thread_data[FFT::elements_per_thread];

  // Custom load
  unsigned int offset = global_fft_id * FFTSize;
  unsigned int index = offset + threadIdx.x;
  unsigned int batch_stride =
      (FFT::ffts_per_block / 2) * cufftdx::size_of<FFT>::value;
  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      thread_data[i] =
          example::to_rrii((input)[index], (input)[index + batch_stride]);
    }
    index += Stride;
  }

  // Execute
#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(thread_data, reinterpret_cast<void *>(smem));
  }

  // Custom store
  index = offset + threadIdx.x;
  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      output[index] = example::to_ri1(thread_data[i]);
      output[index + batch_stride] = example::to_ri2(thread_data[i]);
    }
    index += Stride;
  }
}

template <class FFT, typename T>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void cufftdx_batch_kernel_smem(T *input, T *output, int batch, int repeat) {
  using complex_type = typename FFT::value_type;

  constexpr int FFTsPerBlk = FFT::ffts_per_block;
  constexpr int FFTSize = cufftdx::size_of<FFT>::value;

  extern __shared__ __align__(alignof(float4)) unsigned char shared_mem[];
  complex_type *shared_mem_p = reinterpret_cast<complex_type *>(shared_mem);
  // Custom load
  unsigned int offset =
      blockIdx.x * FFTsPerBlk * FFTSize / FFT::implicit_type_batching;
  complex_type *input_block = reinterpret_cast<complex_type *>(input) + offset;
  complex_type *output_block =
      reinterpret_cast<complex_type *>(output) + offset;
  unsigned int stride = blockDim.x * blockDim.y;
  unsigned int index = threadIdx.y * blockDim.x + threadIdx.x;

  for (int i = 0; i < FFT::elements_per_thread; i++) {
    if (index < blockDim.y * FFT::input_length) {
      shared_mem_p[index] = input_block[index];
    }
    index += stride;
  }
  __syncthreads();

// Execute
#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(shared_mem_p);
  }
  __syncthreads();
  index = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < FFT::elements_per_thread; i++) {
    if (index < blockDim.y * FFT::input_length) {
      output_block[index] = shared_mem_p[index];
    }
    index += stride;
  }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void cufftdx_batch_kernel_smem_half(__half2 *input, __half2 *output,
                                        int batch, int repeat) {
  using complex_type = typename FFT::value_type;

  constexpr int FFTsPerBlk = FFT::ffts_per_block;
  constexpr int FFTSize = cufftdx::size_of<FFT>::value;

  int global_fft_id = (blockIdx.x * FFTsPerBlk + threadIdx.y);
  if (global_fft_id >= batch)
    return;
  extern __shared__ __align__(alignof(float4)) unsigned char smem[];
  complex_type *smem_p = reinterpret_cast<complex_type *>(smem);

  // Custom load
  unsigned int offset = blockIdx.x * FFTsPerBlk * FFTSize;
  __half2 *input_block = input + offset;
  __half2 *output_block = output + offset;
  unsigned int index = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * blockDim.y;
  unsigned int batch_stride =
      (FFT::ffts_per_block / 2) * cufftdx::size_of<FFT>::value;
  for (int i = 0; i < FFT::elements_per_thread; i++) {
    if (index < blockDim.y * FFT::input_length) {
      smem_p[index] = example::to_rrii((input_block)[index],
                                       (input_block)[index + batch_stride]);
    }
    index += stride;
  }
  __syncthreads();

  // Execute
#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(smem_p);
  }
  __syncthreads();

  // Custom store
  index = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < FFT::elements_per_thread; i++) {
    if (index < blockDim.y * FFT::input_length) {
      output_block[index] = example::to_ri1(smem_p[index]);
      output_block[index + batch_stride] = example::to_ri2(smem_p[index]);
    }
    index += stride;
  }
}

template <class FFT>
__global__ void thread_fft_kernel(typename FFT::value_type *input,
                                  typename FFT::value_type *output, int batch,
                                  int repeat) {
  using complex_type = typename FFT::value_type;

  complex_type thread_data[FFT::storage_size];

  unsigned int offset = blockIdx.x * blockDim.x * cufftdx::size_of<FFT>::value;
  unsigned int index = offset + threadIdx.x * FFT::elements_per_thread;
  for (size_t i = 0; i < FFT::elements_per_thread; i++) {
    thread_data[i] = input[index + i];
  }

// Execute FFT
#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(thread_data);
  }

  for (size_t i = 0; i < FFT::elements_per_thread; i++) {
    output[index + i] = thread_data[i];
  }
}

template <class FFT>
__global__ void thread_fft_kernel_half(__half2 *input, __half2 *output,
                                       int batch, int repeat) {
  using complex_type = typename FFT::value_type;

  // Local array for thread
  complex_type thread_data[FFT::storage_size];

  // Load data from global memory to registers.
  // thread_data should have all input data in order.
  unsigned int offset =
      2 * blockIdx.x * blockDim.x * cufftdx::size_of<FFT>::value;
  unsigned int index = offset + threadIdx.x * FFT::elements_per_thread;
  unsigned int batch_stride = blockDim.x * cufftdx::size_of<FFT>::value;
  for (size_t i = 0; i < FFT::elements_per_thread; i++) {
    thread_data[i] =
        example::to_rrii(input[index + i], input[index + i + batch_stride]);
  }

// Execute FFT
#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(thread_data);
  }

  // Save results
  for (size_t i = 0; i < FFT::elements_per_thread; i++) {
    output[index + i] = example::to_ri1(thread_data[i]);
    output[index + i + batch_stride] = example::to_ri2(thread_data[i]);
  }
}

template <unsigned N, typename T, unsigned FPB, unsigned EPT, unsigned SMNUM,
          bool SMEM>
float benchmark_cufftdx_1d_batch_impl(int batch, int comp_repeats = 10000,
                                      int eval_repeats = 5) {
  if (FPB == 0) {
    printf("[cuFFTDx] 1D Batch FFT Default\n");
  } else {
    printf("[cuFFTDx] 1D Batch FFT Type= %s, N = %u, batch = %d, FFTsPerBlock "
           "= %u, "
           "ElementsPerThread = %u SMEM = %d\n",
           typeid(T).name(), N, batch, FPB, EPT, SMEM);
  }

  using precision_t = typename DataTypeToPrecision<T>::type;
  using FFTBase = std::conditional_t<
      (FPB == 0),
      decltype(Size<N>() + precision_t{} + Type<fft_type::c2c>() +
               Direction<fft_direction::forward>() + Block()),
      decltype(Size<N>() + precision_t{} + Type<fft_type::c2c>() +
               Direction<fft_direction::forward>() + FFTsPerBlock<FPB>() +
               ElementsPerThread<EPT>() + Block())>;

  if constexpr (!cufftdx::is_supported<FFTBase, SMNUM>::value) {
    printf("Not supported\n");
    return std::numeric_limits<float>::infinity();
  } else {
    float avg = std::numeric_limits<float>::infinity();
    T *d_input, *d_output;
    checkCuda(cudaMalloc(&d_input, sizeof(T) * N * batch), "malloc input");
    checkCuda(cudaMalloc(&d_output, sizeof(T) * N * batch), "malloc output");
    T *d_ref_output;
    checkCuda(cudaMalloc(&d_ref_output, sizeof(T) * N * batch),
              "malloc ref output");
    T *h_input = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    T *h_output = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    T *h_ref_output = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    generate_input_data(h_input, N, batch);
    bool valid = false;

    try {
      using FFT = decltype(FFTBase() + SM<SMNUM>());
      printf("fft FPB: %d, EPT: %d\n", FFT::ffts_per_block,
             FFT::elements_per_thread);
      printf("fft inputlength: %d, fft block size %d, %d\n", FFT::input_length,
             FFT::block_dim.x, FFT::block_dim.y);
      using complex_type = typename FFT::value_type;
      auto kernel = [](void *ptr) {
        T *in = ((T **)ptr)[0];
        T *out = ((T **)ptr)[1];
        int b = *((int *)(((T **)ptr)[2]));
        int repeats = *((int *)(((T **)ptr)[3]));
        constexpr int FFTsPerBlk = FFT::ffts_per_block;
        int grid_size = (b + FFTsPerBlk - 1) / FFTsPerBlk;
        constexpr unsigned int size_bytes =
            FFT::ffts_per_block / FFT::implicit_type_batching *
            cufftdx::size_of<FFT>::value * sizeof(complex_type);
        constexpr size_t shmem = std::max(FFT::shared_memory_size, size_bytes);
        if constexpr (std::is_same_v<T, __half2>) {
          if constexpr (SMEM) {
            cufftdx_batch_kernel_half<FFT>
                <<<grid_size, FFT::block_dim, shmem>>>(in, out, b, repeats);
          } else {
            cufftdx_batch_kernel_smem_half<FFT>
                <<<grid_size, FFT::block_dim, shmem>>>(in, out, b, repeats);
          }
        } else {
          if constexpr (SMEM) {
            cufftdx_batch_kernel_smem<FFT, T>
                <<<grid_size, FFT::block_dim, shmem>>>(in, out, b, repeats);
          } else {
            cufftdx_batch_kernel<FFT, T>
                <<<grid_size, FFT::block_dim, shmem>>>(in, out, b, repeats);
          }
        }
        cudaDeviceSynchronize();
        checkCuda(cudaPeekAtLastError(), "after kernel execution");
      };
      constexpr unsigned int size_bytes =
          FFT::ffts_per_block / FFT::implicit_type_batching *
          cufftdx::size_of<FFT>::value * sizeof(complex_type);
      constexpr size_t shmem = std::max(FFT::shared_memory_size, size_bytes);
      // constexpr size_t shmem = FFT::shared_memory_size;
      printf("[Setting] Shared memory size: %zu bytes\n", shmem);
      if constexpr (std::is_same_v<T, __half2>) {
        if constexpr (SMEM) {
          cudaFuncSetAttribute(cufftdx_batch_kernel_smem_half<FFT>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               shmem);
        } else {
          cudaFuncSetAttribute(cufftdx_batch_kernel_half<FFT>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               shmem);
        }
      } else {
        if constexpr (SMEM) {
          cudaFuncSetAttribute(cufftdx_batch_kernel_smem<FFT, T>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               shmem);
        } else {
          cudaFuncSetAttribute(cufftdx_batch_kernel<FFT, T>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               shmem);
        }
      }
      printf("Start validation\n");
      // validate
      cudaMemcpy(d_input, h_input, sizeof(T) * N * batch,
                 cudaMemcpyHostToDevice);
      cufftHandle plan;
      if constexpr (std::is_same_v<T, float2>) {
        checkCUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, batch), "create plan");
        cufftExecC2C(plan, (cufftComplex *)d_input,
                     (cufftComplex *)d_ref_output, CUFFT_FORWARD);
      } else if constexpr (std::is_same_v<T, double2>) {
        checkCUFFT(cufftPlan1d(&plan, N, CUFFT_Z2Z, batch), "create plan");
        cufftExecZ2Z(plan, (cufftDoubleComplex *)d_input,
                     (cufftDoubleComplex *)d_ref_output, CUFFT_FORWARD);
      } else if constexpr (std::is_same_v<T, __half2>) {
        long long n = N; // size per transform
        long long inembed[1] = {N};
        long long onembed[1] = {N};
        long long istride = 1; // contiguous
        long long ostride = 1;
        long long idist = N; // distance between consecutive FFTs
        long long odist = N;
        size_t workSize = 0;
        checkCUFFT(cufftCreate(&plan), "create plan");
        checkCUFFT(cufftXtMakePlanMany(plan, 1, &n, inembed, istride, idist,
                                       CUDA_C_16F, onembed, ostride, odist,
                                       CUDA_C_16F, batch, &workSize,
                                       CUDA_C_16F),
                   "plan1d batch");
        cufftXtExec(plan, (cufftComplex *)d_input, (cufftComplex *)d_ref_output,
                    CUFFT_FORWARD);
      }

      checkCuda(cudaMemcpy(h_ref_output, d_ref_output, sizeof(T) * N * batch,
                           cudaMemcpyDeviceToHost),
                "copy ref output to host");
      cudaDeviceSynchronize();
      cufftDestroy(plan);
      // launch kernel
      int val_repeats = 1;
      void *val_args[4] = {d_input, d_output, &batch, &val_repeats};
      kernel(val_args);
      cudaDeviceSynchronize();
      checkCuda(cudaMemcpy(h_output, d_output, sizeof(T) * N * batch,
                           cudaMemcpyDeviceToHost),
                "copy output to host");
      // validate output

      valid = validate_output(h_ref_output, h_output, N * batch, "cuFFTDx");
      // validate_output(ref_output, d_output, N * batch, "cuFFTDx");
      cudaFree(d_ref_output);
      free(h_input);
      free(h_output);
      free(h_ref_output);
      // profile
      cudaDeviceSynchronize();
      printf("profile\n");
      // End to End
      if (comp_repeats == 1) {
        void *args[4] = {d_input, d_output, &batch, &comp_repeats};
        avg = measure_kernel_time(kernel, args, eval_repeats);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError(), "fftdx call");
      } else {
        int repeats2 = 2 * comp_repeats;
        // void *args[3] = {d_input, d_output, &batch};
        void *args[4] = {d_input, d_output, &batch, &comp_repeats};
        avg = measure_kernel_time(kernel, args, eval_repeats);
        cudaDeviceSynchronize();
        void *args2[4] = {d_input, d_output, &batch, &repeats2};
        float avg2 = measure_kernel_time(kernel, args2, eval_repeats);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError(), "fftdx call");
        avg = (avg2 - avg) / comp_repeats;
      }
      printf("Average time: %.6f ms\n", avg);
      checkCuda(cudaPeekAtLastError(), "check CUDA");
    } catch (const std::runtime_error &e) {
      printf("Error during cuFFTDx execution: %s\n", e.what());
      cudaFree(d_input);
      cudaFree(d_ref_output);
      cudaFree(d_output);
      return std::numeric_limits<float>::infinity();
    }
    cudaFree(d_input);
    cudaFree(d_output);
    // if (!valid)
    //   return std::numeric_limits<float>::infinity();
    return avg;
  }
}

template <unsigned N, typename T, unsigned THNUM, unsigned SMNUM>
float benchmark_cufftdx_1d_batch_thread_impl(int batch,
                                             int comp_repeats = 10000,
                                             int eval_repeats = 5) {
  printf("[cuFFTDx] 1D Batch FFT (thread) Type= %s, N = %u, thread num = %d, "
         "batch = %d\n",
         typeid(T).name(), N, THNUM, batch);
  constexpr int num_threads = THNUM;
  using precision_t = typename DataTypeToPrecision<T>::type;
  using FFTBase = decltype(Size<N>() + precision_t{} + Type<fft_type::c2c>() +
                           Direction<fft_direction::forward>() + Thread());
  if constexpr (!cufftdx::is_supported<FFTBase, SMNUM>::value) {
    return std::numeric_limits<float>::infinity();
  } else {
    float avg = std::numeric_limits<float>::infinity();
    T *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(T) * N * batch);
    cudaMalloc(&d_output, sizeof(T) * N * batch);
    T *d_ref_output;
    checkCuda(cudaMalloc(&d_ref_output, sizeof(T) * N * batch),
              "malloc ref output");
    T *h_input = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    T *h_output = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    T *h_ref_output = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    generate_input_data(h_input, N, batch);
    bool valid = false;

    try {
      using FFT = decltype(FFTBase() + SM<SMNUM>());
      auto kernel = [](void *ptr) {
        T *in = ((T **)ptr)[0];
        T *out = ((T **)ptr)[1];
        int b = *((int *)(((T **)ptr)[2]));
        int repeats = *((int *)(((T **)ptr)[3]));
        using complex_type = typename FFT::value_type;
        if constexpr (std::is_same_v<T, __half2>) {
          int grid_size = (b + 2 * num_threads - 1) / (2 * num_threads);
          thread_fft_kernel_half<FFT>
              <<<grid_size, num_threads>>>(in, out, b, repeats);
        } else {
          int grid_size = (b + num_threads - 1) / num_threads;
          thread_fft_kernel<FFT><<<grid_size, num_threads>>>(
              reinterpret_cast<complex_type *>(in),
              reinterpret_cast<complex_type *>(out), b, repeats);
        }
        cudaDeviceSynchronize();
      };

      printf("Start validation\n");
      // validate
      cudaMemcpy(d_input, h_input, sizeof(T) * N * batch,
                 cudaMemcpyHostToDevice);
      cufftHandle plan;
      if constexpr (std::is_same_v<T, float2>) {
        checkCUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, batch), "create plan");
        cufftExecC2C(plan, (cufftComplex *)d_input,
                     (cufftComplex *)d_ref_output, CUFFT_FORWARD);
      } else if constexpr (std::is_same_v<T, double2>) {
        checkCUFFT(cufftPlan1d(&plan, N, CUFFT_Z2Z, batch), "create plan");
        cufftExecZ2Z(plan, (cufftDoubleComplex *)d_input,
                     (cufftDoubleComplex *)d_ref_output, CUFFT_FORWARD);
      } else if constexpr (std::is_same_v<T, __half2>) {
        long long n = N; // size per transform
        long long inembed[1] = {N};
        long long onembed[1] = {N};
        long long istride = 1; // contiguous
        long long ostride = 1;
        long long idist = N; // distance between consecutive FFTs
        long long odist = N;
        size_t workSize = 0;
        checkCUFFT(cufftCreate(&plan), "create plan");
        checkCUFFT(cufftXtMakePlanMany(plan, 1, &n, inembed, istride, idist,
                                       CUDA_C_16F, onembed, ostride, odist,
                                       CUDA_C_16F, batch, &workSize,
                                       CUDA_C_16F),
                   "plan1d batch");
        cufftXtExec(plan, (cufftComplex *)d_input, (cufftComplex *)d_ref_output,
                    CUFFT_FORWARD);
      }

      cudaMemcpy(h_ref_output, d_ref_output, sizeof(T) * N * batch,
                 cudaMemcpyDeviceToHost);
      cufftDestroy(plan);
      // launch kernel
      int val_repeats = 1;
      void *val_args[4] = {d_input, d_output, &batch, &val_repeats};
      kernel(val_args);
      cudaDeviceSynchronize();
      cudaMemcpy(h_output, d_output, sizeof(T) * N * batch,
                 cudaMemcpyDeviceToHost);
      // validate output
      valid = validate_output(h_ref_output, h_output, N * batch, "cuFFTDx");
      // validate_output(ref_output, d_output, N * batch, "cuFFTDx");
      cudaFree(d_ref_output);
      free(h_input);
      free(h_output);
      free(h_ref_output);
      // profile
      cudaDeviceSynchronize();
      // End to End
      if (comp_repeats == 1) {
        void *args[4] = {d_input, d_output, &batch, &comp_repeats};
        avg = measure_kernel_time(kernel, args, eval_repeats);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError(), "fftdx call");
      } else {
        int repeats2 = 2 * comp_repeats;
        // void *args[3] = {d_input, d_output, &batch};
        void *args[4] = {d_input, d_output, &batch, &comp_repeats};
        avg = measure_kernel_time(kernel, args, eval_repeats);
        cudaDeviceSynchronize();
        void *args2[4] = {d_input, d_output, &batch, &repeats2};
        float avg2 = measure_kernel_time(kernel, args2, eval_repeats);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError(), "fftdx call");
        avg = (avg2 - avg) / comp_repeats;
      }
      printf("Average time: %.6f ms\n", avg);

    } catch (const std::runtime_error &e) {
      printf("Error during cuFFTDx execution: %s\n", e.what());
      cudaFree(d_input);
      cudaFree(d_ref_output);
      cudaFree(d_output);
      return std::numeric_limits<float>::infinity();
    }
    cudaFree(d_input);
    cudaFree(d_output);
    if (!valid)
      return std::numeric_limits<float>::infinity();
    return avg;
  }
}

template <unsigned N, typename T, unsigned SMNUM>
float benchmark_cufftdx_1d_batch(int batch, bool smem = false, bool e2e = false,
                                 bool default_only = false) {
  float min_time = std::numeric_limits<float>::infinity();
  // Test all configurations from (FFT Per Block in [1, 32], Elements Per Thread
  // in [2, 32])
  // https://docs.nvidia.com/cuda/cufftdx/api/operators.html#ept-operator-label

  int comp_repeats = 10000;
  int eval_repeats = 5 * 4096 / N;

  if (e2e) {
    comp_repeats = 1;
    eval_repeats = 10000;
  }

  // default
  printf("cuFFTDx with default FPB, EPT\n");
  if (smem) {
    min_time = std::min(
        min_time, benchmark_cufftdx_1d_batch_impl<N, T, 0, 0, SMNUM, true>(
                      batch, comp_repeats, eval_repeats));
  } else {
    min_time = std::min(
        min_time, benchmark_cufftdx_1d_batch_impl<N, T, 0, 0, SMNUM, false>(
                      batch, comp_repeats, eval_repeats));
  }
  if (default_only)
    return min_time;

  // A100 single

  if constexpr (N == 64 && std::is_same_v<T, float2>) {
    // reg & smem
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));

      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_thread_impl<N, T, 16, SMNUM>(
                        batch, comp_repeats, eval_repeats));

      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_thread_impl<N, T, 32, SMNUM>(
                        batch, comp_repeats, eval_repeats));

      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_thread_impl<N, T, 64, SMNUM>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_thread_impl<N, T, 128, SMNUM>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_thread_impl<N, T, 256, SMNUM>(
                        batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 128 && std::is_same_v<T, float2>) {
    // reg
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 8, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
  }

  if constexpr (N == 256 && std::is_same_v<T, float2>) {
    if (smem) {
      // smem
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 8, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      // reg
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
  }
  if constexpr (N == 512 && std::is_same_v<T, float2>) {
    // reg & smem
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 8, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 8, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
  }
  if constexpr (N == 1024 && std::is_same_v<T, float2>) {
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 4, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 32, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
  }
  if constexpr (N == 2048 && std::is_same_v<T, float2>) {
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 4, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
  }
  if constexpr (N == 4096 && std::is_same_v<T, float2>) {
    // smem
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 32, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
  }

  // A100 half
  if constexpr (N == 64 && std::is_same_v<T, __half2>) {
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 8, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 8, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
    min_time = std::min(min_time,
                        benchmark_cufftdx_1d_batch_thread_impl<N, T, 16, SMNUM>(
                            batch, comp_repeats, eval_repeats));
    min_time = std::min(min_time,
                        benchmark_cufftdx_1d_batch_thread_impl<N, T, 32, SMNUM>(
                            batch, comp_repeats, eval_repeats));
    min_time = std::min(min_time,
                        benchmark_cufftdx_1d_batch_thread_impl<N, T, 64, SMNUM>(
                            batch, comp_repeats, eval_repeats));
    min_time = std::min(
        min_time, benchmark_cufftdx_1d_batch_thread_impl<N, T, 128, SMNUM>(
                      batch, comp_repeats, eval_repeats));
    min_time = std::min(
        min_time, benchmark_cufftdx_1d_batch_thread_impl<N, T, 256, SMNUM>(
                      batch, comp_repeats, eval_repeats));
  }
  if constexpr (N == 256 && std::is_same_v<T, __half2>) {
    // smem
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 8, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      // reg
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
  }
  if constexpr (N == 1024 && std::is_same_v<T, __half2>) {
    // reg  & smem
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
  }
  if constexpr (N == 4096 && std::is_same_v<T, __half2>) {
    // smem
    if (smem) {
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 32, SMNUM, true>(
                        batch, comp_repeats, eval_repeats));
    } else {
      // reg
      min_time = std::min(
          min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM, false>(
                        batch, comp_repeats, eval_repeats));
    }
  }

  return min_time;
}

template <unsigned N, typename T, unsigned SMNUM>
float search_parameters(int batch) {
  float min_time = std::numeric_limits<float>::infinity();
  // Test all configurations from (FFT Per Block in [1, 32], Elements Per Thread
  // in [2, 32])
  // https://docs.nvidia.com/cuda/cufftdx/api/operators.html#ept-operator-label

#ifdef SEARCH
  int comp_repeats = 800;
  int eval_repeats = 3;
  if constexpr (!std::is_same_v<T, __half2>) {
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 2, SMNUM>(
                               batch, comp_repeats, eval_repeats));
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 4, SMNUM>(
                               batch, comp_repeats, eval_repeats));
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 8, SMNUM>(
                               batch, comp_repeats, eval_repeats));
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM>(
                               batch, comp_repeats, eval_repeats));
  }

  if constexpr (!std::is_same_v<T, __half2> && !std::is_same_v<T, double2>)
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 32, SMNUM>(
                               batch, comp_repeats, eval_repeats));

  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 2, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 4, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 8, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  if constexpr (!std::is_same_v<T, double2>)
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 32, SMNUM>(
                               batch, comp_repeats, eval_repeats));

  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 2, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 4, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 8, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  if constexpr (!std::is_same_v<T, double2>)
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 32, SMNUM>(
                               batch, comp_repeats, eval_repeats));

  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 2, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 4, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 8, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  if constexpr (!std::is_same_v<T, double2>)
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 32, SMNUM>(
                               batch, comp_repeats, eval_repeats));

  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 2, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 4, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 16, SMNUM>(
                             batch, comp_repeats, eval_repeats));

  if constexpr (!std::is_same_v<T, double2>)
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 32, SMNUM>(
                               batch, comp_repeats, eval_repeats));

  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 2, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 4, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 8, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  min_time =
      std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 16, SMNUM>(
                             batch, comp_repeats, eval_repeats));
  if constexpr (!std::is_same_v<T, double2>)
    min_time =
        std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 32, SMNUM>(
                               batch, comp_repeats, eval_repeats));
#endif
  return min_time;
}
