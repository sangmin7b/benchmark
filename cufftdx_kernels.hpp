#pragma once
#include "benchmark_utils.hpp"
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

  unsigned int offset = global_fft_id * FFTSize / FFT::implicit_type_batching;
  unsigned int index = offset + threadIdx.x;

  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      thread_data[i] = reinterpret_cast<complex_type *>(input)[index];
    }
    index += Stride;
  }

#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(thread_data, reinterpret_cast<void *>(smem));
  }

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

#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(thread_data, reinterpret_cast<void *>(smem));
  }

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

#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(smem_p);
  }
  __syncthreads();

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

  complex_type thread_data[FFT::storage_size];

  unsigned int offset =
      2 * blockIdx.x * blockDim.x * cufftdx::size_of<FFT>::value;
  unsigned int index = offset + threadIdx.x * FFT::elements_per_thread;
  unsigned int batch_stride = blockDim.x * cufftdx::size_of<FFT>::value;
  for (size_t i = 0; i < FFT::elements_per_thread; i++) {
    thread_data[i] =
        example::to_rrii(input[index + i], input[index + i + batch_stride]);
  }

#pragma unroll 1
  for (int i = 0; i < repeat; i++) {
    FFT().execute(thread_data);
  }

  for (size_t i = 0; i < FFT::elements_per_thread; i++) {
    output[index + i] = example::to_ri1(thread_data[i]);
    output[index + i + batch_stride] = example::to_ri2(thread_data[i]);
  }
}


template <unsigned N, typename T, unsigned FPB, unsigned EPT, unsigned SMNUM,
          bool SMEM>
BenchmarkResult benchmark_cufftdx_1d_batch_impl(int batch,
                                                int comp_repeats = 10000,
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
    return {};
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
            cufftdx_batch_kernel_smem_half<FFT>
                <<<grid_size, FFT::block_dim, shmem>>>(in, out, b, repeats);
          } else {
            cufftdx_batch_kernel_half<FFT>
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
        long long n = N;
        long long inembed[1] = {N};
        long long onembed[1] = {N};
        long long istride = 1;
        long long ostride = 1;
        long long idist = N;
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
      int val_repeats = 1;
      void *val_args[4] = {d_input, d_output, &batch, &val_repeats};
      kernel(val_args);
      cudaDeviceSynchronize();
      checkCuda(cudaMemcpy(h_output, d_output, sizeof(T) * N * batch,
                           cudaMemcpyDeviceToHost),
                "copy output to host");
      valid = validate_output(h_ref_output, h_output, N * batch, "cuFFTDx");
      cudaFree(d_ref_output);
      free(h_input);
      free(h_output);
      free(h_ref_output);
      cudaDeviceSynchronize();
      printf("profile\n");
      if (comp_repeats == 1) {
        void *args[4] = {d_input, d_output, &batch, &comp_repeats};
        avg = measure_kernel_time(kernel, args, eval_repeats);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError(), "fftdx call");
      } else {
        int repeats2 = 2 * comp_repeats;
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
      return {};
    }
    cudaFree(d_input);
    cudaFree(d_output);
    return {avg, (int)FPB, (int)EPT, SMEM};
  }
}

template <unsigned N, typename T, unsigned THNUM, unsigned SMNUM>
BenchmarkResult benchmark_cufftdx_1d_batch_thread_impl(int batch,
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
    return {};
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
        long long n = N;
        long long inembed[1] = {N};
        long long onembed[1] = {N};
        long long istride = 1;
        long long ostride = 1;
        long long idist = N;
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
      int val_repeats = 1;
      void *val_args[4] = {d_input, d_output, &batch, &val_repeats};
      kernel(val_args);
      cudaDeviceSynchronize();
      cudaMemcpy(h_output, d_output, sizeof(T) * N * batch,
                 cudaMemcpyDeviceToHost);
      valid = validate_output(h_ref_output, h_output, N * batch, "cuFFTDx");
      cudaFree(d_ref_output);
      free(h_input);
      free(h_output);
      free(h_ref_output);
      cudaDeviceSynchronize();
      if (comp_repeats == 1) {
        void *args[4] = {d_input, d_output, &batch, &comp_repeats};
        avg = measure_kernel_time(kernel, args, eval_repeats);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError(), "fftdx call");
      } else {
        int repeats2 = 2 * comp_repeats;
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
      return {};
    }
    cudaFree(d_input);
    cudaFree(d_output);
    if (!valid)
      return {};
    return {avg, 0, 0, false};
  }
}
