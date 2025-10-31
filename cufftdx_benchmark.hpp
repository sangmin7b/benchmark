#include "benchmark_utils.hpp"
#include <cufftdx.hpp>
#include <cstdio>

using namespace cufftdx;

template <class FFT, typename T> __global__ void cufftdx_single_kernel(T *input, T *output) {
  using complex_type = typename FFT::value_type;

  extern __shared__ __align__(alignof(float4)) void *smem;
  complex_type thread_data[FFT::elements_per_thread];

  // Custom load
  constexpr int FFTSize = cufftdx::size_of<FFT>::value;
  constexpr int Stride = FFT::stride;
  unsigned int index = threadIdx.x;

  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      thread_data[i] = reinterpret_cast<complex_type *>(input)[index];
    }
    index += Stride;
  }

  // Execute
  FFT().execute(thread_data, smem);

  // Custom store
  index = threadIdx.x;
  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      reinterpret_cast<complex_type *>(output)[index] = thread_data[i];
    }
    index += Stride;
  }
}

template <unsigned N, typename T, unsigned FPB, unsigned EPT, unsigned SMNUM>
float benchmark_cufftdx_1d() {
  printf("[cuFFTDx] 1D Single FFT: N = %u\n", N);
  using precision_t = typename DataTypeToPrecision<T>::type;
  using FFT = decltype(Size<N>() + precision_t{} + Type<fft_type::c2c>() +
                       Direction<fft_direction::forward>() + FFTsPerBlock<FPB>() +
                       ElementsPerThread<EPT>() + SM<SMNUM>() + Block());
  T *d_input;
  T *d_output;
  cudaMalloc(&d_input, sizeof(T) * N);
  cudaMalloc(&d_output, sizeof(T) * N);

  float avg = 0.0f;
  if constexpr ((cufftdx::is_supported<FFT, SMNUM>::value)) {
    auto kernel = [](void *ptr) {
      T *in = ((T **)ptr)[0];
      T *out = ((T **)ptr)[1];
      cufftdx_single_kernel<FFT, T><<<1, FFT::block_dim, FFT::shared_memory_size>>>(in, out);
      cudaDeviceSynchronize();
    };

    void *args[2] = {d_input, d_output};
    avg = measure_kernel_time(kernel, args);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError(), "fftdx call");
    printf("Average time: %.3f ms\n", avg);
  } else {
    printf("Not supported: FFT<%u, %u, %u, %u, %u>\n", N, FPB, EPT, SMNUM);
  }
  cudaFree(d_input);
  cudaFree(d_output);
  return avg;
}

template <class FFT, typename T>
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_batch_kernel(T *input, T *output, int batch, int repeat) {
  using complex_type = typename FFT::value_type;

  constexpr int FFTsPerBlk = FFT::ffts_per_block;
  constexpr int FFTSize = cufftdx::size_of<FFT>::value;
  constexpr int Stride = FFT::stride;

  int global_fft_id = (blockIdx.x * FFTsPerBlk + threadIdx.y);
  if (global_fft_id >= batch)
    return;
  extern __shared__ __align__(alignof(float4)) void *smem;
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
  for (int i = 0; i < repeat; i++){
    FFT().execute(thread_data, smem);
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
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_batch_kernel_half(__half2 *input, __half2 *output, int batch, int repeat) {
  using complex_type = typename FFT::value_type;

  constexpr int FFTsPerBlk = FFT::ffts_per_block;
  constexpr int FFTSize = cufftdx::size_of<FFT>::value;
  constexpr int Stride = FFT::stride;

  int global_fft_id = (blockIdx.x * FFTsPerBlk + threadIdx.y);
  if (global_fft_id >= batch)
    return;
  extern __shared__ __align__(alignof(float4)) void *smem;
  complex_type thread_data[FFT::elements_per_thread];

  // Custom load
  unsigned int offset = global_fft_id * FFTSize;
  unsigned int index = offset + threadIdx.x;
  unsigned int batch_stride = (FFT::ffts_per_block / 2) * cufftdx::size_of<FFT>::value;
  for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
    if ((i * Stride + threadIdx.x) < FFTSize) {
      thread_data[i] = example::to_rrii((input)[index], (input)[index + batch_stride]);
    }
    index += Stride;
  }

  // Execute
  #pragma unroll 1
  for (int i = 0; i < repeat; i++){
    FFT().execute(thread_data, smem);
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


// template <class FFT, typename T>
// __launch_bounds__(FFT::max_threads_per_block) __global__
//     void cufftdx_batch_kernel_thread(T *input, T *output, int batch) {
//   using complex_type = typename FFT::value_type;

//   constexpr int FFTsPerBlk = FFT::ffts_per_block;
//   constexpr int FFTSize = cufftdx::size_of<FFT>::value;
//   constexpr int Stride = FFT::stride;

//   int global_fft_id = (blockIdx.x * FFTsPerBlk + threadIdx.y) / FFT::implicit_type_batching;
//   if (global_fft_id >= batch)
//     return;
//   extern __shared__ __align__(alignof(float4)) void *smem;
//   complex_type thread_data[FFT::elements_per_thread];

//   // Custom load
//   unsigned int offset = global_fft_id * FFTSize;
//   unsigned int index = offset + threadIdx.x;

//   for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
//     if ((i * Stride + threadIdx.x) < FFTSize) {
//       thread_data[i] = reinterpret_cast<complex_type *>(input)[index];
//     }
//     index += Stride;
//   }

//   // Execute
//   for (int i = 0; i < 10000; i++){
//      FFT().execute(thread_data, smem);
//   }

//   // Custom store
//   index = offset + threadIdx.x;
//   for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
//     if ((i * Stride + threadIdx.x) < FFTSize) {
//       reinterpret_cast<complex_type *>(output)[index] = thread_data[i];
//     }
//     index += Stride;
//   }
// }


template <unsigned N, typename T, unsigned FPB, unsigned EPT, unsigned SMNUM>
float benchmark_cufftdx_1d_batch_impl(int batch, int comp_repeats = 10000, int eval_repeats = 5) {
  printf("[cuFFTDx] 1D Batch FFT Type= %s, N = %u, batch = %d, FFTsPerBlock = %u, "
         "ElementsPerThread = %u\n",
         typeid(T).name(), N, batch, FPB, EPT);

  using precision_t = typename DataTypeToPrecision<T>::type;
  using FFTBase = decltype(Size<N>() + precision_t{} + Type<fft_type::c2c>() +
                           Direction<fft_direction::forward>() + FFTsPerBlock<FPB>() +
                           ElementsPerThread<EPT>() + Block());
  if constexpr (!cufftdx::is_supported<FFTBase, SMNUM>::value) {
    return std::numeric_limits<float>::infinity();
  } else {
    float avg = std::numeric_limits<float>::infinity();
    T *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(T) * N * batch);
    cudaMalloc(&d_output, sizeof(T) * N * batch);
    T *d_ref_output;
    // checkCuda(cudaMalloc(&d_ref_input, sizeof(T) * N * batch), "malloc ref input");
    checkCuda(cudaMalloc(&d_ref_output, sizeof(T) * N * batch), "malloc ref output");
    T *h_input = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    T *h_output = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    T *h_ref_output = reinterpret_cast<T *>(malloc(sizeof(T) * N * batch));
    generate_input_data(h_input, N, batch);
    bool valid = true;
      
    try {
      using FFT = decltype(FFTBase() + SM<SMNUM>());
      printf("fft inputlength: %d, fft block size %d, %d\n", FFT::input_length, FFT::block_dim.x,
             FFT::block_dim.y);
      cudaFuncAttributes fa{};
      cudaFuncGetAttributes(&fa, cufftdx_batch_kernel_half<FFT>);
      printf("numRegs=%d\n", fa.numRegs);

      auto kernel = [](void *ptr) {
        T *in = ((T **)ptr)[0];
        T *out = ((T **)ptr)[1];
        int b = *((int *)(((T **)ptr)[2]));
        int repeats = *((int *)(((T **)ptr)[3]));
        constexpr int FFTsPerBlk = FFT::ffts_per_block;
        int grid_size = (b + FFTsPerBlk - 1) / FFTsPerBlk;
        cufftdx_batch_kernel_half<FFT>
            <<<grid_size, FFT::block_dim, FFT::shared_memory_size>>>(in, out, b, repeats);
        cudaDeviceSynchronize();
      };

      constexpr size_t shmem = FFT::shared_memory_size;
      cudaFuncSetAttribute(cufftdx_batch_kernel_half<FFT>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
      printf("Shared memory size: %zu bytes\n", shmem);
      printf("Start validation\n");
      // validate
      cudaMemcpy(d_input, h_input, sizeof(T) * N * batch, cudaMemcpyHostToDevice);
      cufftHandle plan;
      if constexpr (std::is_same_v<T, float2>) {
        checkCUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, batch), "create plan");
        cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_ref_output, CUFFT_FORWARD);
      } else if constexpr (std::is_same_v<T, double2>) {
        checkCUFFT(cufftPlan1d(&plan, N, CUFFT_Z2Z, batch), "create plan");
        cufftExecZ2Z(plan, (cufftDoubleComplex *)d_input, (cufftDoubleComplex *)d_ref_output,
                     CUFFT_FORWARD);
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
        checkCUFFT(cufftXtMakePlanMany(plan, 1, &n, inembed, istride, idist, CUDA_C_16F, onembed,
                                      ostride, odist, CUDA_C_16F, batch, &workSize, CUDA_C_16F),
                  "plan1d batch");
        cufftXtExec(plan, (cufftComplex *)d_input, (cufftComplex *)d_ref_output, CUFFT_FORWARD);
      }

      cudaMemcpy(h_ref_output, d_ref_output, sizeof(T) * N * batch, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      cufftDestroy(plan);
      // launch kernel
      int val_repeats = 1;
      void *val_args[4] = {d_input, d_output, &batch, &val_repeats};
      kernel(val_args);
      cudaMemcpy(h_output, d_output, sizeof(T) * N * batch, cudaMemcpyDeviceToHost);
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
      if (comp_repeats == 1){
        void *args[4] = {d_input, d_output, &batch, &comp_repeats};
        avg = measure_kernel_time(kernel, args, eval_repeats);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError(), "fftdx call");
      } else{
        int repeats2 = 2 * comp_repeats;
        checkCuda(cudaGetLastError(), "fftdx call");
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

template <unsigned N, typename T, unsigned SMNUM> float benchmark_cufftdx_1d_batch(int batch) {
  float min_time = std::numeric_limits<float>::infinity();
  // Test all configurations from (FFT Per Block in [1, 32], Elements Per Thread in [2, 32])
  // https://docs.nvidia.com/cuda/cufftdx/api/operators.html#ept-operator-label



  // if constexpr (!std::is_same_v<T, __half2>) {
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 2, SMNUM>(batch));
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 4, SMNUM>(batch));
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 8, SMNUM>(batch));
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM>(batch));
  // }
  // if constexpr (!std::is_same_v<T, __half2> && !std::is_same_v<T, double2>)
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 32, SMNUM>(batch));

  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 2, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 4, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 8, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM>(batch));
  // if constexpr (!std::is_same_v<T, double2>)
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 32, SMNUM>(batch));

  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 2, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 4, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 8, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM>(batch));
  // if constexpr (!std::is_same_v<T, double2>)
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 32, SMNUM>(batch));

  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 2, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 4, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 8, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM>(batch));
  // if constexpr (!std::is_same_v<T, double2>)
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 32, SMNUM>(batch));

  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 2, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 4, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 16, SMNUM>(batch));
  // if constexpr (!std::is_same_v<T, double2>)
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 32, SMNUM>(batch));

  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 2, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 4, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 8, SMNUM>(batch));
  // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 16, SMNUM>(batch));
  // if constexpr (!std::is_same_v<T, double2>)
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 32, SMNUM>(batch));

  int comp_repeats = 1;
  int eval_repeats = 1000;

  // A100 single 
  if constexpr (N == 64 && std::is_same_v<T, float2>){
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM>(batch));
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM>(batch));
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM>(batch));
  }

  if constexpr (N == 256 && std::is_same_v<T, float2>){
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM>(batch));
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM>(batch));
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM>(batch));
  }
  if constexpr (N == 1024 && std::is_same_v<T, float2>){
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 16, SMNUM>(batch));
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 1, 32, SMNUM>(batch));
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM>(batch));
  }
  if constexpr (N == 4096 && std::is_same_v<T, float2>){
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM>(batch));
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM>(batch));
    // min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 8, 16, SMNUM>(batch));
  }


  // A100 half
  // if constexpr (N == 64 && std::is_same_v<T, __half2>){
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 32, 8, SMNUM>(batch, comp_repeats, eval_repeats));
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM>(batch, comp_repeats, eval_repeats));
  // }
  // if constexpr (N == 256 && std::is_same_v<T, __half2>){
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 16, 8, SMNUM>(batch, comp_repeats, eval_repeats));
  // }
  // if constexpr (N == 1024 && std::is_same_v<T, __half2>){
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM>(batch, comp_repeats, eval_repeats));
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 4, 16, SMNUM>(batch, comp_repeats, eval_repeats));
  // }
  // if constexpr (N == 4096 && std::is_same_v<T, __half2>){
  //   min_time = std::min(min_time, benchmark_cufftdx_1d_batch_impl<N, T, 2, 16, SMNUM>(batch, comp_repeats, eval_repeats));
  // }

  return min_time;
}
