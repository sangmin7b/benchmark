// benchmark_utils.hpp
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cufftdx.hpp>

void checkCuda(cudaError_t err, const char *msg);
void checkCUFFT(cufftResult res, const char *msg);

// 함수 포인터 + context 구조로 시간 측정
typedef void (*BenchmarkFunc)(void *user_data);

float measure_kernel_time(BenchmarkFunc func, void *user_data, int repeat = 5);

bool validate_output(float2 *ref, float2 *target, int size, const char *tag,
                     float atol = 1e-3f);
bool validate_output(double2 *ref, double2 *target, int size, const char *tag,
                     float atol = 1e-3f);
bool validate_output(__half2 *ref, __half2 *target, int size, const char *tag,
                     float atol = 5e-1f);

void generate_input_data(float2 *data, int size, int batch, int seed = 42);
void generate_input_data(double2 *data, int size, int batch, int seed = 42);
void generate_input_data(__half2 *data, int size, int batch, int seed = 42);

template <typename T> struct DataTypeToCufftType;

template <> struct DataTypeToCufftType<float2> {
  static constexpr ::cufftType type = CUFFT_C2C;
};

template <> struct DataTypeToCufftType<double2> {
  static constexpr ::cufftType type = CUFFT_Z2Z;
};

template <typename T> struct DataTypeToPrecision;

template <> struct DataTypeToPrecision<float2> {
  using type = cufftdx::Precision<float>;
};

template <> struct DataTypeToPrecision<double2> {
  using type = cufftdx::Precision<double>;
};

template <> struct DataTypeToPrecision<__half2> {
  using type = cufftdx::Precision<__half>;
};

// Ref: NVIDIA CUDALibrarySamples
// https://github.com/NVIDIA/CUDALibrarySamples/blob/main/MathDx/cuFFTDx/common/fp16_common.hpp
namespace example {
__device__ __host__ __forceinline__ cufftdx::complex<__half2>
to_rrii(__half2 ri1, __half2 ri2) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  cufftdx::complex<__half2> rrii(__lows2half2(ri1, ri2),
                                 __highs2half2(ri1, ri2));
#else
  cufftdx::complex<__half2> rrii(__half2{ri1.x, ri2.x}, __half2{ri1.y, ri2.y});
#endif
  return rrii;
}

__device__ __host__ __forceinline__ __half2
to_ri1(cufftdx::complex<__half2> rrii) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  return __lows2half2(rrii.x, rrii.y);
#else
  return __half2{rrii.x.x, rrii.y.x};
#endif
}

// Return the second half complex number (as __half2) from complex<__half2>
// value with
// ((Real, Real), (Imag, Imag)) layout.
// Example: for rrii equal to ((1,2), (3,4)), it return __half2 (2, 4).
__device__ __host__ __forceinline__ __half2
to_ri2(cufftdx::complex<__half2> rrii) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  return __highs2half2(rrii.x, rrii.y);
#else
  return __half2{rrii.x.y, rrii.y.y};
#endif
}

__device__ __host__ __forceinline__ cufftdx::complex<__half2>
to_riri(cufftdx::complex<__half2> rrii) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  cufftdx::complex<__half2> riri(__lows2half2(rrii.x, rrii.y),
                                 __highs2half2(rrii.x, rrii.y));
#else
  cufftdx::complex<__half2> riri(__half2{rrii.x.x, rrii.y.x},
                                 __half2{rrii.x.y, rrii.y.y});
#endif
  return riri;
}

template <class FFT> struct io {

  template <unsigned int EPT, typename DataType>
  static inline __device__ void copy(const DataType *source, DataType *target,
                                     unsigned int n) {
    unsigned int stride = blockDim.x * blockDim.y;
    unsigned int index = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < EPT; i++) {
      if (index < n) {
        target[index] = source[index];
      }
      index += stride;
    }
  }

  template <class DataType>
  static inline __device__ void load_to_smem(const DataType *global,
                                             unsigned char *shared) {
    using input_t = typename FFT::input_type;
    copy<FFT::input_ept>(reinterpret_cast<const input_t *>(global),
                         reinterpret_cast<input_t *>(shared),
                         blockDim.y * FFT::input_length);
    __syncthreads();
  }

  template <class DataType>
  static inline __device__ void store_from_smem(const unsigned char *shared,
                                                DataType *global) {
    __syncthreads();
    using output_t = typename FFT::output_type;
    copy<FFT::output_ept>(reinterpret_cast<const output_t *>(shared),
                          reinterpret_cast<output_t *>(global),
                          blockDim.y * FFT::output_length);
  }
};

template <class FFT> struct io_fp16 {
  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;

  static_assert(std::is_same<scalar_type, __half2>::value,
                "This IO class is only for half precision FFTs");
  static_assert((FFT::ffts_per_block % 2 == 0),
                "This IO class works only for even FFT::ffts_per_block");

  static inline __device__ unsigned int
  batch_offset(unsigned int local_fft_id) {
    unsigned int global_fft_id =
        FFT::ffts_per_block == 1
            ? blockIdx.x
            : (blockIdx.x * FFT::ffts_per_block + local_fft_id);
    return cufftdx::size_of<FFT>::value * global_fft_id;
  }

  static inline __device__ void load(const __half2 *input,
                                     complex_type *thread_data,
                                     unsigned int local_fft_id) {
    // Calculate global offset of FFT batch
    const unsigned int offset = batch_offset(local_fft_id);
    // Get stride, this shows how elements from batch should be split between
    // threads
    const unsigned int stride = FFT::stride;
    unsigned int index = offset + threadIdx.x;
    // We bundle __half2 data of X-th and X+(FFT::ffts_per_block/2) batches
    // together.
    const unsigned int batch_stride =
        (FFT::ffts_per_block / 2) * cufftdx::size_of<FFT>::value;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
      thread_data[i] = to_rrii(input[index], input[index + batch_stride]);
      index += stride;
    }
  }

  static inline __device__ void store(const complex_type *thread_data,
                                      __half2 *output,
                                      unsigned int local_fft_id) {
    const unsigned int offset = batch_offset(local_fft_id);
    const unsigned int stride = FFT::stride;
    unsigned int index = offset + threadIdx.x;
    const unsigned int batch_stride =
        (FFT::ffts_per_block / 2) * cufftdx::size_of<FFT>::value;
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
      output[index] = to_ri1(thread_data[i]);
      output[index + batch_stride] = to_ri2(thread_data[i]);
      index += stride;
    }
  }
};
} // namespace example