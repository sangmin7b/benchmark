# Paths
# MATHDX_ROOT = /home/s6/chaewon2/install/nvidia-mathdx-25.06.0/nvidia/mathdx/25.06
MATHDX_ROOT = /home/sangmin/.local/nvidia-mathdx-25.06.0/nvidia/mathdx/25.06

TARGET = fft_benchmark
SRCS = main.cu  benchmark_utils.cu
NVCC = nvcc
# V100(70), A100(80) 3090(86), 4090(89), H100(90)
# COMPUTE_CAPABILITY := 86 89 90
COMPUTE_CAPABILITY := 80
# LD_FLAGS = -L/home/s4/sangmin7b/cudnn9.7/lib
INCLUDES = -I. -I$(MATHDX_ROOT)/include -I$(MATHDX_ROOT)/external/cutlass/include/
LIBS = -lcudart -lcudnn -lcufft


all: $(COMPUTE_CAPABILITY:%=$(TARGET)-sm%)

$(TARGET)-sm%: $(SRCS)
	$(NVCC)  -gencode arch=compute_$*,code=sm_$* -std=c++17 -O3 -DCUFFT_TARGET_ARCHS=$*0fft_benchmark -DARCHNUM=$*0 -o $@ $^ $(INCLUDES) $(LIBS) 

clean:
	rm -f $(COMPUTE_CAPABILITY:%=$(TARGET)-sm%)


	# Makefile for building cufftdx.cu with cuFFT/cuFFTDx
