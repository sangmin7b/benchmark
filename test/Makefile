# Paths
# MATHDX_ROOT = /home/s6/chaewon2/install/nvidia-mathdx-25.06.0/nvidia/mathdx/25.06
MATHDX_ROOT = /home/sangmin/.local/nvidia-mathdx-25.06.0/nvidia/mathdx/25.06

TARGET = test
SRCS = test.cu 
NVCC = nvcc
# V100(70), A100(80) 3090(86), 4090(89), H100(90)
# COMPUTE_CAPABILITY := 86 89 90
COMPUTE_CAPABILITY := 80
# LD_FLAGS = -L/home/s4/sangmin7b/cudnn9.7/lib
INCLUDES = -I. -I$(MATHDX_ROOT)/include -I$(MATHDX_ROOT)/external/cutlass/include/
LIBS = -lcudart -lcudnn -lcufft


all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) -gencode arch=compute_80,code=sm_80 -std=c++17 -O3 -DCUFFT_TARGET_ARCHS=$(COMPUTE_CAPABILITY)0 -DARCHNUM=$(COMPUTE_CAPABILITY)0 -o $@ $^ $(INCLUDES) $(LIBS) 

clean:
	rm -f $(TARGET)
