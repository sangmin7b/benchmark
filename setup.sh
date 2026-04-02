#!/bin/bash

MATHDX_PATH="./"

wget https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda12/nvidia-mathdx-25.06.1-cuda12.zip

unzip nvidia-mathdx-25.06.1-cuda12.zip -d $MATHDX_PATH
rm nvidia-mathdx-25.06.1-cuda12.zip