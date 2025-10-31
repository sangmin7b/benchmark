#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --exclusive
#SBATCH --partition=P2


./fft_benchmark-sm86

