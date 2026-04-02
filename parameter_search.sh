for n in 64 128 256 512 1024 2048 4096; do
    stdbuf -oL -eL ./fft_benchmark -N $n --search >> search_${n}.txt 
    stdbuf -oL -eL ./fft_benchmark -N $n --search --smem >> search_smem_${n}.txt 
done
