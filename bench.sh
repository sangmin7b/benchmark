for n in 512 2048; do
    stdbuf -oL -eL ./fft_benchmark -N $n --type 1 >> a100_${n}.txt 
done


for n in 512 2048; do
    stdbuf -oL -eL ./fft_benchmark -N $n --type 1 --smem >> a100_smem_${n}.txt 
done