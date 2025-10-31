for n in 64 256 1024 4096; do
    ./fft_benchmark -N $n > h_bench_e2e_$n.txt
done;