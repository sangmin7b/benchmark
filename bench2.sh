for n in 64 128 256 512 1024 2048 4096; do
    ./fft_benchmark -N $n --e2e > bench_comm_e2e_$n.txt
done;
