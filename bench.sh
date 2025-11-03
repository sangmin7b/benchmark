# for n in 4096; do
#     stdbuf -oL -eL ./fft_benchmark -N $n --search --type 1 | tee h_search_s_$n.txt 
#     notify "$n"
# done;


for n in 4096; do
    stdbuf -oL -eL ./fft_benchmark -N $n --type 1 > h_bench_s_$n.txt 
    notify "$n"
done;


# for n in 128 512 2048; do
#     ./fft_benchmark -N $n --type 1 > h_test_$n.txt
#     notify "$n half"
# done;