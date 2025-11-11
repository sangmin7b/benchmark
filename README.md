## Benchmark

type: 
- 0 (float) 
- 1 (half)

```bash
./fft-benchmark -N 1024 --type 1
```

--default
: run cuFFTDx with default configuration (FPB, EPT) 

--smem
: benchmark smem kernel  

## cuFFTDx Parameter Tuning (FPB, EPT)

after build with -DSEARCH

```bash
./fft-benchmark -N 1024 --search --type 1
```
