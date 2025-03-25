# LargeScaleSW
## SW(Smith-Waterman) algorithm in CPU & GPU version to caculate mtDNA & nDNA alignment.

## Introduction
[Refernce](https://dl.acm.org/doi/10.1145/1837853.1693473):CUDAlign: using GPU to accelerate the comparison of megabase genomic sequences

This tool, referencing the aforementioned paper, optimizes the algorithm and GPU memory usage to implement a GPU-based version of local alignment with an affine gap. It is applied to align mtDNA and nDNA, which have significantly different lengths, but can in fact be used for any sequence alignment involving one short and one long sequence.
## How to use
:::danger
You need to expand your mtDNA data first if you want to use expand alignment function
:::
### CPU
```shell
# switch to cpu.c folder first
gcc ./cpu.c -o cpu
./cpu <mtDNA Path> <nDNA Path>
```

### GPU
```shell
# switch to gpu.cu folder first
nvcc ./gpu.cu -o gpu
gpu <mtDNA Data Path> <nDNA Data Path> <threads Per Block> <expand nDNA option 0 = no 1 = yes>
```

### run in different threads
```shell
nvcc ./gpu.cu -o gpu
gcc ./experiment.c -0 experiment
./experiment <nDNA file name int output folder>
```
