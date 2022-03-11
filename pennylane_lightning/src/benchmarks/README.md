# Lightning Google-Benchmark Suite 

This is the PennyLane-Lightning benchmark suite powered by [google-benchmark](https://github.com/google/benchmark) (GB). To use GB scripts, you can perform `make gbenchmark` or run 
```Bash
	$ cmake pennylane_lightning/src/ -BBuildGBench -DBUILD_BENCHMARKS=ON -DENABLE_OPENMP=ON -DENABLE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
	$ cmake --build ./BuildGBench --target utils apply_operations apply_multirz
```

## Google-Benchmark
The main requirement for these scripts is [google-benchmark](https://github.com/google/benchmark). We use the CMake `FetchContent` command to install the library if the `find_package` command fails to find google-benchmark. 

## Google-Benchmark CLI Options
```Bash
benchmark [--benchmark_list_tests={true|false}]
          [--benchmark_filter=<regex>]
          [--benchmark_min_time=<min_time>]
          [--benchmark_repetitions=<num_repetitions>]
          [--benchmark_enable_random_interleaving={true|false}]
          [--benchmark_report_aggregates_only={true|false}]
          [--benchmark_display_aggregates_only={true|false}]
          [--benchmark_format=<console|json|csv>]
          [--benchmark_out=<filename>]
          [--benchmark_out_format=<json|console|csv>]
          [--benchmark_color={auto|true|false}]
          [--benchmark_counters_tabular={true|false}]
          [--benchmark_context=<key>=<value>,...]
          [--benchmark_time_unit={ns|us|ms|s}]
          [--v=<verbosity>]
```

## Implementation details:
The `make gbenchmark` command compiles the benchmark executables from the following files,
- `Bench_BitUtil.cpp`,
- `Bench_LinearAlgebra.cpp`,
- `Bench_ApplyOperations.cpp`,
- `Bench_ApplyMultiRZ.cpp`.


### `benchmarks/utils`
To benchmark the linear algebra and bit opertions in PennyLane-Lightning, you can run:
```Bash
	$ make gbenchmark
    $ ./BuildGBench/benchmarks/utils 
```

For example, in `Bench_LinearAlgebra.cpp`, The `std_innerProd_cmplx<T>` method performs the benchmark for the inner product of two vectors with complex numbers (`std::complex<T>`) using `std::inner_product`. This is usual to try a few arguments in some ranges and generate a benchmark for each such value. The google-benchmark offers `Range` and `Ranges` to do so, 
```C
BENCHMARK(std_innerProd_cmplx<float>)
    ->Range(1l << 5, 1l << 10);
```

By default the arguments in the range are generated in multiples of 8 and the command above picks `{32, 64, 512, 1024}`. To change the range multiplier, one should use `RangeMultiplier`. The following code selects `{4, 8, 16, 32, 64, 128, 256, 512, 1024}`,
```C
BENCHMARK(std_innerProd_cmplx<float>)
    ->RangeMultiplier(2)
    ->Range(1l << 2, 1l << 10);
```

To filter the benchmark results to run only `std_innerProd_cmplx`, you can run:
```Bash
	$ ./BuildGBench/benchmarks/utils --benchmark_filter=std_innerProd_cmplx
```

You can use `--benchmark_time_unit` to get the results in `{ns|us|ms|s}` too. Check **Google-Benchmark CLI Options** for the list of options. 


### `benchmarks/apply_operations`
To benchmark the `Pennylane::StateVectorManaged` and `applyOperation` in PennyLane-Lightning, you can run:
```Bash
	$ make gbenchmark
    $ ./BuildGBench/benchmarks/apply_operations
```

You can alter the follwoing arguments.
- `Pennylane::Gates::KernelType` 
- The list of gates that script would randomly pick from to apply to the state vector. 
- The range for the number of gates to be applied.
- The range for the number of qubits. 

For example, in the code below, 
- `applyOperationsFromRandOps` is the name of the method in `Bench_ApplyOperations.cpp`, 
- `LM_RXYZ` is just a name for the parameters, 
- `Pennylane::Gates::KernelType::LM` is the kernel type,
- `{"RX", "RY", "RZ"}` is the list of gates,
- `{8, 64}` is a range for the number of gates to be randomly picked from the list of gates and be applied, and 
- `{4, 24}` is a range for the number of qubits.  

```C
BENCHMARK_CAPTURE(applyOperationsFromRandOps, LM_RXYZ,
                  Pennylane::Gates::KernelType::LM, {"RX", "RY", "RZ"})
    ->RangeMultiplier(2)
    ->Ranges({{8, 64}, {4, 24}});
```

The results are:

```Bash
-----------------------------------------------------------------------------------
Benchmark                                         Time             CPU   Iterations
-----------------------------------------------------------------------------------
applyOperationsFromRandOps/LM_RXYZ/8/4          800 ns          800 ns       768179
applyOperationsFromRandOps/LM_RXYZ/16/4        1574 ns         1575 ns       448749
applyOperationsFromRandOps/LM_RXYZ/32/4        3130 ns         3130 ns       211491
applyOperationsFromRandOps/LM_RXYZ/64/4        6085 ns         6085 ns       106148
applyOperationsFromRandOps/LM_RXYZ/8/8         5471 ns         5472 ns       110769
applyOperationsFromRandOps/LM_RXYZ/16/8        9533 ns         9534 ns        65119
applyOperationsFromRandOps/LM_RXYZ/32/8       24695 ns        24696 ns        35564
applyOperationsFromRandOps/LM_RXYZ/64/8       40523 ns        40525 ns        17710
applyOperationsFromRandOps/LM_RXYZ/8/16     1191970 ns      1191987 ns          526
applyOperationsFromRandOps/LM_RXYZ/16/16    2384587 ns      2384050 ns          290
applyOperationsFromRandOps/LM_RXYZ/32/16    4656297 ns      4655338 ns          130
applyOperationsFromRandOps/LM_RXYZ/64/16    9520071 ns      9518236 ns           71
applyOperationsFromRandOps/LM_RXYZ/8/24   520595743 ns    520473426 ns            1
applyOperationsFromRandOps/LM_RXYZ/16/24  833631105 ns    833427865 ns            1
applyOperationsFromRandOps/LM_RXYZ/32/24 1572258140 ns   1571911717 ns            1
applyOperationsFromRandOps/LM_RXYZ/64/24 3219987188 ns   3219288609 ns            1
```

You can use `--benchmark_format` to get the results in other formats: `<console|json|csv>`. Check **Google-Benchmark CLI Options** for the list of options. 


### `benchmarks/apply_multirz`
To benchmark the `Pennylane::StateVectorManaged` and `applyOperation` specificly for `"MultiRZ"` in PennyLane-Lightning, you can run:
```Bash
	$ make gbenchmark
    $ ./BuildGBench/benchmarks/apply_multirz
```


For example, in the code below, 
- `applyOperation_MultiRZ` is the name of the method in `Bench_ApplyMultiRZ.cpp`, 
- `kernel_LM` is just a name for the parameters, 
- `Pennylane::Gates::KernelType::LM` is the kernel type,
- `{8, 64}` is a range for the number of gates to be randomly picked from the list of gates and be applied, 
- `{4, 24}` is a range for the number of qubits, and
- `{2, 4}` is a range for the number of wires to be considered in `"MultiRZ"`.

```C
BENCHMARK_CAPTURE(applyOperation_MultiRZ, kernel_LM,
                  Pennylane::Gates::KernelType::LM)
    ->RangeMultiplier(2)
    ->Ranges({{8, 64}, {4, 24}, {2, 4}});
```

The results are:

```Bash
-----------------------------------------------------------------------------------
Benchmark                                         Time             CPU   Iterations
-----------------------------------------------------------------------------------
applyOperation_MultiRZ/kernel_LM/8/4/2          751 ns          751 ns       934715
applyOperation_MultiRZ/kernel_LM/16/4/2        1527 ns         1527 ns       464098
applyOperation_MultiRZ/kernel_LM/32/4/2        3962 ns         3962 ns       223937
applyOperation_MultiRZ/kernel_LM/64/4/2        8870 ns         8870 ns        74256
applyOperation_MultiRZ/kernel_LM/8/8/2         9308 ns         9308 ns        72495
applyOperation_MultiRZ/kernel_LM/16/8/2       17713 ns        17713 ns        38438
applyOperation_MultiRZ/kernel_LM/32/8/2       36395 ns        36395 ns        19501
applyOperation_MultiRZ/kernel_LM/64/8/2       70646 ns        70644 ns         9136
applyOperation_MultiRZ/kernel_LM/8/16/2     2152253 ns      2152151 ns          299
applyOperation_MultiRZ/kernel_LM/16/16/2    4578853 ns      4578747 ns          159
applyOperation_MultiRZ/kernel_LM/32/16/2    8879127 ns      8878146 ns           75
applyOperation_MultiRZ/kernel_LM/64/16/2   17638414 ns     17635982 ns           39
applyOperation_MultiRZ/kernel_LM/8/24/2   757876645 ns    757861150 ns            1
applyOperation_MultiRZ/kernel_LM/16/24/2 1402177984 ns   1402021613 ns            1
applyOperation_MultiRZ/kernel_LM/32/24/2 2733113910 ns   2733035141 ns            1
applyOperation_MultiRZ/kernel_LM/64/24/2 5380131547 ns   5379970051 ns            1
applyOperation_MultiRZ/kernel_LM/8/4/4         1116 ns         1116 ns       585217
applyOperation_MultiRZ/kernel_LM/16/4/4        2231 ns         2231 ns       321302
applyOperation_MultiRZ/kernel_LM/32/4/4        4343 ns         4343 ns       139047
applyOperation_MultiRZ/kernel_LM/64/4/4        8840 ns         8840 ns        75577
applyOperation_MultiRZ/kernel_LM/8/8/4         8901 ns         8901 ns        72539
applyOperation_MultiRZ/kernel_LM/16/8/4       18131 ns        18131 ns        38946
applyOperation_MultiRZ/kernel_LM/32/8/4       35478 ns        35477 ns        19646
applyOperation_MultiRZ/kernel_LM/64/8/4       71719 ns        71719 ns         8025
applyOperation_MultiRZ/kernel_LM/8/16/4     2179260 ns      2179260 ns          320
applyOperation_MultiRZ/kernel_LM/16/16/4    4275326 ns      4275289 ns          164
applyOperation_MultiRZ/kernel_LM/32/16/4    8685350 ns      8685181 ns           74
applyOperation_MultiRZ/kernel_LM/64/16/4   16877559 ns     16877134 ns           40
applyOperation_MultiRZ/kernel_LM/8/24/4   756551377 ns    756517719 ns            1
applyOperation_MultiRZ/kernel_LM/16/24/4 1390896340 ns   1390871112 ns            1
applyOperation_MultiRZ/kernel_LM/32/24/4 2810216299 ns   2809949708 ns            1
applyOperation_MultiRZ/kernel_LM/64/24/4 5512339216 ns   5511768787 ns            1
```
