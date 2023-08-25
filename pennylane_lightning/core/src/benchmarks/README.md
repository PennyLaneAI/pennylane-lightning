# Lightning Google-Benchmark Suite 
The PennyLane-Lightning benchmark suite powered by [google-benchmark](https://github.com/google/benchmark) (GB). 
To use GB scripts, you can perform `make gbenchmark` or run 
```console
$ cmake pennylane_lightning/src/ -BBuildGBench -DBUILD_BENCHMARKS=ON -DENABLE_OPENMP=ON -DENABLE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./BuildGBench
```

## Google-Benchmark
The main requirement for these scripts is [google-benchmark](https://github.com/google/benchmark). 
The CMake uses `FetchContent` to fetch and install the library if the `find_package` command fails 
to find and load GB.

### GB CLI Flags
```console
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

## Implementation details
`make gbenchmark` compiles benchmark executables from,
- `Bench_BitUtil.cpp`,
- `Bench_LinearAlgebra.cpp`,
- `Bench_ApplyOperations.cpp`,
- `Bench_Kernels.cpp`.


### `benchmarks/utils`
To benchmark the linear algebra and bit operations in PennyLane-Lightning, one can run:
```console
$ make gbenchmark
$ ./BuildGBench/benchmarks/utils 
```

The `std_innerProd_cmplx<T>` method in `Bench_LinearAlgebra.cpp` runs benchmarks computing 
the inner product of two randomly generated vectors with complex numbers (`std::complex<T>`) 
using `std::inner_product`. 
If one wants to try a few arguments in some ranges and generate a benchmark for each such value,
GB offers `Range` and `Ranges` to do so, 
```C
BENCHMARK(std_innerProd_cmplx<float>)
    ->Range(1l << 5, 1l << 10);
```

By default the arguments in the range are generated in multiples of 8 and the command above picks
`{32, 64, 512, 1024}`. To change the range multiplier, one should use `RangeMultiplier`. 
The following code selects `{4, 8, 16, 32, 64, 128, 256, 512, 1024}`,
```C
BENCHMARK(std_innerProd_cmplx<float>)
    ->RangeMultiplier(2)
    ->Range(1l << 2, 1l << 10);
```

To filter the results one can use regex and `--benchmark_filter`. For example, 
the following command runs only `std_innerProd_cmplx` benchmark tests in `./benchmarks/utils`:

```console
$ ./BuildGBench/benchmarks/utils --benchmark_filter=std_innerProd_cmplx

Running ./BuildGBench/benchmarks/utils
Run on (8 X 2270.36 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 1280 KiB (x4)
  L3 Unified 12288 KiB (x1)
Load Average: 0.34, 0.38, 0.48
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
std_innerProd_cmplx<float>/32          20.8 ns         20.8 ns     27948132
std_innerProd_cmplx<float>/64          43.2 ns         43.2 ns     16170905
std_innerProd_cmplx<float>/512          420 ns          420 ns      1665298
std_innerProd_cmplx<float>/1024         847 ns          847 ns       817415
std_innerProd_cmplx<double>/32         26.7 ns         26.7 ns     28857715
std_innerProd_cmplx<double>/64         62.6 ns         62.6 ns     10719664
std_innerProd_cmplx<double>/512         518 ns          518 ns      1305655
std_innerProd_cmplx<double>/1024       1028 ns         1028 ns       662981
```

Besides, one can use `--benchmark_time_unit` to get the results in `ns`, `us`, `ms`, or `s`. 
Check **GB CLI Flags** for the list of flags. 


### `benchmarks/pennylane_lightning_bench_operations`
To benchmark the `Pennylane::StateVectorManagedCPU` and `applyOperation` in PennyLane-Lightning, one can run:
```console
$ make gbenchmark
$ ./BuildGBench/benchmarks/pennylane_lightning_bench_operations
```

The following arguments could be altered:
- `Pennylane::Gates::KernelType`;
- Floating point precision type;
- List of gates that script would randomly pick from to apply to the state vector; 
- Range for the number of gates to be applied; and 
- Range for the number of qubits. 

For example, in the code below, 
- `applyOperations_RandOps` is the name of the method in `Bench_ApplyOperations.cpp`, 
- `double` is the floating point precision type,
- `LM_RXYZ` is just a name for the parameters, 
- `Kernel::LM` is the kernel type,
- `{"RX", "RY", "RZ"}` is the list of gates,
- `CreateRange(8, 16, 2)` creates a (sparse) range for the number of gates, and 
- `CreateDenseRange(6, 10, 2)` creates a (dense) range for the number of qubits.

```C
BENCHMARK_APPLYOPS(applyOperations_RandOps, double, LM_RXYZ, Kernel::LM,
                   {"RX", "RY", "RZ"})
    ->ArgsProduct({
        benchmark::CreateRange(8, 16, /*mul=*/2),       // num_gates
        benchmark::CreateDenseRange(6, 10, /*step=*/2), // num_qubits
    });
```

```console
Run on (8 X 3002.54 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 1280 KiB (x4)
  L3 Unified 12288 KiB (x1)
Load Average: 0.81, 0.66, 0.50
----------------------------------------------------------------------------------------
Benchmark                                              Time             CPU   Iterations
----------------------------------------------------------------------------------------
applyOperations_RandOps<double>/LM_RXYZ/8/6         1254 ns         1254 ns       529355
applyOperations_RandOps<double>/LM_RXYZ/16/6        2630 ns         2630 ns       281743
applyOperations_RandOps<double>/LM_RXYZ/8/8         3794 ns         3794 ns       171244
applyOperations_RandOps<double>/LM_RXYZ/16/8        7969 ns         7969 ns        82555
applyOperations_RandOps<double>/LM_RXYZ/8/10       14773 ns        14773 ns        46396
applyOperations_RandOps<double>/LM_RXYZ/16/10      30174 ns        30174 ns        23573
```

One can use `--benchmark_format` to get the results in other formats: `<console|json|csv>`. 
Check **GB CLI Flags** for the list of flags. 


### `benchmarks/pennylane_lightning_bench_kernels`
To benchmark the `Pennylane::StateVectorManagedCPU` for all gates, generators, matrix operations using different kernels:
```console
$ make gbenchmark
$ ./BuildGBench/benchmarks/pennylane_lightning_bench_kernels
```

The output is
```console
Run on (8 X 4800 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 1280 KiB (x4)
  L3 Unified 12288 KiB (x1)
Load Average: 2.05, 1.92, 1.82
CPU::AVX: True
CPU::AVX2: True
CPU::AVX512F: True
CPU::Brand: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz
CPU::Vendor: GenuineIntel
Compiler::AVX2: 0
Compiler::AVX512F: 0
Compiler::Name: GCC
Compiler::Version: 9.4.0
--------------------------------------------------------------------------------------
Benchmark                                            Time             CPU   Iterations
--------------------------------------------------------------------------------------
PauliX<float>/LM/6                                2576 ns         2575 ns       274752
PauliX<float>/LM/8                                6522 ns         6521 ns       105910
PauliX<float>/LM/10                              22811 ns        22810 ns        30130
PauliX<float>/LM/12                              87670 ns        87646 ns         7964
PauliX<float>/LM/14                             349718 ns       349693 ns         2020
PauliX<float>/LM/16                            1393902 ns      1393828 ns          491
PauliX<float>/LM/18                            5782506 ns      5781553 ns          123
PauliX<float>/LM/20                           25763147 ns     25761755 ns           26
PauliX<float>/LM/22                          122700064 ns    122694232 ns            5
```

We provide a simple benchmark script and drawing utilities:
```console
$ ./benchmarks/benchmark_all.sh
```
will record the results to `bench_result.json` file in the current directory. You may draw corresponding plots with
```console
$ ./plot_gate_benchmark.py bench_result.json
```
The plots will be available inside `./plots` directory. To only draw results for `float` or `double`, you can provide the arumgnet to the script
```console
$ ./plot_gate_benchmark.py bench_result.json (float|double)
```



## GB Compare Tooling
One can use [`compare.py`](https://github.com/google/benchmark/blob/main/tools/compare.py) to compare the results of the GB scripts. 

### Compare float vs double 
```console
$ python3 compare.py filters ./BuildGBench/benchmarks/utils float double

Run on (8 X 1853.84 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 1280 KiB (x4)
  L3 Unified 12288 KiB (x1)
Load Average: 2.14, 1.03, 0.72

Comparing float to double (from ./BuildGBench/benchmarks/utils)
Benchmark                                                             Time             CPU      Time Old      Time New       CPU Old       CPU New
--------------------------------------------------------------------------------------------------------------------------------------------------
create_random_cmplx_vector<[float vs. double]>/32                  +0.1779         +0.1809           756           890           754           890
create_random_cmplx_vector<[float vs. double]>/64                  +0.1846         +0.1846          1446          1713          1446          1713
create_random_cmplx_vector<[float vs. double]>/512                 +0.2030         +0.2030         10894         13105         10894         13105
create_random_cmplx_vector<[float vs. double]>/1024                +0.1904         +0.1904         21741         25879         21741         25879
std_innerProd_cmplx<[float vs. double]>/32                         +0.4076         +0.4075            21            29            21            29
std_innerProd_cmplx<[float vs. double]>/64                         +0.2559         +0.2559            50            63            50            63
std_innerProd_cmplx<[float vs. double]>/512                        +0.0820         +0.0820           481           520           481           520
std_innerProd_cmplx<[float vs. double]>/1024                       +0.0677         +0.0677           964          1030           964          1030
omp_innerProd_cmplx<[float vs. double]>/32                         -0.0439         -0.0439           201           193           201           193
omp_innerProd_cmplx<[float vs. double]>/64                         -0.0501         -0.0501           237           225           237           225
omp_innerProd_cmplx<[float vs. double]>/512                        +0.0556         +0.0556           661           698           661           698
omp_innerProd_cmplx<[float vs. double]>/1024                       +0.0599         +0.0599          1159          1228          1159          1228
blas_innerProd_cmplx<[float vs. double]>/32                        +1.6790         +1.6789            12            33            12            33
blas_innerProd_cmplx<[float vs. double]>/64                        +2.4392         +2.4391            21            73            21            73
blas_innerProd_cmplx<[float vs. double]>/512                       +3.1349         +3.1348           139           577           139           577
blas_innerProd_cmplx<[float vs. double]>/1024                      +3.1949         +3.1947           281          1177           281          1177
naive_transpose_cmplx<[float vs. double]>/32                       +1.5293         +1.5293           867          2192           867          2192
naive_transpose_cmplx<[float vs. double]>/64                       +5.6706         +5.6704          3104         20706          3104         20706
naive_transpose_cmplx<[float vs. double]>/512                      +1.3715         +1.3714        772694       1832435        772680       1832363
naive_transpose_cmplx<[float vs. double]>/1024                     +1.8269         +1.8266       3528387       9974273       3528267       9972909
cf_transpose_cmplx<[float vs. double], 16>/32                      +1.5507         +1.5507           754          1924           754          1924
cf_transpose_cmplx<[float vs. double], 16>/64                      +1.9015         +1.9012          3245          9416          3245          9415
cf_transpose_cmplx<[float vs. double], 16>/512                     +1.3707         +1.3708        687593       1630107        687562       1630078
cf_transpose_cmplx<[float vs. double], 16>/1024                    +1.4324         +1.4324       3019284       7344169       3019256       7343956
cf_transpose_cmplx<[float vs. double], 32>/32                      -0.1493         -0.1493           860           732           860           732
cf_transpose_cmplx<[float vs. double], 32>/64                      +0.1806         +0.1806          3202          3780          3202          3780
cf_transpose_cmplx<[float vs. double], 32>/512                     +0.1310         +0.1310        692794        783577        692789        783567
cf_transpose_cmplx<[float vs. double], 32>/1024                    +0.3344         +0.3344       3035260       4050185       3035148       4050084
omp_matrixVecProd_cmplx<[float vs. double]>/16                     +0.4616         +0.4612          2173          3177          2173          3175
omp_matrixVecProd_cmplx<[float vs. double]>/64                     +0.8316         +0.8329          4407          8072          4401          8066
omp_matrixVecProd_cmplx<[float vs. double]>/256                    +0.3514         +0.3484         43206         58388         43169         58209
blas_matrixVecProd_cmplx<[float vs. double]>/16                    +0.2655         +0.2655           327           414           327           414
blas_matrixVecProd_cmplx<[float vs. double]>/64                    +0.5087         +0.5475          7924         11956          7684         11891
blas_matrixVecProd_cmplx<[float vs. double]>/256                   -0.1335         -0.1542         34236         29665         33248         28123
omp_matrixMatProd_cmplx<[float vs. double]>/16                     +0.8089         +0.8214          4978          9006          4846          8827
omp_matrixMatProd_cmplx<[float vs. double]>/64                     +0.3022         +0.3174        164390        214068        162245        213740
omp_matrixMatProd_cmplx<[float vs. double]>/256                    +0.1685         +0.2645      10982033      12832305      10108172      12781493
blas_matrixMatProd_cmplx<[float vs. double]>/16                    +0.3376         +0.3376          1813          2426          1813          2426
blas_matrixMatProd_cmplx<[float vs. double]>/64                    -0.2104         -0.1625         75876         59908         70623         59146
blas_matrixMatProd_cmplx<[float vs. double]>/256                   +1.0512         +1.1981       3147247       6455757       2926291       6432367
OVERALL_GEOMEAN                                                    +0.6082         +0.6174             0             0             0             0
```

### Compare LM vs PI
```console
$ python3 compare.py filters ./BuildGBench/benchmarks/apply_operations LM_all PI_all

Run on (8 X 1853.84 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 1280 KiB (x4)
  L3 Unified 12288 KiB (x1)
Load Average: 2.14, 1.03, 0.72

Comparing LM_all to PI_all (from ./BuildGBench/benchmarks/apply_operations)
Benchmark                                                                   Time             CPU      Time Old      Time New       CPU Old       CPU New
--------------------------------------------------------------------------------------------------------------------------------------------------------
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/6                   +1.9704         +1.9704           740          2199           740          2199
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/6                  +2.0117         +2.0122          1674          5043          1674          5043
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/6                  +2.2919         +2.2922          3346         11015          3346         11014
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/6                  +2.5689         +2.5689          5935         21183          5935         21183
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/8                   +2.7298         +2.7300          1786          6663          1786          6663
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/8                  +3.3816         +3.3817          4096         17949          4096         17949
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/8                  +3.3162         +3.3160          8413         36313          8413         36312
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/8                  +3.3376         +3.3383         17901         77648         17898         77648
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/10                  +0.8145         +0.8149         12753         23140         12750         23140
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/10                 +3.3284         +3.3283         12224         52910         12224         52908
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/10                 +3.5780         +3.5782         29088        133166         29087        133165
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/10                 +3.6910         +3.6910         57028        267523         57027        267516
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/12                  +2.1417         +2.1422         34042        106947         34035        106946
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/12                 +1.2923         +1.2924         69446        159194         69445        159193
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/12                 +0.3308         +0.3311        123251        164026        123225        164027
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/12                 +0.4931         +0.4932        228365        340976        228358        340977
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/14                  +0.2166         +0.2170        166710        202824        166660        202826
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/14                 +0.2849         +0.2850        290310        373017        290278        373015
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/14                 +0.7255         +0.7255        487546        841280        487537        841262
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/14                 +0.2781         +0.2782       1183131       1512171       1183031       1512189
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/16                  +0.4468         +0.4468        603890        873689        603850        873680
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/16                 +0.7906         +0.7906        818479       1465578        818477       1465574
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/16                 +0.6816         +0.6816       2073721       3487107       2073704       3487078
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/16                 +1.6422         +1.6422       3754217       9919561       3754227       9919574
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/18                  +5.3381         +5.3383       1148955       7282233       1148880       7281975
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/18                 +3.6852         +3.6851       2853039      13366939       2852961      13366542
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/18                 +2.8063         +2.8062       7753666      29512416       7753528      29511707
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/18                 +3.7002         +3.7001      12625921      59344364      12625599      59342202
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/20                  +5.1516         +5.1515       5338621      32840815       5338531      32840042
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/20                 +2.2917         +2.2918      19277997      63457796      19277350      63456982
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/20                 +3.9042         +3.9043      26541069     130163532      26540619     130161877
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/20                 +2.4273         +2.4273      59168965     202787913      59166800     202783283
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/22                  +1.9594         +1.9594      47074334     139312151      47072997     139308537
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/22                 +0.2689         +0.2689      81669457     103631032      81668216     103629527
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/22                 +0.6116         +0.6116     140642752     226654300     140633151     226648168
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/22                 +0.7737         +0.7739     274673406     487190407     274632334     487182892
applyOperations_RandOps<float>/[LM_all vs. PI_all]/8/24                  +1.5876         +1.5877     134665624     348462053     134658560     348456864
applyOperations_RandOps<float>/[LM_all vs. PI_all]/16/24                 +1.1122         +1.1124     299975114     633617035     299955193     633610974
applyOperations_RandOps<float>/[LM_all vs. PI_all]/32/24                 +1.2715         +1.2717     586053918    1331235001     585993913    1331231522
applyOperations_RandOps<float>/[LM_all vs. PI_all]/64/24                 +2.6378         +2.6360    1310929400    4768865343    1310874809    4766317332
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/6                  +4.3271         +4.3269          1111          5917          1111          5917
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/6                 +4.8872         +4.8872          2067         12167          2067         12166
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/6                 +1.6727         +1.6728          3856         10306          3856         10306
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/6                 +2.0128         +2.0128          7517         22649          7517         22647
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/8                  +1.2648         +1.2649          2667          6041          2667          6041
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/8                 +1.5491         +1.5495          4749         12107          4749         12106
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/8                 +1.2613         +1.2615         10592         23952         10591         23952
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/8                 +1.3703         +1.3703         19844         47037         19844         47036
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/10                 +0.4299         +0.4299         11146         15937         11146         15937
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/10                +1.5375         +1.5374         16193         41090         16193         41088
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/10                +2.1457         +2.1459         42644        134147         42642        134144
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/10                +1.1057         +1.1057        125253        263739        125246        263736
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/12                 +0.5885         +0.5886         78426        124582         78420        124578
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/12                +0.9362         +0.9362        115317        223274        115315        223275
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/12                +0.6730         +0.6736        288087        481955        287962        481934
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/12                +0.8035         +0.8038        562554       1014588        562473       1014591
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/14                 +0.8138         +0.8137        321458        583072        321441        583007
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/14                +1.3614         +1.3614        441768       1043179        441746       1043136
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/14                -0.2703         -0.2703       1039344        758407       1039229        758344
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/14                -0.3541         -0.3540       2447532       1580937       2447451       1580945
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/16                 -0.2848         -0.2846       1168091        835419       1167696        835421
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/16                -0.2519         -0.2519       2576772       1927698       2576744       1927609
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/16                -0.1967         -0.1967       4287545       3444148       4287377       3444219
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/16                +0.9295         +0.9297       3517705       6787403       3517485       6787529
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/18                 +0.6427         +0.6431       2452274       4028443       2451798       4028488
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/18                +0.8408         +0.8409       5010837       9224016       5010720       9224049
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/18                +2.4354         +2.4354       8253116      28352787       8253155      28353130
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/18                +3.0803         +3.0807      15936887      65027739      15934953      65025322
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/20                 +2.1608         +2.1611      11403578      36044911      11401519      36040897
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/20                +1.7726         +1.7726      23552648      65301677      23551243      65298261
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/20                +1.7032         +1.7036      50289215     135940574      50283336     135945470
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/20                +1.0846         +1.0854     121426703     253123057     121355090     253072588
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/22                 +0.3131         +0.3113     136573314     179341161     136557503     179070048
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/22                -0.2532         -0.2532     164691973     122992503     164689541     122992680
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/22                -0.2946         -0.2946     339590201     239558327     339567138     239522753
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/22                -0.2928         -0.2927     739949287     523301029     739828092     523291634
applyOperations_RandOps<double>/[LM_all vs. PI_all]/8/24                 -0.2177         -0.2177     559569887     437775862     559545239     437751825
applyOperations_RandOps<double>/[LM_all vs. PI_all]/16/24                +0.0081         +0.0082     789837195     796244100     789800171     796240876
applyOperations_RandOps<double>/[LM_all vs. PI_all]/32/24                -0.0240         -0.0239    1511013927    1474718667    1510856430    1474724199
applyOperations_RandOps<double>/[LM_all vs. PI_all]/64/24                +1.8154         +1.8156    1931502602    5438015975    1931394005    5438018004
OVERALL_GEOMEAN                                                          +1.1905         +1.1906             0             0             0             0
```
