============================================  =======================================
Pennylane Lightning ``Cmake`` option           Description
============================================  =======================================
``-DENABLE_BLAS=OFF``                         Enable BLAS
``-DENABLE_CLANG_TIDY=OFF``                   Enable clang-tidy build checks
``-DENABLE_COVERAGE=OFF``                     Enable code coverage
``-DENABLE_GATE_DISPATCHER=ON``               Enable gate kernel dispatching on AVX/AVX2/AVX512
``-DENABLE_LAPACK=OFF``                       Enable compilation with scipy/LAPACK
``-DENABLE_NATIVE=OFF``                       Enable native CPU build tuning
``-DENABLE_OPENMP=ON``                        Enable OpenMP
``-DENABLE_PYTHON=ON``                        Enable compilation of the Python module
``-DENABLE_WARNINGS=ON``                      Enable warnings
``-DLQ_ENABLE_KERNEL_AVX_STREAMING=OFF``      Enable AVX2/512 streaming operations for gate kernels
``-DLQ_ENABLE_KERNEL_OMP=OFF``                Enable OpenMP pragmas for gate kernels
============================================  =======================================

============================================  =======================================
Pennylane Lightning Kokkos ``Cmake`` option   Description
============================================  =======================================
``-DKokkos_ENABLE_SERIAL=ON``                 Enable Kokkos SERIAL  device
``-DKokkos_ENABLE_OPENMP=OFF``                Enable Kokkos OPENMP  device
``-DKokkos_ENABLE_THREADS=OFF``               Enable Kokkos THREADS device
``-DKokkos_ENABLE_HIP=OFF``                   Enable Kokkos HIP     device
``-DKokkos_ENABLE_CUDA=OFF``                  Enable Kokkos CUDA    device
``-DKokkos_ENABLE_COMPLEX_ALIGN=OFF``         Enable complex alignment in memory
``-DPLKOKKOS_ENABLE_NATIVE=OFF``              Enable native CPU build tuning
``-DPLKOKKOS_ENABLE_SANITIZER=OFF``           Enable address sanitizer
``-DPLKOKKOS_ENABLE_WARNINGS=ON``             Enable warnings
============================================  =======================================

============================================  =======================================
Pennylane Lightning GPU ``Cmake`` option      Description
============================================  =======================================
``-DPLLGPU_DISABLE_CUDA_SAFETY=OFF``          Build without CUDA call safety checks
============================================  =======================================

============================================  =======================================
Pennylane Lightning Test``Cmake`` option      Description
============================================  =======================================
``-DBUILD_BENCHMARKS=OFF``                    Enable cpp benchmarks
``-DBUILD_TESTS=OFF``                         Build cpp tests
============================================  =======================================
