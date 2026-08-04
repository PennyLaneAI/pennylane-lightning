Kernel performance tuning
#########################

OpenMP threaded kernels
-----------------------

OpenMP acceleration of gate kernels across all kernel types (LM, AVX2, and AVX512) is enabled
by default on Linux and MacOS wheels in Lightning-Qubit.

If building from source, OpenMP support can be enabled by compiling with the CMake
flag ``-DLQ_ENABLE_KERNEL_OMP=ON``. Once enabled, you can control the number of threads used
at runtime by setting the ``OMP_NUM_THREADS`` environment variable before starting your
Python session, or if already running, before simulating your PennyLane programs.

For workloads that involve gradient computations with many observable measurements,
OpenMP acceleration may reduce performance due to oversubscription of threads to CPU cores.
To mitigate this, use the CMake flag ``-DLQ_ENABLE_KERNEL_OMP=OFF`` when building
Lightning-Qubit.

For workloads that show benefit from the use of threaded gate kernels,
sometimes updating the CPU cache to accommodate recently modified data can become a bottleneck,
and saturates the performance gained at high thread counts.
This may be alleviated somewhat on systems supporting AVX2 and AVX-512 operations using
the ``-DLQ_ENABLE_KERNEL_AVX_STREAMING=on`` CMake flag. This forces the data to avoid updating
the CPU cache and can improve performance for larger workloads.
