Kernel performance tuning
#########################

Lightning-Qubit's kernel implementations are by default tuned for high throughput single-threaded performance with gradient workloads.
To enable this, we add OpenMP threading within the adjoint differentiation method implementation
and use SIMD-level intrinsics to ensure fast performance for each given circuit in such a workload.

However, sometimes we may want to modify the above defaults to favour a given workload, such as by enabling multi-threaded execution of the gate kernels instead.
For this, we have several compile-time flags to change the operating behaviour of Lightning-Qubit kernels.

OpenMP threaded kernels
-----------------------

OpenMP acceleration of gate kernels across all kernel types (LM, AVX2, and AVX512) is enabled
by default on Linux wheels in Lightning-Qubit.

On other operating systems, OpenMP support can be enabled by setting the environment variable
``LQ_ENABLE_KERNEL_OMP=ON`` before starting your Python session, or if already running, before
simulating your PennyLane programs.
You can also control the number of threads used by setting the ``OMP_NUM_THREADS``
environment variable.

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
