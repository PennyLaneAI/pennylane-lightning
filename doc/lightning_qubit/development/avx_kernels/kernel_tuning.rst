Kernel performance tuning
#########################

Lightning-Qubit's kernel implementations are by default tuned for high throughput single-threaded performance with gradient workloads. To enable this, we add OpenMP threading within the adjoint differentiation method implementation and use SIMD-level intrinsics to ensure fast performance for each given circuit in such a workload.

However, sometimes we may want to modify the above defaults to favour a given workload, such as by enabling multi-threaded execution of the gate kernels instead. For this, we have several compile-time flags to change the operating behaviour of Lightning-Qubit kernels.

OpenMP threaded kernels
-----------------------

To enable OpenMP acceleration of the gate kernels, Lightning-Qubit can be compiled with the ``-DLQ_ENABLE_KERNEL_OMP=ON`` CMake flag. Not, that for gradient workloads with many observables, this may reduce performance in comparison with the default mode, so this behaviour is opt-in only.

For workloads that show benefit from the use of threaded gate kernels, sometimes updating the CPU cache to accommodate recently modified data can become a bottleneck, and saturates the performance gained at high thread counts. This may be alleviated somewhat on systems supporting AVX2 and AVX-512 operations using the ``-DLQ_ENABLE_KERNEL_AVX_STREAMING=on`` CMake flag. This forces the data to avoid updating the CPU cache and can improve performance for larger workloads.