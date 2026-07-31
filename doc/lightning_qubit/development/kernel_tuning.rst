Kernel performance tuning
#########################

Lightning-Qubit exposes a number of compile-time options that control how gate kernels are
built and selected. This page describes the options relevant to performance, their defaults,
and when changing them is likely to help.

All options are passed to CMake when building from source, e.g. via the ``CMAKE_ARGS``
environment variable:

.. code-block:: bash

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DLQ_ENABLE_KERNEL_OMP=ON" \
        pip install -e . --config-settings editable_mode=compat -vv

Summary of options
------------------

.. list-table::
   :header-rows: 1
   :widths: 32 12 56

   * - CMake option
     - Default
     - Effect
   * - ``CMAKE_BUILD_TYPE``
     - ``RelWithDebInfo``
     - Optimization level of the build. Released wheels are built with ``Release``.
   * - ``ENABLE_GATE_DISPATCHER``
     - ``ON``
     - Compile the AVX2/AVX512 kernels and the runtime kernel map. x86-64 UNIX only.
   * - ``ENABLE_OPENMP``
     - ``ON``
     - Find and link OpenMP. Required for any OpenMP-threaded kernel to run.
   * - ``LQ_ENABLE_KERNEL_OMP``
     - ``OFF``
     - Enable the OpenMP pragmas inside the gate kernels.
   * - ``LQ_ENABLE_KERNEL_AVX_STREAMING``
     - ``OFF``
     - Use AVX2/AVX512 non-temporal (streaming) stores in the gate kernels.
   * - ``ENABLE_BLAS``
     - ``OFF``
     - Link a BLAS implementation (MKL, or CBLAS) for the linear-algebra routines.

Build type
----------

If ``CMAKE_BUILD_TYPE`` is not specified, Lightning defaults to ``RelWithDebInfo``.
Released wheels are built with ``-DCMAKE_BUILD_TYPE=Release``, so a source build should
set this explicitly when comparing performance against an installed wheel.

Gate kernel dispatching
-----------------------

Lightning-Qubit ships several kernel families: the generic ``LM`` kernels and specialized
AVX2 and AVX512 kernels. The ``-DENABLE_GATE_DISPATCHER=ON`` flag (the default) compiles all
of them into the binary and registers the AVX2/AVX512 kernels with the dynamic dispatcher only
when the runtime CPU and the memory alignment of the state vector support them. See
:doc:`avx_kernels/build_system` for details of how this works.

The AVX2/AVX512 kernels are only built on x86-64 UNIX systems. On other platforms, and when
building with ``-DENABLE_GATE_DISPATCHER=OFF``, only the ``LM`` kernels are compiled and used.
On macOS, the build system sets ``-DENABLE_GATE_DISPATCHER=OFF`` automatically, so options
that affect the AVX kernels have no effect there.

OpenMP threaded kernels
-----------------------

OpenMP acceleration of gate kernels across all kernel types (LM, AVX2, and AVX512) is enabled
in the Linux and macOS wheels of Lightning-Qubit.

When building from source it is disabled by default, and can be enabled by compiling with
the CMake flag ``-DLQ_ENABLE_KERNEL_OMP=ON``. This requires ``-DENABLE_OPENMP=ON`` (the
default); with OpenMP disabled, or if the OpenMP headers are unavailable, the gate kernels
fall back to single-threaded execution regardless of ``LQ_ENABLE_KERNEL_OMP``.

Once enabled, you can control the number of threads used at runtime by setting the
``OMP_NUM_THREADS`` environment variable before starting your Python session, or if already
running, before simulating your PennyLane programs.

For workloads that involve gradient computations with many observable measurements,
OpenMP acceleration may reduce performance due to oversubscription of threads to CPU cores.
To mitigate this, use the CMake flag ``-DLQ_ENABLE_KERNEL_OMP=OFF`` when building
Lightning-Qubit.

AVX streaming operations
------------------------

For workloads that show benefit from the use of threaded gate kernels,
sometimes updating the CPU cache to accommodate recently modified data can become a bottleneck,
and saturates the performance gained at high thread counts.
This may be alleviated somewhat on systems supporting AVX2 and AVX-512 operations using
the ``-DLQ_ENABLE_KERNEL_AVX_STREAMING=ON`` CMake flag. This forces the data to avoid updating
the CPU cache and can improve performance for larger workloads.

This option only takes effect together with ``-DLQ_ENABLE_KERNEL_OMP=ON`` and with the
AVX2/AVX512 kernels compiled in, i.e. ``-DENABLE_GATE_DISPATCHER=ON`` on an x86-64 UNIX system.
Enabling it without OpenMP kernels emits a CMake warning at configure time.

BLAS backend
------------

The ``-DENABLE_BLAS=ON`` flag links Lightning-Qubit against a BLAS implementation, preferring
MKL when found and falling back to CBLAS. This does not affect the gate kernels themselves,
but is used by the linear-algebra routines backing operations such as expectation values of
matrix-valued observables.
