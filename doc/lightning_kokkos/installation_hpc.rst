Lightning-Kokkos installation on HPC
************************************

The Lightning-Kokkos simulator is well suited for various parallel HPC platforms. To install Lightning-Kokkos on HPC systems, we provide the following example for building from source. Please also consult the documentation of your HPC system for specific instructions on how to load modules and set up the environment.

Building and Running Lightning-Kokkos with MPI on Frontier
==========================================================

Here we demonstrate the steps to build Lightning-Kokkos with MPI on `Frontier <https://www.olcf.ornl.gov/frontier/>`_ for AMD GPUs.

We can load the following modules to enable the relevant compilers and Python environment:

.. code-block:: console

    # Load the required Python and compiler modules
    module load cray-python
    module load PrgEnv-amd


Install Kokkos (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    Lightning-Kokkos was tested with Kokkos version <= 4.5.0
    
We suggest first installing Kokkos with the wanted configuration following the instructions found in the Kokkos documentation. For example, the following will build Kokkos for AMD MI210/250/250X GPUs.

Download the `Kokkos code <https://github.com/kokkos/kokkos/releases>`_.

.. code-block:: bash

    # Replace x, y, and z by the correct version
    wget https://github.com/kokkos/kokkos/archive/refs/tags/4.x.yz.tar.gz
    tar -xvf 4.x.y.z.tar.gz
    cd kokkos-4.x.y.z

Build Kokkos for AMD GPU (``GFX90A`` architecture), and append the install location to ``CMAKE_PREFIX_PATH``.

.. code-block:: console

    # Install Kokkos:
    export KOKKOS_INSTALL_PATH=<install-path>
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_PATH \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DBUILD_SHARED_LIBS:BOOL=ON \
        -DBUILD_TESTING:BOOL=ON \
        -DKokkos_ENABLE_SERIAL:BOOL=ON \
        -DKokkos_ENABLE_HIP:BOOL=ON \
        -DKokkos_ARCH_AMD_GFX90A:BOOL=ON \
        -DKokkos_ENABLE_COMPLEX_ALIGN:BOOL=OFF \
        -DKokkos_ENABLE_EXAMPLES:BOOL=OFF \
        -DKokkos_ENABLE_TESTS:BOOL=OFF \
        -DKokkos_ENABLE_LIBDL:BOOL=OFF
    cmake --build build && cmake --install build
    export CMAKE_PREFIX_PATH=$KOKKOS_INSTALL_PATH  


Install Lightning-Kokkos
^^^^^^^^^^^^^^^^^^^^^^^^

Lightning-Qubit needs to be 'installed' by ``pip`` before Lightning-Kokkos (compilation is not necessary).

.. code-block:: console

    # Install Lightning-Qubit
    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    pip install -r requirements.txt
    pip install git+https://github.com/PennyLaneAI/pennylane.git@master
    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DCMAKE_CXX_COMPILER=CC" pip install .

Then to install Lightning-Kokkos with MPI support:

.. code-block:: console

    # Install Lightning-Kokkos with MPI support

    # Extra MPI flags for optimized inter-GPU communication
    export MPI_EXTRA_LINKER_FLAGS="${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"

    # CMAKE variables for building Lightning-Kokkos with MPI
    export CMAKE_ARGS="-DENABLE_MPI=ON -DCMAKE_CXX_COMPILER=hipcc"

    # Extra variables to avoid hipcc linking issues
    export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CXX_FLAGS='--gcc-install-dir=/opt/cray/pe/gcc/11.2.0/snos/lib/gcc/x86_64-suse-linux/11.2.0/'"
    export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS:FILEPATH=/opt/rocm-6.2.4/lib/llvm/bin/clang-scan-deps" 

    PL_BACKEND="lightning_kokkos" python scripts/configure_pyproject_toml.py
    python -m pip install .

.. note::

    Different compilers and optimization flags dramatically affect performance. We observed significant performance degradation when compiling with ``amdclang++`` and ``CMAKE_BUILD_TYPE`` set to ``RelWithDebugInfo``. For optimal results, we recommend using either ``hipcc`` or ``amdclang++`` with ``CMAKE_BUILD_TYPE`` set to ``Release``.

To submit a job, for example on 2 nodes, the following SLURM script can be used:

.. code-block:: bash

    #!/bin/sh
    #SBATCH -J pennylane
    #SBATCH -t 00:10:00
    #SBATCH -N 2

    module load cray-python
    module load PrgEnv-amd
    module load rocm
    module load cray-pmi
    export MPICH_GPU_SUPPORT_ENABLED=1
    export HSA_ENABLE_PEER_SDMA=0

    srun --ntasks=16 --cpus-per-task=7 --gpus-per-task=1 --gpu-bind=closest python pennylane_quantum_script.py
    
