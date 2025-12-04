Lightning-AMDGPU installation
*****************************

Install Lightning-AMDGPU from source
====================================

Lightning-AMDGPU is an instantiation of the Lighting-Kokkos device, specifically for AMD GPUs using the HIP backend. For building Lightning-Kokkos for targets other than AMD GPUs, please refer to the :doc:`/lightning_kokkos/installation` page.

Install Kokkos (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    Lightning-Kokkos is tested with Kokkos version <= 4.5.00

We recommend first installing Kokkos with your desired configuration by following the instructions in the Kokkos documentation at <https://kokkos.github.io/kokkos-core-wiki/building.html>.
For example, the following will build Kokkos for AMD MI300 GPU:

Download the `Kokkos code <https://github.com/kokkos/kokkos/releases>`_.

.. code-block:: bash

    # Replace x, y, and z by the correct version
    wget https://github.com/kokkos/kokkos/archive/refs/tags/4.x.yz.tar.gz
    tar -xvf 4.x.y.z.tar.gz
    cd kokkos-4.x.y.z

Build Kokkos for AMD MI300 GPU (``GFX942`` architecture), and append the install location to ``CMAKE_PREFIX_PATH``.

.. code-block:: bash

    # Replace <install-path> with the path to install Kokkos
    # e.g. $HOME/kokkos-install/4.5.0/GFX942
    export KOKKOS_INSTALL_PATH=<install-path>
    mkdir -p ${KOKKOS_INSTALL_PATH}

    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_PATH} \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DCMAKE_PREFIX_PATH="/opt/rocm" \
        -DBUILD_SHARED_LIBS:BOOL=ON \
        -DBUILD_TESTING:BOOL=OFF \
        -DKokkos_ENABLE_SERIAL:BOOL=ON \
        -DKokkos_ENABLE_HIP:BOOL=ON \
        -DKokkos_ARCH_AMD_GFX942:BOOL=ON \
        -DKokkos_ENABLE_COMPLEX_ALIGN:BOOL=OFF \
        -DKokkos_ENABLE_EXAMPLES:BOOL=OFF \
        -DKokkos_ENABLE_TESTS:BOOL=OFF \
        -DKokkos_ENABLE_LIBDL:BOOL=OFF
    cmake --build build && cmake --install build
    export CMAKE_PREFIX_PATH=:"${KOKKOS_INSTALL_PATH}":/opt/rocm:$CMAKE_PREFIX_PATH


.. note::

    - Requires AMD compiler ``hipcc`` or ``amdclang`` from the ROCm software stack.
    - ``-DCMAKE_PREFIX_PATH="/opt/rocm"`` enables CMake to properly discover the ``rocthrust`` library
    - For information on choosing the correct architecture flag for your AMD GPU, please refer to the `Kokkos wiki <https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html>`_.


Install Lightning-AMDGPU
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    pip install -r requirements.txt
    pip install git+https://github.com/PennyLaneAI/pennylane.git@master
    
    # First Install Lightning-Qubit
    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    python -m pip install . -vv

    # Install Lightning-AMDGPU
    PL_BACKEND="lightning_amdgpu" python scripts/configure_pyproject_toml.py
    export CMAKE_ARGS="-DCMAKE_CXX_COMPILER=hipcc \
                       -DCMAKE_CXX_FLAGS='--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11/'"
    python -m pip install . -vv

.. note::

    Make sure that ``gcc-11`` is installed and accessible on your system, since it is required to compile the Lightning-AMDGPU device. This can be done on Ubuntu via ``sudo apt install gcc-11 g++-11``.


.. _install-lightning-AMDGPU-with-mpi:

Install Lightning-AMDGPU with MPI
=================================

.. note::

    To build Lightning-AMDGPU with MPI support, please consult the Lightning-Kokkos installation guide at :doc:`/lightning_kokkos/installation` and :doc:`/lightning_kokkos/installation_hpc`.
