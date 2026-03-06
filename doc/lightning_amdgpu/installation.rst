Lightning-AMDGPU installation
*****************************

Install Lightning-AMDGPU from source
====================================

Lightning-AMDGPU is an instantiation of the Lighting-Kokkos device, specifically for AMD GPUs using the HIP backend. For building Lightning-Kokkos for targets other than AMD GPUs, please refer to the :doc:`/lightning_kokkos/installation` page.

The installation instruction here is specifically for AMD MI300 GPU (GFX942); for other architecture, please refer to the `Kokkos wiki <https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html>`_ for the correct flag.

.. note::

    Lightning-Kokkos and Lightning-AMDGPU is tested with Kokkos version 5.0.0


Install Lightning-AMDGPU
^^^^^^^^^^^^^^^^^^^^^^^^
.. note::

    - An AMD compiler ``hipcc`` or ``amdclang`` from the ROCm software stack is required.
    - ``-DCMAKE_PREFIX_PATH="/opt/rocm"`` enables CMake to properly discover the ``rocthrust`` library

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    python -m pip install --group base
    pip install git+https://github.com/PennyLaneAI/pennylane.git@master

    # First Install Lightning-Qubit
    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    python -m pip install . -vv

    # Install Lightning-AMDGPU
    PL_BACKEND="lightning_amdgpu" python scripts/configure_pyproject_toml.py
    export CMAKE_ARGS="-DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX942=ON"
    python -m pip install . -vv



.. _install-lightning-AMDGPU-with-mpi:

Install Lightning-AMDGPU with MPI
=================================

.. note::

    To build Lightning-AMDGPU with MPI support, please consult the Lightning-Kokkos installation guide at :doc:`/lightning_kokkos/installation` and :doc:`/lightning_kokkos/installation_hpc`.
