Lightning-Kokkos installation
*****************************

Standard installation
=====================
On most Linux systems, Lightning-Kokkos can be installed by following our installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__.

Install Lightning-Kokkos from source
====================================

.. note::

    The section below contains instructions for installing Lightning-Kokkos **from source**. For most cases, one can install Lightning-Kokkos via Spack or Docker by the installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__. If those instructions do not work for you, or you have a more complex build environment that requires building from source, then consider reading on.

As Kokkos enables support for many different HPC-targeted hardware platforms, ``lightning.kokkos`` can be built to support any of these platforms when building from source.

Install Kokkos (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^

We suggest first installing Kokkos with the wanted configuration following the instructions found in the `Kokkos documentation <https://kokkos.github.io/kokkos-core-wiki/building.html>`_.
For example, the following will build Kokkos for NVIDIA A100 cards:

Download the `Kokkos code <https://github.com/kokkos/kokkos/releases>`_. Lightning-Kokkos was tested with Kokkos version <= 4.5.0

.. code-block:: bash

    # Replace x, y, and z by the correct version
    wget https://github.com/kokkos/kokkos/archive/refs/tags/4.x.yz.tar.gz
    tar -xvf 4.x.y.z.tar.gz
    cd kokkos-4.x.y.z

Build Kokkos for NVIDIA A100 cards (``SM80`` architecture), and append the install location to ``CMAKE_PREFIX_PATH``.

.. code-block:: bash

    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebugInfo \
        -DCMAKE_INSTALL_PREFIX=/opt/kokkos/4.x.y.z/AMPERE80 \
        -DCMAKE_CXX_STANDARD=20 \
        -DBUILD_SHARED_LIBS:BOOL=ON \
        -DBUILD_TESTING:BOOL=OFF \
        -DKokkos_ENABLE_SERIAL:BOOL=ON \
        -DKokkos_ENABLE_CUDA:BOOL=ON \
        -DKokkos_ARCH_AMPERE80:BOOL=ON \
        -DKokkos_ENABLE_EXAMPLES:BOOL=OFF \
        -DKokkos_ENABLE_TESTS:BOOL=OFF \
        -DKokkos_ENABLE_LIBDL:BOOL=OFF
    cmake --build build && cmake --install build
    export CMAKE_PREFIX_PATH=/opt/kokkos/4.x.y.z/AMPERE80:$CMAKE_PREFIX_PATH


Note that the C++20 standard is required (enabled via the ``-DCMAKE_CXX_STANDARD=20`` option), hence CUDA 12 is required for the CUDA backend.

Install Lightning-Kokkos
^^^^^^^^^^^^^^^^^^^^^^^^

If an installation of Kokkos is not found, then our builder will automatically clone and install it during the build process. Lightning-Qubit needs to be 'installed' by ``pip`` before Lightning-Kokkos (compilation is not necessary).

The simplest way to install Lightning-Kokkos (OpenMP backend) through ``pip``.

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    SKIP_COMPILATION=True pip install -e . --config-settings editable_mode=compat
    PL_BACKEND="lightning_kokkos" python scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DKokkos_ENABLE_OPENMP=ON" python -m pip install -e . --config-settings editable_mode=compat -vv

The supported backend options are

.. list-table::
    :align: center
    :width: 100 %
    :widths: 20 20 20 20 20
    :header-rows: 0

    * - ``SERIAL``
      - ``OPENMP``
      - ``THREADS``
      - ``HIP``
      - ``CUDA``

and the corresponding build options are ``-DKokkos_ENABLE_XYZ=ON``, where ``XYZ`` needs be replaced by the backend name, for instance ``OPENMP``.

One can simutaneously activate one serial, one parallel CPU host (e.g. ``OPENMP``, ``THREADS``) and one parallel GPU device backend (e.g. ``HIP``, ``CUDA``), but not two of any category at the same time.
For ``HIP`` and ``CUDA``, the appropriate software stacks are required to enable compilation and subsequent use.
Similarly, the CMake option ``-DKokkos_ARCH_{...}=ON`` must also be specified to target a given architecture.
A list of the architectures is found on the `Kokkos wiki <https://kokkos.org/kokkos-core-wiki/API/core/Macros.html#architectures>`_.
Note that ``THREADS`` backend is not recommended since `Kokkos does not guarantee its safety <https://github.com/kokkos/kokkos-core-wiki/blob/17f08a6483937c26e14ec3c93a2aa40e4ce081ce/docs/source/ProgrammingGuide/Initialization.md?plain=1#L67>`_.
