Lightning-Tensornet installation
********************************

Install Lightning-Tensornet from source
=======================================

.. note::

    Lightning-Tensornet is currently installed **from source** only.

Lightning-Tensornet is built with `Kokkos <https://kokkos.github.io/kokkos-core-wiki/index.html>`_, in the same way as Lightning-Kokkos,
and can therefore target any of the Kokkos backends.
The SVDs of the MPS are computed with the LAPACK routines of `scipy-openblas32 <https://pypi.org/project/scipy-openblas32/>`_, which is installed as a dependency.

Install Kokkos (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^

If an installation of Kokkos is not found, our builder will automatically clone and install it during the build process.
To use a pre-installed Kokkos (for example, to target a GPU), follow the instructions of the
:ref:`Lightning-Kokkos installation guide <install-kokkos-optional>` and append the Kokkos install location to ``CMAKE_PREFIX_PATH``.

Install Lightning-Tensornet
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lightning-Qubit needs to be 'installed' by ``pip`` before Lightning-Tensornet (compilation is not necessary).

The simplest way to install Lightning-Tensornet (OpenMP backend) is through ``pip``:

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    python -m pip install --group base
    pip install git+https://github.com/PennyLaneAI/pennylane.git@main

    # Lightning-Qubit needs to be 'installed' by pip before Lightning-Tensornet
    # (compilation is not necessary)
    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    SKIP_COMPILATION=True pip install -e . --config-settings editable_mode=compat

    # Install Lightning-Tensornet with OpenMP backend
    PL_BACKEND="lightning_tensornet" python scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DKokkos_ENABLE_OPENMP=ON" python -m pip install -e . --config-settings editable_mode=compat -vv

As for Lightning-Kokkos, the Kokkos backend is selected with the ``-DKokkos_ENABLE_XYZ=ON`` CMake option, where ``XYZ`` is one of
``SERIAL``, ``OPENMP``, ``THREADS``, ``HIP``, ``CUDA`` or ``SYCL``, and the target architecture of a GPU backend with ``-DKokkos_ARCH_{...}=ON``.
For example, for NVIDIA A100 GPUs:

.. code-block:: bash

    PL_BACKEND="lightning_tensornet" python scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON" python -m pip install -e . --config-settings editable_mode=compat -vv

The Catalyst runtime device of Lightning-Tensornet (``liblightning_tensornet_catalyst``), which is used by :func:`~pennylane.qjit`,
is built together with the Python module. To compile it against a local `Catalyst <https://github.com/PennyLaneAI/catalyst>`_
source tree instead of the version fetched by the builder, add ``-DLIGHTNING_CATALYST_SRC_PATH=<path-to-catalyst>`` to ``CMAKE_ARGS``.

.. note::

    Use ``pip`` for editable installs: the build looks for ``scipy-openblas32`` in the current Python environment.
    When building several backends from the same checkout, remove the ``build`` directory between the builds.

Test Lightning-Tensornet
========================

The Python tests of Lightning-Tensornet are in the ``tests/lightning_tensornet`` directory, and are run with:

.. code-block:: bash

    python -m pip install --group tests
    PL_DEVICE="lightning.tensornet" python -m pytest tests/

The PennyLane device test suite can be run with:

.. code-block:: bash

    pl-device-test --device lightning.tensornet --skip-ops --shots=10000
    pl-device-test --device lightning.tensornet --skip-ops

To compile and test the C++ code (MPS backend, shared noise-model helpers and Catalyst runtime device), use:

.. code-block:: bash

    PL_BACKEND="lightning_tensornet" make test-cpp

By default this compiles Kokkos with the ``SERIAL`` backend. If using a different pre-compiled Kokkos backend,
export the environment variable ``CMAKE_PREFIX_PATH`` with the Kokkos install location.

Please refer to the `plugin documentation <https://docs.pennylane.ai/projects/lightning/>`_ as
well as to the `PennyLane documentation <https://docs.pennylane.ai/>`_ for further reference.
