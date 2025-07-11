Lightning-Qubit installation
****************************

Standard installation
=====================
| **Lightning-Qubit comes pre-installed with PennyLane.**
| Please follow our `installation instructions <https://pennylane.ai/install/#high-performance-computing-and-gpus>`_ to install PennyLane.

Install from source
===================

.. note::

    The section below contains instructions for installing Lightning-Qubit **from source**. For most cases, *this is not required* and one can simply use the installation instructions at `pennylane.ai/install <https://pennylane.ai/install>`__.
    If those instructions do not work for you, or you have a more complex build environment that requires building from source, then consider reading on.

To build Lightning-Qubit from the `sdist` release you can run

.. code-block:: bash

    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    python -m pip install --verbose pennylane-lightning --no-binary "pennylane_lightning"

where ``${PL_BACKEND}`` can be ``lightning_qubit`` (default), ``lightning_gpu``,  ``lightning_kokkos``, or ``lightning_tensor``.
If installing Lightning-GPU, Lightning-Tensor, or Lightning-Kokkos, additional dependencies may be required. We recommend referring to the respective guides for `Lightning-GPU installation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/installation.html>`_, `Lightning-Tensor installation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_tensor/installation.html>`_, and `Lightning-Kokkos installation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/installation.html>`_.

A C++ compiler such as ``g++``, ``clang++``, or ``MSVC`` is required.
On Debian-based systems, this can be installed via ``apt``:

.. code-block:: bash

    sudo apt -y update && sudo apt install -y g++ libomp-dev

where ``libomp-dev`` is included to also install OpenMP.
On MacOS, we recommend using the latest version of ``clang++`` and ``libomp`` via `Homebrew <https://brew.sh>`__:

.. code-block:: bash

    brew install llvm libomp

Development installation
========================

For development and testing, you can install by cloning the repository:

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    
    pip install -r requirements.txt
    pip install git+https://github.com/PennyLaneAI/pennylane.git@master

    PL_BACKEND=${PL_BACKEND} python scripts/configure_pyproject_toml.py
    pip install -e . --config-settings editable_mode=compat -vv

Note that subsequent calls to ``pip install -e .`` will use cached binaries stored in the
``build`` folder, and the ``pyproject.toml`` file defined by the configuration script. Run ``make clean`` if you would like to recompile from scratch.

You can also pass ``cmake`` options with ``CMAKE_ARGS`` as follows:

.. code-block:: bash

    CMAKE_ARGS="-DENABLE_OPENMP=OFF -DENABLE_BLAS=OFF" pip install -e . --config-settings editable_mode=compat -vv


Supported options are ``-DENABLE_WARNINGS``, ``-DENABLE_NATIVE`` (for ``-march=native``) ``-DENABLE_BLAS``, ``-DENABLE_OPENMP``,  and ``-DENABLE_CLANG_TIDY``.

Compile MSVC (Windows)
======================

Lightning-Qubit can be compiled on Windows using the
`Microsoft Visual C++ <https://visualstudio.microsoft.com/vs/features/cplusplus/>`_ compiler.
You need `cmake <https://cmake.org/download/>`_ and appropriate Python environment
(e.g. using `Anaconda <https://www.anaconda.com/>`_).

We recommend using ``[x64 (or x86)] Native Tools Command Prompt for VS [version]`` to compile the library.
Be sure that ``cmake`` and ``python`` can be called within the prompt.

.. code-block:: bash

    cmake --version
    python --version

Then a common command will work.

.. code-block:: bash

    pip install -r requirements.txt
    pip install -e .

Note that OpenMP and BLAS are disabled on this platform.


Testing
=======

To test that a plugin is working correctly, one can check both Python and C++ unit tests for each device.

Python Test
^^^^^^^^^^^

Test the Python code with:

.. code-block:: bash

    pip install -r requirements-tests.txt
    make test-python device=${PL.DEVICE}

where ``${PL.DEVICE}`` differs from ``${PL_BACKEND}`` by replacing the underscore with a period. Options for ``${PL.DEVICE}`` are

- ``lightning.qubit`` (default)
- ``lightning.gpu``
- ``lightning.kokkos``
- ``lightning.tensor``

C++ Test
^^^^^^^^

The C++ code can be tested with

.. code-block:: bash

    PL_BACKEND=${PL_BACKEND} make test-cpp
