Lightning Plugins
#################

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_linux.yml?branch=master&label=Test%20%28Linux%29&style=flat-square
    :alt: Linux x86_64 tests (branch)
    :target: https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_linux.yml

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_windows.yml?branch=master&label=Test%20%28Windows%29&style=flat-square
    :alt: Windows tests (branch)
    :target: https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_windows.yml

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/.github/workflows/wheel_linux_x86_64.yml?branch=master&logo=github&style=flat-square
    :alt: Linux x86_64 wheel builds (branch)
    :target: https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/wheel_linux_x86_64.yml?query=branch%3Amaster++

.. image:: https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-lightning/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane-lightning

.. image:: https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane-lightning/master?logo=codefactor&style=flat-square
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/pennylaneai/pennylane-lightning

.. image:: https://readthedocs.com/projects/xanaduai-pennylane-lightning/badge/?version=latest&style=flat-square
    :alt: Read the Docs
    :target: https://docs.pennylane.ai/projects/lightning

.. image:: https://img.shields.io/pypi/v/PennyLane-Lightning.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-Lightning

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-Lightning.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-Lightning

.. header-start-inclusion-marker-do-not-remove

The Lightning plugin ecosystem provides fast state-vector and tensor network simulators written in C++.

`PennyLane <https://docs.pennylane.ai>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.
PennyLane supports Python 3.10 and above.

Features
********

PennyLane-Lightning high performance simulators include the following backends:

* ``lightning.qubit``: is a fast state-vector simulator written in C++.
* ``lightning.gpu``: is a state-vector simulator based on the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_. It notably implements a distributed state-vector simulator based on MPI.
* ``lightning.kokkos``: is a state-vector simulator written with `Kokkos <https://kokkos.github.io/kokkos-core-wiki/index.html>`_. It can exploit the inherent parallelism of modern processing units supporting the `OpenMP <https://www.openmp.org/>`_, `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ or `HIP <https://docs.amd.com/projects/HIP/en/docs-5.3.0/index.html>`_ programming models.
* ``lightning.tensor``: is a tensor network simulator based on the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ (requires NVIDIA GPUs with SM 7.0 or greater). The supported method is Matrix Product State (MPS).

.. header-end-inclusion-marker-do-not-remove

The following table summarizes the supported platforms and the primary installation mode:

+-----------+---------+--------+-------------+----------------+-----------------+----------------+----------------+
|           | L-Qubit | L-GPU  | L-GPU (MPI) | L-Kokkos (OMP) | L-Kokkos (CUDA) | L-Kokkos (HIP) |    L-Tensor    |
+===========+=========+========+=============+================+=================+================+================+
| Linux x86 | pip     | pip    | source      | pip            | source          | source         |     pip        |
+-----------+---------+--------+-------------+----------------+-----------------+----------------+----------------+
| Linux ARM | pip     | pip    |             | pip            | source          | source         |     pip        |
+-----------+---------+--------+-------------+----------------+-----------------+----------------+----------------+
| Linux PPC | pip     | source |             | source         | source          | source         |                |
+-----------+---------+--------+-------------+----------------+-----------------+----------------+----------------+
| MacOS x86 | pip     |        |             | pip            |                 |                |                |
+-----------+---------+--------+-------------+----------------+-----------------+----------------+----------------+
| MacOS ARM | pip     |        |             | pip            |                 |                |                |
+-----------+---------+--------+-------------+----------------+-----------------+----------------+----------------+
| Windows   | pip     |        |             |                |                 |                |                |
+-----------+---------+--------+-------------+----------------+-----------------+----------------+----------------+


.. installation_LQubit-start-inclusion-marker-do-not-remove

Lightning-Qubit installation
****************************

Lightning-Qubit comes pre-installed with PennyLane. Please follow our 
`installation instructions <https://pennylane.ai/install/#high-performance-computing-and-gpus>`_ 
to install PennyLane.

Install from source
===================

.. note::

    The below contains instructions for installing Lightning-Qubit ***from source***. For most cases, *this is not required* and one can simply use the installation instructions at `pennylane.ai/install <https://pennylane.ai/install>`__.
    If those instructions do not work for you, or you have a more complex build environment that requires building from source, then consider reading on.

To build Lightning plugins from source you can run

.. code-block:: bash

    PL_BACKEND=${PL_BACKEND} pip install pybind11 pennylane-lightning --no-binary :all:

where ``${PL_BACKEND}`` can be ``lightning_qubit`` (default), ``lightning_gpu``,  ``lightning_kokkos``, or ``lightning_tensor``.
The `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ library is required to bind the C++ functionality to Python.

A C++ compiler such as ``g++``, ``clang++``, or ``MSVC`` is required.
On Debian-based systems, this can be installed via ``apt``:

.. code-block:: bash

    sudo apt -y update && sudo apt install -y g++ libomp-dev

where ``libomp-dev`` is included to also install OpenMP.
On MacOS, we recommend using the latest version of ``clang++`` and ``libomp``:

.. code-block:: bash

    brew install llvm libomp

The Lightning-GPU backend has several dependencies (e.g. ``CUDA``, ``custatevec-cu12``, etc.), and hence we recommend referring to `Lightning-GPU installation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/installation.html>`_ section.
Similarly, for Lightning-Kokkos it is recommended to configure and install Kokkos independently as prescribed in the `Lightning-Kokkos installation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/installation.html>`_ section.

Development installation
========================

For development and testing, you can install by cloning the repository:

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    pip install -r requirements.txt
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

    make test-python device=${PL.DEVICE}

where ``${PL.DEVICE}`` differ from ``${PL_BACKEND}`` by replacing the underscore by a dot. And can be 

- ``lightning.qubit`` (default) 
- ``lightning.gpu``  
- ``lightning.kokkos``
- ``lightning.tensor``

C++ Test
^^^^^^^^
 
The C++ code can be tested with

.. code-block:: bash

    PL_BACKEND=${PL_BACKEND} make test-cpp

.. installation_LQubit-end-inclusion-marker-do-not-remove

.. installation_LGPU-start-inclusion-marker-do-not-remove


Lightning-GPU installation
**************************

For the majority of cases, Lightning-GPU can be installed by following our installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__.

Install Lightning-GPU from source
=================================

.. note::

    The below contains instructions for installing Lightning-GPU ***from source***. For most cases, *this is not required* and one can simply use the installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__. If those instructions do not work for you, or you have a more complex build environment that requires building from source, then consider reading on.

To install Lightning-GPU from the package sources using the direct SDK path, Lightning-Qubit should be install before Lightning-GPU (compilation is not necessary):

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    pip install -r requirements.txt
    pip install custatevec-cu12
    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    SKIP_COMPILATION=True pip install -e . --config-settings editable_mode=compat -vv

Then a ``CUQUANTUM_SDK`` environment variable can be set:

.. code-block:: bash

    export CUQUANTUM_SDK=$(python -c "import site; print( f'{site.getsitepackages()[0]}/cuquantum')")

The Lightning-GPU can then be installed with ``pip``:

.. code-block:: bash

    PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py
    python -m pip install -e . --config-settings editable_mode=compat -vv

To simplify the build, we recommend using the containerized build process described in Docker support section.

Install Lightning-GPU with MPI
==============================

Building Lightning-GPU with MPI also requires the ``NVIDIA cuQuantum SDK`` (currently supported version: `custatevec-cu12 <https://pypi.org/project/cuquantum-cu12/>`_), ``mpi4py`` and ``CUDA-aware MPI`` (Message Passing Interface).
``CUDA-aware MPI`` allows data exchange between GPU memory spaces of different nodes without the need for CPU-mediated transfers.
Both the ``MPICH`` and ``OpenMPI`` libraries are supported, provided they are compiled with CUDA support.
The path to ``libmpi.so`` should be found in ``LD_LIBRARY_PATH``.
It is recommended to install the ``NVIDIA cuQuantum SDK`` and ``mpi4py`` Python package within ``pip`` or ``conda`` inside a virtual environment.
Please consult the `cuQuantum SDK`_ , `mpi4py <https://mpi4py.readthedocs.io/en/stable/install.html>`_,
`MPICH <https://www.mpich.org/static/downloads/4.1.1/mpich-4.1.1-README.txt>`_, or `OpenMPI <https://www.open-mpi.org/faq/?category=buildcuda>`_ install guide for more information.

Before installing Lightning-GPU with MPI support using the direct SDK path, please ensure Lightning-Qubit, ``CUDA-aware MPI`` and ``custatevec`` are installed and the environment variable ``CUQUANTUM_SDK`` is set properly.
Then Lightning-GPU with MPI support can then be installed in the *editable* mode:

.. code-block:: bash

    PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DENABLE_MPI=ON" python -m pip install -e . --config-settings editable_mode=compat -vv


Test Lightning-GPU with MPI
===========================

You may test the Python layer of the MPI enabled plugin as follows:

.. code-block:: bash

    mpirun -np 2 python -m pytest mpitests --tb=short

The C++ code is tested with

.. code-block:: bash

    rm -rf ./BuildTests
    cmake . -BBuildTests -DBUILD_TESTS=1 -DBUILD_TESTS=1 -DENABLE_MPI=ON -DCUQUANTUM_SDK=<path to sdk>
    cmake --build ./BuildTests --verbose
    cd ./BuildTests
    for file in *runner_mpi ; do mpirun -np 2 ./BuildTests/$file ; done;

.. installation_LGPU-end-inclusion-marker-do-not-remove

.. installation_LKokkos-start-inclusion-marker-do-not-remove

Lightning-Kokkos installation
*****************************

On most Linux systems, Lightning-Kokkos can be installed via Spack or Docker by following our installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__.

Install Lightning-Kokkos from source
====================================

.. note::

    The below contains instructions for installing Lightning-Kokkos ***from source***. For most cases, one can install Lightning-Kokkos via Spack or Docker by the installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__. If those instructions do not work for you, or you have a more complex build environment that requires building from source, then consider reading on.

As Kokkos enables support for many different HPC-targeted hardware platforms, ``lightning.kokkos`` can be built to support any of these platforms when building from source.

Install Kokkos (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^

We suggest first installing Kokkos with the wanted configuration following the instructions found in the `Kokkos documentation <https://kokkos.github.io/kokkos-core-wiki/building.html>`_.
For example, the following will build Kokkos for NVIDIA A100 cards

Download the `Kokkos code <https://github.com/kokkos/kokkos/releases>`_. Lightning Kokkos was tested with Kokkos version <= 4.3.01  

.. code-block:: bash

    # Replace x, y, and z by the correct version
    wget https://github.com/kokkos/kokkos/archive/refs/tags/4.x.yz.tar.gz 
    tar -xvf 4.x.y.z.tar.gz
    cd kokkos-4.x.y.z

Build Kokkos for NVIDIA A100 cards (``SM80`` architecture)

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


Next, append the install location to ``CMAKE_PREFIX_PATH``.
Note that the C++20 standard is required (``-DCMAKE_CXX_STANDARD=20`` option), and hence CUDA v12 is required for the CUDA backend.

Install Lightning-Kokkos
^^^^^^^^^^^^^^^^^^^^^^^^

If an installation of Kokkos is not found, then our builder will clone and install it during the build process. Lightning-Qubit should be installed (compilation is not necessary):

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

and the corresponding build options are ``-DKokkos_ENABLE_XXX=ON``, where ``XXX`` needs be replaced by the backend name, for instance ``OPENMP``.

One can activate simultaneously one serial, one parallel CPU host (e.g. ``OPENMP``, ``THREADS``) and one parallel GPU device backend (e.g. ``HIP``, ``CUDA``), but not two of any category at the same time.
For ``HIP`` and ``CUDA``, the appropriate software stacks are required to enable compilation and subsequent use.
Similarly, the CMake option ``-DKokkos_ARCH_{...}=ON`` must also be specified to target a given architecture.
A list of the architectures is found on the `Kokkos wiki <https://kokkos.org/kokkos-core-wiki/API/core/Macros.html#architectures>`_.
Note that ``THREADS`` backend is not recommended since `Kokkos does not guarantee its safety <https://github.com/kokkos/kokkos-core-wiki/blob/17f08a6483937c26e14ec3c93a2aa40e4ce081ce/docs/source/ProgrammingGuide/Initialization.md?plain=1#L67>`_.

.. installation_LKokkos-end-inclusion-marker-do-not-remove

.. installation_LTensor-start-inclusion-marker-do-not-remove


Lightning-Tensor installation
*****************************
Lightning-Tensor requires CUDA 12 and the `cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ (only the `cutensornet <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html>`_ library is required).
The SDK may be installed within the Python environment ``site-packages`` directory using ``pip`` or ``conda`` or the SDK library path appended to the ``LD_LIBRARY_PATH`` environment variable.
Please see the `cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ install guide for more information.

Lightning-Tensor and ``cutensornet-cu12`` can be installed via:

.. code-block:: bash
    
    pip install cutensornet-cu12
    pip install pennylane-lightning-tensor

Install Lightning-Tensor from source
====================================

.. note::

    The below contains instructions for installing Lightning-Tensor ***from source***. For most cases, *this is not required* and one can simply use the installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__. If those instructions do not work for you, or you have a more complex build environment that requires building from source, then consider reading on.

Lightning-Qubit should be installed before Lightning-Tensor (compilation is not necessary):

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    pip install -r requirements.txt
    pip install cutensornet-cu12
    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    SKIP_COMPILATION=True pip install -e . --config-settings editable_mode=compat

Then a ``CUQUANTUM_SDK`` environment variable can be set:

.. code-block:: bash

    export CUQUANTUM_SDK=$(python -c "import site; print( f'{site.getsitepackages()[0]}/cuquantum')")

The Lightning-Tensor can then be installed with ``pip``:

.. code-block:: bash

    PL_BACKEND="lightning_tensor" python scripts/configure_pyproject_toml.py
    pip install -e . --config-settings editable_mode=compat -vv

.. installation_LTensor-end-inclusion-marker-do-not-remove


Please refer to the `plugin documentation <https://docs.pennylane.ai/projects/lightning/>`_ as
well as to the `PennyLane documentation <https://docs.pennylane.ai/>`_ for further reference.

.. docker-start-inclusion-marker-do-not-remove


Docker support
**************

Docker images for the various backends are found on the
`PennyLane Docker Hub <https://hub.docker.com/u/pennylaneai>`_ page, where there is also a detailed description about PennyLane Docker support.
Briefly, one can build the Docker Lightning images using:

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    docker build -f docker/Dockerfile --target ${TARGET} .

where ``${TARGET}`` is one of the following

* ``wheel-lightning-qubit``
* ``wheel-lightning-gpu``
* ``wheel-lightning-kokkos-openmp``
* ``wheel-lightning-kokkos-cuda``
* ``wheel-lightning-kokkos-rocm``

.. docker-end-inclusion-marker-do-not-remove

Contributing
************

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributors to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.

Black & Pylint
==============

If you contribute to the Python code, please mind the following.
The Python code is formatted with the PEP 8 compliant opinionated formatter `Black <https://github.com/psf/black>`_ (`black==23.7.0`).
We set a line width of a 100 characters.
The Python code is statically analyzed with `Pylint <https://pylint.readthedocs.io/en/stable/>`_.
We set up a pre-commit hook (see `Git hooks <https://git-scm.com/docs/githooks>`_) to run both of these on `git commit`.
Please make your best effort to comply with `black` and `pylint` before using disabling pragmas (e.g. `# pylint: disable=missing-function-docstring`).

Authors
*******

.. citation-start-inclusion-marker-do-not-remove

Lightning is the work of `many contributors <https://github.com/PennyLaneAI/pennylane-lightning/graphs/contributors>`_.

If you are using Lightning for research, please cite:

.. code-block:: bibtex

    @misc{
        asadi2024,
        title={{Hybrid quantum programming with PennyLane Lightning on HPC platforms}},
        author={Ali Asadi and Amintor Dusko and Chae-Yeun Park and Vincent Michaud-Rioux and Isidor Schoch and Shuli Shu and Trevor Vincent and Lee James O'Riordan},
        year={2024},
        eprint={2403.02512},
        archivePrefix={arXiv},
        primaryClass={quant-ph},
        url={https://arxiv.org/abs/2403.02512},
    }

.. citation-end-inclusion-marker-do-not-remove
.. support-start-inclusion-marker-do-not-remove

Support
*******

- **Source Code:** https://github.com/PennyLaneAI/pennylane-lightning
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-lightning/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
*******

The Lightning plugins are **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
The Lightning-GPU and Lightning-Tensor plugins make use of the NVIDIA cuQuantum SDK headers to
enable the device bindings to PennyLane, which are held to their own respective license.

.. license-end-inclusion-marker-do-not-remove
.. acknowledgements-start-inclusion-marker-do-not-remove

Acknowledgements
****************

PennyLane Lightning makes use of the following libraries and tools, which are under their own respective licenses:

- **pybind11:** https://github.com/pybind/pybind11
- **Kokkos Core:** https://github.com/kokkos/kokkos
- **NVIDIA cuQuantum:** https://developer.nvidia.com/cuquantum-sdk
- **Xanadu JET:** https://github.com/XanaduAI/jet

.. acknowledgements-end-inclusion-marker-do-not-remove
