Lightning-GPU installation
**************************

Standard installation
=====================
For the majority of cases,
Lightning-GPU can be installed by following our installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__.

Install Lightning-GPU from source
=================================

.. note::

    The section below contains instructions for installing Lightning-GPU **from source**. For most cases, *this is not required* and one can simply use the installation instructions at `pennylane.ai/install <https://pennylane.ai/install/#high-performance-computing-and-gpus>`__. If those instructions do not work for you, or you have a more complex build environment that requires building from source, then consider reading on.

Since you will be installing PennyLane-Lightning from the master branch, it is recommended to install PennyLane from master:

.. code-block:: bash

    pip install git+https://github.com/PennyLaneAI/pennylane.git@master

To install Lightning-GPU from the package sources using the direct SDK path first install Lightning-Qubit (compilation is not necessary):

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    pip install -r requirements.txt
    pip install custatevec-cu12
    PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    SKIP_COMPILATION=True pip install -e . --config-settings editable_mode=compat -vv

Note that `custatevec-cu12` is a requirement for Lightning-GPU, and is installed by ``pip`` separately. After `custatevec-cu12` is installed, the ``CUQUANTUM_SDK`` environment variable should be set to enable discovery during installation:

.. code-block:: bash

    export CUQUANTUM_SDK=$(python -c "import site; print( f'{site.getsitepackages()[0]}/cuquantum')")

Lightning-GPU can then be installed with ``pip``:

.. code-block:: bash

    PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py
    python -m pip install -e . --config-settings editable_mode=compat -vv

Lightning-GPU also requires additional NVIDIA libraries including ``nvJitLink``, ``cuSPARSE``, ``cuBLAS``, and ``CUDA runtime``. These can be installed through the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit/>`_ or from ``pip``.

To simplify the build, we recommend using the containerized build process described in Docker support section.

Install Lightning-GPU with MPI
==============================

.. note::

    Building Lightning-GPU with MPI also requires the ``NVIDIA cuQuantum SDK`` (currently supported version: `custatevec-cu12 <https://pypi.org/project/cuquantum-cu12/>`_), ``mpi4py`` and ``CUDA-aware MPI`` (Message Passing Interface).
    ``CUDA-aware MPI`` allows data exchange between GPU memory spaces of different nodes without the need for CPU-mediated transfers.
    Both the ``MPICH`` and ``OpenMPI`` libraries are supported, provided they are compiled with CUDA support.
    It is recommended to install the ``NVIDIA cuQuantum SDK`` and ``mpi4py`` Python package within ``pip`` or ``conda`` inside a virtual environment.
    Please consult the `cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_ , `mpi4py <https://mpi4py.readthedocs.io/en/stable/install.html>`_,
    `MPICH <https://www.mpich.org/static/downloads/4.1.1/mpich-4.1.1-README.txt>`_, or `OpenMPI <https://www.open-mpi.org/faq/?category=buildcuda>`_ install guide for more information.

**Before installing Lightning-GPU with MPI support using the direct SDK path, please ensure that:**

.. note::

    - Lightning-Qubit, ``CUDA-aware MPI`` and ``custatevec-cu12`` are installed.
    - The environment variable ``CUQUANTUM_SDK`` is set properly.
    - ``path/to/libmpi.so`` is added to ``LD_LIBRARY_PATH``.

Then Lightning-GPU with MPI support can be installed in the *editable* mode:

.. code-block:: bash

    PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DENABLE_MPI=ON" python -m pip install -e . --config-settings editable_mode=compat -vv


Test Lightning-GPU with MPI
===========================

You can test the Python layer of the MPI enabled plugin as follows:

.. code-block:: bash

    mpirun -np 2 python -m pytest mpitests --tb=short

The C++ code can be tested with:

.. code-block:: bash

    PL_BACKEND="lightning_gpu" make test-cpp-mpi
