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

The Lightning plugin ecosystem provides fast state-vector simulators written in C++.

`PennyLane <https://docs.pennylane.ai>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.
PennyLane supports Python 3.9 and above.

.. header-end-inclusion-marker-do-not-remove


Features
========

* Combine Lightning's high performance simulators with PennyLane's
  automatic differentiation and optimization.

.. installation_LQubit-start-inclusion-marker-do-not-remove


Lightning Qubit installation
============================

Lightning Qubit can be installed using ``pip``:

.. code-block:: console

    $ pip install pennylane-lightning

To build Lightning from source you can run

.. code-block:: console

    $ pip install pybind11 pennylane-lightning --no-binary :all:

A C++ compiler such as ``g++``, ``clang++``, or ``MSVC`` is required.
On Debian-based systems, this can be installed via ``apt``:

.. code-block:: console

    $ sudo apt install g++

On MacOS, we recommend using the latest version of ``clang++`` and ``libomp``:

.. code-block:: console

    $ brew install llvm libomp

The `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ library is also used for binding the
C++ functionality to Python.

Alternatively, for development and testing, you can install by cloning the repository:

.. code-block:: console

    $ git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    $ cd pennylane-lightning
    $ pip install -r requirements.txt
    $ pip install -e .

Note that subsequent calls to ``pip install -e .`` will use cached binaries stored in the
``build`` folder. Run ``make clean`` if you would like to recompile.

You can also pass ``cmake`` options with ``CMAKE_ARGS`` as follows:

.. code-block:: console

    $ CMAKE_ARGS="-DENABLE_OPENMP=OFF -DENABLE_BLAS=OFF -DENABLE_KOKKOS=OFF" pip install -e . -vv

or with ``build_ext`` and the ``--define`` flag as follows:

.. code-block:: console

    $ python3 setup.py build_ext -i --define="ENABLE_OPENMP=OFF;ENABLE_BLAS=OFF;ENABLE_KOKKOS=OFF"
    $ python3 setup.py develop


Testing
-------

To test that the plugin is working correctly you can test the Python code within the cloned
repository:

.. code-block:: console

    $ make test-python

while the C++ code can be tested with

.. code-block:: console

    $ make test-cpp


CMake Support
-------------

One can also build the plugin using CMake:

.. code-block:: console

    $ cmake -S. -B build
    $ cmake --build build

To test the C++ code:

.. code-block:: console

    $ mkdir build && cd build
    $ cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
    $ make

Other supported options are

- ``-DENABLE_WARNINGS=ON``
- ``-DENABLE_NATIVE=ON`` (for ``-march=native``)
- ``-DENABLE_BLAS=ON``
- ``-DENABLE_OPENMP=ON``
- ``-DENABLE_KOKKOS=ON``
- ``-DENABLE_CLANG_TIDY=ON``

Compile on Windows with MSVC
----------------------------

You can also compile Lightning on Windows using
`Microsoft Visual C++ <https://visualstudio.microsoft.com/vs/features/cplusplus/>`_ compiler.
You need `cmake <https://cmake.org/download/>`_ and appropriate Python environment
(e.g. using `Anaconda <https://www.anaconda.com/>`_).


We recommend to use ``[x64 (or x86)] Native Tools Command Prompt for VS [version]`` for compiling the library.
Be sure that ``cmake`` and ``python`` can be called within the prompt.


.. code-block:: console

    $ cmake --version
    $ python --version

Then a common command will work.

.. code-block:: console

    $ pip install -r requirements.txt
    $ pip install -e .

Note that OpenMP and BLAS are disabled in this setting.


.. installation_LQubit-end-inclusion-marker-do-not-remove


.. installation_LKokkos-start-inclusion-marker-do-not-remove

Lightning Kokkos installation
=============================

We suggest first installing Kokkos with the wanted configuration following the instructions found in the `Kokkos documentation <https://kokkos.github.io/kokkos-core-wiki/building.html>`_.
Next, append the install location to ``CMAKE_PREFIX_PATH``.
If an installation is not found, our builder will install it from scratch nevertheless.

The simplest way to install PennyLane-Lightning-Kokkos (OpenMP backend) is using ``pip``.

.. code-block:: console

   CMAKE_ARGS="-DKokkos_ENABLE_OPENMP=ON" PL_BACKEND="lightning_kokkos" python -m pip install .

or for an editable ``pip`` installation with:

.. code-block:: console

   CMAKE_ARGS="-DKokkos_ENABLE_OPENMP=ON" PL_BACKEND="lightning_kokkos" python -m pip install -e .

Alternatively, you can install the Python interface with:

.. code-block:: console

   CMAKE_ARGS="-DKokkos_ENABLE_OPENMP=ON" PL_BACKEND="lightning_kokkos" python setup.py build_ext
   python setup.py bdist_wheel
   pip install ./dist/PennyLane*.whl --force-reinstall

To build the plugin directly with CMake:

.. code-block:: console

   cmake -B build -DKokkos_ENABLE_OPENMP=ON -DPLKOKKOS_BUILD_TESTS=ON -DPL_BACKEND=lightning_kokkos -G Ninja
   cmake --build build

Supported backend options are "SERIAL", "OPENMP", "THREADS", "HIP" and "CUDA" and the corresponding build options are ``-DKokkos_ENABLE_XXX=ON``, where ``XXX`` needs be replaced by the backend name, for instance ``OPENMP``.
One can activate simultaneously one serial, one parallel CPU host (e.g. "OPENMP", "THREADS") and one parallel GPU device backend (e.g. "HIP", "CUDA"), but not two of any category at the same time.
For "HIP" and "CUDA", the appropriate software stacks are required to enable compilation and subsequent use.
Similarly, the CMake option ``-DKokkos_ARCH_{...}=ON`` must also be specified to target a given architecture.
A list of the architectures is found on the `Kokkos wiki <https://github.com/kokkos/kokkos/wiki/Macros#architectures>`_.
Note that "THREADS" backend is not recommended since `Kokkos <https://github.com/kokkos/kokkos-core-wiki/blob/17f08a6483937c26e14ec3c93a2aa40e4ce081ce/docs/source/ProgrammingGuide/Initialization.md?plain=1#L67>`_ does not guarantee its safety.


Testing
=======

To test with the ROCm stack using a manylinux2014 container we must first mount the repository into the container:

.. code-block:: console

    docker run -v `pwd`:/io -it quay.io/pypa/manylinux2014_x86_64 bash

Next, within the container, we install the ROCm software stack:

.. code-block:: console

    yum install -y https://repo.radeon.com/amdgpu-install/21.40.2/rhel/7.9/amdgpu-install-21.40.2.40502-1.el7.noarch.rpm
    amdgpu-install --usecase=hiplibsdk,rocm --no-dkms

We next build the test suite, with a given AMD GPU target in mind, as listed `here <https://github.com/kokkos/kokkos/blob/master/Makefile.kokkos>`_.

.. code-block:: console

    cd /io
    export PATH=$PATH:/opt/rocm/bin/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
    export CXX=/opt/rocm/hip/bin/hipcc
    cmake -B build -DCMAKE_CXX_COMPILER=/opt/rocm/hip/bin/hipcc -DKokkos_ENABLE_HIP=ON -DPLKOKKOS_BUILD_TESTS=ON -DKokkos_ARCH_VEGA90A=ON
    cmake --build build --verbose

We may now leave the container, and run the built test suite on a machine with access to the targeted GPU.

For a system with access to the ROCm stack outside of a manylinux container, an editable ``pip`` installation can be built and installed as:

.. code-block:: console

   CMAKE_ARGS="-DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_VEGA90A=ON" PL_BACKEND="lightning_kokkos" python -m pip install -e .

.. installation_LKokkos-end-inclusion-marker-do-not-remove

Please refer to the `plugin documentation <https://docs.pennylane.ai/projects/lightning/>`_ as
well as to the `PennyLane documentation <https://docs.pennylane.ai/>`_ for further reference.


GPU support
-----------

For GPU support, `PennyLane-Lightning-GPU <https://github.com/PennyLaneAI/pennylane-lightning-gpu>`_
can be installed by providing the optional ``[gpu]`` tag:

.. code-block:: console

    $ pip install pennylane-lightning[gpu]

For more information, please refer to the PennyLane Lightning GPU `documentation <https://docs.pennylane.ai/projects/lightning-gpu>`_.

Docker Support
--------------

One can also build the Lightning image using Docker:

.. code-block:: console

    $ git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    $ cd pennylane-lightning
    $ docker build -t lightning/base -f docker/Dockerfile .

Please refer to the `PennyLane installation <https://docs.pennylane.ai/en/stable/development/guide/installation.html#installation>`_ for detailed description about PennyLane Docker support.


Contributing
============

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
=======

Lightning is the work of `many contributors <https://github.com/PennyLaneAI/pennylane-lightning/graphs/contributors>`_.

If you are doing research using PennyLane and Lightning, please cite `our paper <https://arxiv.org/abs/1811.04968>`_:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968

.. support-start-inclusion-marker-do-not-remove


Support
=======

- **Source Code:** https://github.com/PennyLaneAI/pennylane-lightning
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-lightning/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove


License
=======

The PennyLane lightning plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

.. license-end-inclusion-marker-do-not-remove
.. acknowledgements-start-inclusion-marker-do-not-remove

Acknowledgements
================

PennyLane Lightning makes use of the following libraries and tools, which are under their own respective licenses:

- **pybind11:** https://github.com/pybind/pybind11
- **Kokkos Core:** https://github.com/kokkos/kokkos

.. acknowledgements-end-inclusion-marker-do-not-remove