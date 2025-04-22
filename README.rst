Lightning Plugins
#################

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_lqcpu_python.yml?branch=master&label=LQubit&style=flat-square
    :alt: Linux x86_64 L-Qubit Python tests (branch)
    :target: https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_lqcpu_python.yml

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_gpu_python.yml?branch=master&label=LGPU&style=flat-square
    :alt: Linux x86_64 L-GPU Python tests (branch)
    :target: https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_gpu_python.yml

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_lkcpu_python.yml?branch=master&label=LKokkos&style=flat-square
    :alt: Linux x86_64 L-Kokkos Python tests (branch)
    :target: https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_lkcpu_python.yml

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_gpu_python.yml?branch=master&label=LTensor&style=flat-square
    :alt: Linux x86_64 L-Tensor Python tests (branch)
    :target: https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_gpu_python.yml

.. image:: https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-lightning/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane-lightning

.. image:: https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane-lightning/master?logo=codefactor&style=flat-square
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/pennylaneai/pennylane-lightning

.. image:: https://readthedocs.com/projects/xanaduai-pennylane-lightning/badge/?version=latest&style=flat-square
    :alt: Read the Docs
    :target: https://docs.pennylane.ai/projects/lightning

.. image:: https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?logo=discourse&style=flat-square
    :alt: PennyLane Forum
    :target: https://discuss.pennylane.ai

.. image:: https://img.shields.io/pypi/v/PennyLane-Lightning.svg?style=flat-square
    :alt: PyPI - Version
    :target: https://pypi.org/project/PennyLane-Lightning

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-Lightning.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-Lightning

.. image:: https://img.shields.io/pypi/l/PennyLane.svg?logo=apache&style=flat-square
    :alt: License
    :target: https://www.apache.org/licenses/LICENSE-2.0


.. header-start-inclusion-marker-do-not-remove

The Lightning plugin ecosystem provides fast state-vector and tensor-network simulators written in C++.

`PennyLane <https://docs.pennylane.ai>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.
PennyLane supports Python 3.10 and above.

Backends
********

PennyLane-Lightning high performance simulators include the following backends:

* ``lightning.qubit``: a fast state-vector simulator written in C++ with optional `OpenMP <https://www.openmp.org/>`_ additions and parallelized gate-level SIMD kernels.
* ``lightning.gpu``: a state-vector simulator based on the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_. It notably implements a distributed state-vector simulator based on `MPI <https://www.mpi-forum.org/docs/>`_.
* ``lightning.kokkos``: a state-vector simulator written with `Kokkos <https://kokkos.github.io/kokkos-core-wiki/index.html>`_. It can exploit the inherent parallelism of modern processing units supporting the `OpenMP <https://www.openmp.org/>`_, `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ or `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest>`_ programming models.
* ``lightning.tensor``: a tensor-network simulator based on the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_. The supported methods are Matrix Product State (MPS) and Exact Tensor Network (TN).

If you're not sure what simulator to use, check out our `PennyLane performance <https://pennylane.ai/performance>`_ page.

.. header-end-inclusion-marker-do-not-remove

Installation
************

The following table summarizes the supported platforms and the primary installation mode:

+------------------------+-----------+-----------+-----------+-----------+-----------+
|                        | Linux x86 | Linux ARM | MacOS x86 | MacOS ARM | Windows   |
|                        |           |           |           |           |           |
+========================+===========+===========+===========+===========+===========+
| Lightning-Qubit        | pip       | pip       | pip       | pip       | pip       |
+------------------------+-----------+-----------+-----------+-----------+-----------+
| Lightning-GPU          | pip       | pip       |           |           |           |
+------------------------+-----------+-----------+-----------+-----------+-----------+
| Lightning-GPU (MPI)    | source    |           |           |           |           |
+------------------------+-----------+-----------+-----------+-----------+-----------+
| Lightning-Kokkos (OMP) | pip       | pip       | pip       | pip       |           |
+------------------------+-----------+-----------+-----------+-----------+-----------+
| Lightning-Kokkos (CUDA)| source    | source    |           |           |           |
+------------------------+-----------+-----------+-----------+-----------+-----------+
| Lightning-Kokkos (HIP) | source    | source    |           |           |           |
+------------------------+-----------+-----------+-----------+-----------+-----------+
| Lightning-Tensor       | pip       | pip       |           |           |           |
+------------------------+-----------+-----------+-----------+-----------+-----------+

To install the latest stable version of these plugins, check out the `PennyLane installation guide <https://pennylane.ai/install#high-performance-computing-and-gpus>`_.

If you wish to install the latest development version, instructions for `building from source <https://docs.pennylane.ai/projects/lightning/en/stable/dev/installation.html>`_ are also available for each backend.

.. docker-start-inclusion-marker-do-not-remove

Docker support
**************

Docker images for the various backends are found on the
`PennyLane Docker Hub <https://hub.docker.com/u/pennylaneai>`_ page, where a detailed description about PennyLane Docker support can be found.
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
The Python code is formatted with the PEP 8 compliant opinionated formatter `Black <https://github.com/psf/black>`_ (`black==25.1.0`).
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
- **scipy-openblas32:** https://pypi.org/project/scipy-openblas32/
- **Xanadu JET:** https://github.com/XanaduAI/jet

.. acknowledgements-end-inclusion-marker-do-not-remove
