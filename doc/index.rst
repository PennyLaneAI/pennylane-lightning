Lightning plugins
#################

:Release: |release|

.. image:: _static/pennylane_lightning.png
    :align: left
    :width: 210px
    :target: javascript:void(0);

The Lightning plugin ecosystem provides fast state-vector and tensor-network simulators written in C++.

`PennyLane <https://docs.pennylane.ai>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.
PennyLane supports Python 3.11 and above.

Backends
********

PennyLane-Lightning high performance simulators include the following backends:

*   ``lightning.qubit``: a fast state-vector simulator written in C++
    with optional `OpenMP <https://www.openmp.org/>`_ additions and parallelized gate-level SIMD kernels.
*   ``lightning.gpu``: a state-vector simulator based on
    the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_.
    It notably implements a distributed state-vector simulator based on `MPI <https://www.mpi-forum.org/docs/>`_.
*   ``lightning.kokkos``: a state-vector simulator written with `Kokkos <https://kokkos.github.io/kokkos-core-wiki/index.html>`_.
    It can exploit the inherent parallelism of modern processing units supporting the `OpenMP <https://www.openmp.org/>`_,
    `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ or `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest>`_ programming models.
    It also offers distributed state-vector simulation via `MPI <https://www.mpi-forum.org/docs/>`_.
*   ``lightning.tensor``: a tensor-network simulator based on the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`_.
    The supported methods are Matrix Product State (MPS) and Exact Tensor Network (TN).

If you're not sure which simulator to use, check out our `PennyLane performance <https://pennylane.ai/performance>`_ page.

Devices
*******

The Lightning ecosystem provides the following devices:

.. title-card::
    :name: 'lightning.qubit'
    :description: A fast state-vector qubit simulator written in C++
    :link: lightning_qubit/device.html

.. title-card::
    :name: 'lightning.gpu'
    :description: A heterogeneous backend state-vector simulator with NVIDIA cuQuantum library support.
    :link: lightning_gpu/device.html

.. title-card::
    :name: 'lightning.kokkos'
    :description: A heterogeneous backend state-vector simulator with Kokkos library support.
    :link: lightning_kokkos/device.html

.. title-card::
    :name: 'lightning.tensor'
    :description: A tensor network simulator with NVIDIA cuQuantum library support.
    :link: lightning_tensor/device.html

.. raw:: html

    <div style='clear:both'></div>
    </br>

Authors
*******

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

.. raw:: html

    <div style='clear:both'></div>
    </br>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   dev/installation
   dev/docker
   dev/support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   lightning_qubit/device
   lightning_gpu/device
   lightning_kokkos/device
   lightning_tensor/device

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   lightning_qubit/development/index

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code/__init__
   C++ API <api/library_root>
