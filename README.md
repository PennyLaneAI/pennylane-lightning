<p align="center">
  <!-- 01 - Linux x86_64 L-Qubit Python tests (branch) -->
  <a href="https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_lqcpu_python.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_lqcpu_python.yml?branch=master&label=LQubit&style=flat-square" />
  </a>
  <!-- 02 - Linux x86_64 L-GPU Python tests (branch) -->
  <a href="https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_gpu_python.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_gpu_python.yml?branch=master&label=LGPU&style=flat-square" />
  </a>
  <!-- 03 - Linux x86_64 L-Kokkos Python tests (branch) -->
  <a href="https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_lkcpu_python.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_lkcpu_python.yml?branch=master&label=LKokkos&style=flat-square" />
  </a>
  <!-- 04 - Linux x86_64 L-Tensor Python tests (branch) -->
  <a href="https://github.com/PennyLaneAI/pennylane-lightning/actions/workflows/tests_gpu_python.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-lightning/tests_gpu_python.yml?branch=master&label=LTensor&style=flat-square" />
  </a>
  <!-- 05 - Codecov coverage -->
  <a href="https://codecov.io/gh/PennyLaneAI/pennylane-lightning">
    <img src="https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-lightning/master.svg?logo=codecov&style=flat-square" />
  </a>
  <!-- 06 - CodeFactor Grade -->
  <a href="https://www.codefactor.io/repository/github/pennylaneai/pennylane-lightning">
    <img src="https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane-lightning/master?logo=codefactor&style=flat-square" />
  </a>
  <!-- 07 - Read the Docs -->
  <a href="https://docs.pennylane.ai/projects/lightning">
    <img src="https://readthedocs.com/projects/xanaduai-pennylane-lightning/badge/?version=latest&style=flat-square" />
  </a>
  <!-- 08 - PennyLane Forum -->
  <a href="https://discuss.pennylane.ai">
    <img src="https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?logo=discourse&style=flat-square" />
  </a>
  <!-- 09 - PyPI - Version -->
  <a href="https://pypi.org/project/PennyLane-Lightning">
    <img src="https://img.shields.io/pypi/v/PennyLane-Lightning.svg?style=flat-square" />
  </a>
  <!-- 10 - PyPI - Python Version -->
  <a href="https://pypi.org/project/PennyLane-Lightning">
    <img src="https://img.shields.io/pypi/pyversions/PennyLane-Lightning.svg?style=flat-square" />
  </a>
  <!-- 11 - License -->
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/pypi/l/PennyLane.svg?logo=apache&style=flat-square" />
  </a>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane-lightning/master/doc/_static/readme/pl-lightning-logo-lightmode.png#gh-light-mode-only" width="700px">
    <!--
    Use a relative import for the dark mode image. When loading on PyPI, this
    will fail automatically and show nothing.
    -->
    <img src="./doc/_static/readme/pl-lightning-logo-darkmode.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt=""/>
</p>

The Lightning plugin ecosystem provides fast state-vector and tensor-network simulators written in C++.

[PennyLane](https://docs.pennylane.ai) is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.
PennyLane supports Python 3.11 and above.

## Backends

PennyLane-Lightning high performance simulators include the following backends:
* ``lightning.qubit``: a fast state-vector simulator written in C++ with optional [OpenMP](https://www.openmp.org/) additions and parallelized gate-level SIMD kernels.
* ``lightning.gpu``: a state-vector simulator based on the [NVIDIA cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk).
  It notably implements a distributed state-vector simulator based on [MPI](https://www.mpi-forum.org/docs/).
* ``lightning.kokkos``: a state-vector simulator written with [Kokkos](https://kokkos.github.io/kokkos-core-wiki/index.html).
  It can exploit the inherent parallelism of modern processing units supporting the [OpenMP](https://www.openmp.org/>`),
  [CUDA](https://developer.nvidia.com/cuda-toolkit) or [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest) programming models.
  It also offers distributed state-vector simulation via [MPI](https://www.mpi-forum.org/docs/).
* ``lightning.tensor``: a tensor-network simulator based on the [NVIDIA cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk).
  The supported methods are Matrix Product State (MPS) and Exact Tensor Network (TN).

If you're not sure which simulator to use, check out our [PennyLane Performance](https://pennylane.ai/performance) page.

## Installation

The following table summarizes the supported platforms and the primary installation mode:

|                        | Linux x86 | Linux ARM | MacOS x86 | MacOS ARM | Windows   |
|------------------------|-----------|-----------|-----------|-----------|-----------|
| Lightning-Qubit        | pip       | pip       | pip       | pip       | pip       |
| Lightning-GPU          | pip       | pip       |           |           |           |
| Lightning-GPU (MPI)    | source    |           |           |           |           |
| Lightning-Kokkos (OMP) | pip       | pip       | pip       | pip       |           |
| Lightning-Kokkos (CUDA)| source    | source    |           |           |           |
| Lightning-Kokkos (HIP) | source    | source    |           |           |           |
| Lightning-Kokkos (MPI) | source    |           |           |           |           |
| Lightning-Tensor       | pip       | pip       |           |           |           |

To install the latest stable version of these plugins,
check out the [PennyLane installation guide](https://pennylane.ai/install#high-performance-computing-and-gpus).

If you wish to install the latest development version, instructions for
[building from source](https://docs.pennylane.ai/projects/lightning/en/stable/dev/installation.html) are also available for each backend.

## Docker support

Docker images for the various backends are found on the
[PennyLane Docker Hub](https://hub.docker.com/u/pennylaneai) page, where a detailed description about PennyLane Docker support can be found.
Briefly, one can build the Docker Lightning images using:

```bash
git clone https://github.com/PennyLaneAI/pennylane-lightning.git
cd pennylane-lightning
docker build -f docker/Dockerfile --target ${TARGET} .
```

where ``${TARGET}`` is one of the following:
* ``wheel-lightning-qubit``
* ``wheel-lightning-gpu``
* ``wheel-lightning-kokkos-openmp``
* ``wheel-lightning-kokkos-cuda``
* ``wheel-lightning-kokkos-rocm``

## Contributing

We welcome contributions - simply fork the repository of this plugin, and then make a
[pull request](https://help.github.com/articles/about-pull-requests) containing your contribution.
All contributors to this plugin will be listed as authors on the releases.  
We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.

### Black & Pylint

If you contribute to the Python code, please mind the following.
The Python code is formatted with the PEP 8 compliant opinionated formatter [Black](https://github.com/psf/black) (`black==25.1.0`).
We set a line width of a 100 characters.
The Python code is statically analyzed with [Pylint](https://pylint.readthedocs.io/en/stable/).
We set up a pre-commit hook (see [Git hooks](https://git-scm.com/docs/githooks)) to run both of these on `git commit`.
Please make your best effort to comply with `black` and `pylint` before using disabling pragmas (e.g. `# pylint: disable=missing-function-docstring`).

## Authors

Lightning is the work of [many contributors](https://github.com/PennyLaneAI/pennylane-lightning/graphs/contributors).

If you are using Lightning for research, please cite:

```bibtex
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
```

## Support

- **Source Code:** https://github.com/PennyLaneAI/pennylane-lightning
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-lightning/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

## License

The Lightning plugins are **free** and **open source**, released under
the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
The Lightning-GPU and Lightning-Tensor plugins make use of the NVIDIA cuQuantum SDK headers to
enable the device bindings to PennyLane, which are held to their own respective license.

## Acknowledgements

PennyLane Lightning makes use of the following libraries and tools, which are under their own respective licenses:

- **nanobind:** https://github.com/wjakob/nanobind
- **Kokkos Core:** https://github.com/kokkos/kokkos
- **NVIDIA cuQuantum:** https://developer.nvidia.com/cuquantum-sdk
- **scipy-openblas32:** https://pypi.org/project/scipy-openblas32/
- **Xanadu JET:** https://github.com/XanaduAI/jet