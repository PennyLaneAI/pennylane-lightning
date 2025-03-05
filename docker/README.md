# PennyLane and the Lightning plugins

PennyLane is an open-source software framework for quantum machine learning, quantum chemistry, and quantum computing, with the ability to run on all hardware. Maintained with ❤️ by Xanadu.

## Documentation

For more information on PennyLane, including the demos, APIs and development guide, visit the [PennyLane documentation site](https://pennylane.ai/).
You can also check the Lightning-plugins documentation:

- [pennylane-lightning](https://docs.pennylane.ai/projects/lightning/en/stable/)
- [pennylane-lightning-gpu](https://docs.pennylane.ai/projects/lightning-gpu/en/stable/)
- [pennylane-lightning-kokkos](https://docs.pennylane.ai/projects/lightning-kokkos/en/latest/)

## Download & install

Each image contains PennyLane and one of several high-performance plugins.
Choose a version (e.g. `v0.34.0`) and append a device (plugin) name among the following:

- `lightning-qubit`: [pennylane-lightning](https://github.com/PennyLaneAI/pennylane-lightning) provides a fast state-vector simulator written in C++.
- `lightning-gpu`: [pennylane-lightning-gpu](https://github.com/PennyLaneAI/pennylane-lightning-gpu) is a plugin based on the NVIDIA [cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk).
- `lightning-kokkos-cuda`: [pennylane-lightning-kokkos](https://github.com/PennyLaneAI/pennylane-lightning#lightning-kokkos-installation) parallelizes state-vector simulations using Kokkos' [CUDA](https://developer.nvidia.com/cuda-toolkit) backend.
- `lightning-kokkos-openmp`: [pennylane-lightning-kokkos](https://github.com/PennyLaneAI/pennylane-lightning#lightning-kokkos-installation) parallelizes state-vector simulations using Kokkos' [OpenMP](https://www.openmp.org/) backend.
- `lightning-kokkos-rocm`: [pennylane-lightning-kokkos](https://github.com/PennyLaneAI/pennylane-lightning#lightning-kokkos-installation) parallelizes state-vector simulations using Kokkos' [HIP](https://docs.amd.com/projects/HIP/en/docs-5.3.0/index.html) backend.

If you have Docker installed, download and spawn a container with `pennylane-lightning` as follows

```shell
docker run -v pwd:/io -it pennylaneai/pennylane:v0.34.0-lightning-qubit bash
```

On certain systems, there may be other solutions supporting Docker containers.
For instance, NERSC computers (e.g. Perlmutter) have [Shifter](https://docs.nersc.gov/development/shifter/) bringing containers to HPC.
In this case, spawning a container is simple as

```shell
shifterimg pull pennylaneai/pennylane:v0.34.0-lightning-qubit
shifter --image=pennylaneai/pennylane:v0.34.0-lightning-qubit /bin/bash
```

where the first command downloads the image and the second spawns a container.

## Test

You can test PennyLane and the `lightning-qubit` plugin, for example, with

```shell
pip install pytest pytest-mock flaky
pl-device-test --device default.qubit
pl-device-test --device default.qubit --shots 10000
pl-device-test --device lightning.qubit
pl-device-test --device lightning.qubit --shots 10000
```

## Build

Decide on a target among:

- `wheel-lightning-qubit`
- `wheel-lightning-gpu`
- `wheel-lightning-kokkos-cuda`
- `wheel-lightning-kokkos-openmp`
- `wheel-lightning-kokkos-rocm`

For instance `TARGET=wheel-lightning-qubit`.
Then the following command will build the target

```shell
docker build -f docker/Dockerfile --tag=${TARGET} --target ${TARGET} .
```

You may also feed a git tag or version as follows

```shell
docker build -f docker/Dockerfile --tag=${TARGET} --target ${TARGET} --build-arg="LIGHTNING_VERSION=${VERSION}" .
```

To start a container with a `bash` shell use

```shell
docker run -v `pwd`:/io -it ${TARGET} bash
```
