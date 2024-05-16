# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Create basic runtime environment base on Ubuntu 22.04 (jammy)
# Define build arguments
# Create and activate runtime virtual environment
FROM ubuntu:jammy AS base-runtime
ARG CUDA_ARCH=AMPERE80
ARG AMD_ARCH=AMD_GFX90A
ARG CUDA_INSTALLER=https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
ARG DEBIAN_FRONTEND=noninteractive
ARG GCC_VERSION=11
ARG LIGHTNING_VERSION=v0.36.0
ARG PENNYLANE_VERSION=v0.36.0
ARG ROCM_INSTALLER=https://repo.radeon.com/amdgpu-install/6.1.1/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    apt-utils \
    ca-certificates \
    git \
    libgomp1 \
    python3 \
    python3-pip \
    python3-venv \
    tzdata \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create basic build environment with build tools and compilers
FROM base-runtime AS base-build
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    build-essential \
    ccache \
    cmake \
    curl \
    ninja-build \
    python3-dev \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} cpp-${GCC_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VERSION}
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache
RUN ccache --set-config=cache_dir=/opt/ccache

# Create and activate build virtual environment
# Install Lightning dev requirements
FROM base-build AS base-build-python
WORKDIR /opt/pennylane-lightning
ENV VIRTUAL_ENV=/opt/venv-build
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN rm -rf tmp && git clone --depth 1 --branch ${LIGHTNING_VERSION} https://github.com/PennyLaneAI/pennylane-lightning.git tmp\
    && mv tmp/* /opt/pennylane-lightning && rm -rf tmp
RUN pip install --no-cache-dir cmake ninja pybind11 wheel

# Download Lightning release and build lightning-qubit backend
FROM base-build-python AS build-wheel-lightning-qubit
WORKDIR /opt/pennylane-lightning
RUN pip uninstall -y pennylane-lightning
RUN python setup.py build_ext
RUN python setup.py bdist_wheel

# Install lightning-qubit backend
FROM base-runtime AS wheel-lightning-qubit
COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
RUN pip install --force-reinstall --no-cache-dir PennyLane_Lightning*.whl && rm PennyLane_Lightning*.whl
RUN pip install --no-cache-dir git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION}

# Download Lightning release and build lightning-kokkos backend with Kokkos-OpenMP
FROM base-build-python AS build-wheel-lightning-kokkos-openmp
WORKDIR /opt/pennylane-lightning
ENV PL_BACKEND=lightning_kokkos
RUN pip uninstall -y pennylane-lightning
RUN python setup.py build_ext -i --define="PL_BACKEND=lightning_kokkos;Kokkos_ENABLE_SERIAL:BOOL=ON;Kokkos_ENABLE_OPENMP:BOOL=ON"
RUN python setup.py bdist_wheel

# Install lightning-kokkos OpenMP backend
FROM base-runtime AS wheel-lightning-kokkos-openmp
COPY --from=build-wheel-lightning-kokkos-openmp /opt/pennylane-lightning/dist/ /
COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
RUN pip install --force-reinstall --no-cache-dir PennyLane_Lightning*.whl && rm PennyLane_Lightning*.whl
RUN pip install --no-cache-dir git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION}

# Install CUDA-12 in build venv image
FROM base-build-python as base-build-cuda
WORKDIR /opt/cuda-build
RUN curl -o cuda-install.run ${CUDA_INSTALLER}
RUN chmod a+x cuda-install.run
RUN ./cuda-install.run --silent --toolkit --toolkitpath=/usr/local/cuda-$(echo ${CUDA_INSTALLER} | grep -o -P '/cuda/.{0,4}' | cut -d / -f 3)
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Download Lightning release and build lightning-kokkos backend with Kokkos-CUDA
FROM base-build-cuda AS build-wheel-lightning-kokkos-cuda
WORKDIR /opt/pennylane-lightning
ENV PL_BACKEND=lightning_kokkos
RUN pip uninstall -y pennylane-lightning
RUN echo >> cmake/support_kokkos.cmake && echo "find_package(CUDAToolkit REQUIRED)" >> cmake/support_kokkos.cmake
RUN python setup.py build_ext -i --define="PL_BACKEND=lightning_kokkos;Kokkos_ENABLE_SERIAL:BOOL=ON;Kokkos_ENABLE_OPENMP:BOOL=ON;Kokkos_ENABLE_CUDA:BOOL=ON;Kokkos_ARCH_${CUDA_ARCH}=ON"
RUN python setup.py bdist_wheel

# Install python3 and setup runtime virtual env in CUDA-12-runtime image (includes CUDA runtime and math libraries)
# Install lightning-kokkos CUDA backend
FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS wheel-lightning-kokkos-cuda
ARG PENNYLANE_VERSION=v0.36.0
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    libgomp1 \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=build-wheel-lightning-kokkos-cuda /opt/pennylane-lightning/dist/ /
COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
RUN pip install --force-reinstall --no-cache-dir PennyLane_Lightning*.whl && rm PennyLane_Lightning*.whl
RUN pip install --no-cache-dir git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION}

# Download and build Lightning-GPU release
FROM base-build-cuda AS build-wheel-lightning-gpu
WORKDIR /opt/pennylane-lightning
RUN pip install --no-cache-dir wheel custatevec-cu12
RUN pip uninstall -y pennylane-lightning
RUN CUQUANTUM_SDK=$(python -c "import site; print( f'{site.getsitepackages()[0]}/cuquantum/lib')") PL_BACKEND=lightning_gpu python setup.py build_ext
RUN python setup.py bdist_wheel

# Install python3 and setup runtime virtual env in CUDA-12-runtime image (includes CUDA runtime and math libraries)
# Install lightning-kokkos CUDA backend
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS wheel-lightning-gpu
ARG PENNYLANE_VERSION=v0.36.0
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    git \
    libgomp1 \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir custatevec-cu12
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/cuquantum/lib:$LD_LIBRARY_PATH"
COPY --from=build-wheel-lightning-gpu /opt/pennylane-lightning/dist/ /
COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
RUN pip install --no-cache-dir --force-reinstall PennyLane_Lightning*.whl && rm PennyLane_Lightning*.whl
RUN pip install --no-cache-dir git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION}

# Install ROCm in build venv image
FROM base-build-python as base-build-rocm
RUN wget --progress=dot:giga ${ROCM_INSTALLER}
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)" \
    ./$(echo ${ROCM_INSTALLER} | xargs -I {} basename {}) \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN amdgpu-install -y --accept-eula --usecase=rocm,hiplibsdk --no-dkms

# Download Lightning release and build lightning-kokkos backend with Kokkos-ROCm
FROM base-build-rocm AS build-wheel-lightning-kokkos-rocm
WORKDIR /opt/pennylane-lightning
ENV CMAKE_PREFIX_PATH=/opt/rocm:$CMAKE_PREFIX_PATH
ENV CXX=hipcc
ENV PL_BACKEND=lightning_kokkos
RUN pip uninstall -y pennylane-lightning
RUN python setup.py build_ext -i --define="PL_BACKEND=lightning_kokkos;Kokkos_ENABLE_SERIAL:BOOL=ON;Kokkos_ENABLE_OPENMP:BOOL=ON;Kokkos_ENABLE_HIP:BOOL=ON;Kokkos_ARCH_${AMD_ARCH}=ON"
RUN python setup.py bdist_wheel

# Install lightning-kokkos HIP backend
FROM rocm/dev-ubuntu-22.04:5.7 AS wheel-lightning-kokkos-rocm
ARG PENNYLANE_VERSION=v0.36.0
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    git \
    libgomp1 \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=build-wheel-lightning-kokkos-rocm /opt/pennylane-lightning/dist/ /
COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
RUN pip install --force-reinstall --no-cache-dir PennyLane_Lightning*.whl && rm PennyLane_Lightning*.whl
RUN pip install --no-cache-dir git+https://github.com/PennyLaneAI/pennylane.git@${PENNYLANE_VERSION}
