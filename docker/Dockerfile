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

# Setup and build pennylane-lightning


FROM ubuntu:jammy AS base
ARG KOKKOS_VERSION=4.1.00
ARG KOKKOS_BACKEND=OPENMP
ARG LIGHTNING_VERSION=0.32.0
ARG CUDA_VERSION=12.2

FROM base AS base-runtime
RUN apt-get update
RUN apt-get install --no-install-recommends -y \
    apt-utils \
    ca-certificates \
    libgomp1 \
    python3 \
    python3-pip \
    python3-venv \
    tzdata \
    wget
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv-build
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

FROM base-runtime AS base-build
RUN apt-get update
RUN apt-get install --no-install-recommends -y \
    build-essential \
    ccache \
    cmake \
    curl \
    git \
    ninja-build \ 
    python3-dev \
    gcc-11 g++-11 cpp-11
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-10
RUN /usr/sbin/update-ccache-symlinks 
RUN mkdir /opt/ccache 
RUN ccache --set-config=cache_dir=/opt/ccache 

FROM base-build AS base-build-python
RUN pip install --no-cache-dir wheel
RUN --mount=type=bind,source=requirements-dev.txt,target=requirements-dev.txt \
    pip install --no-cache-dir -r requirements-dev.txt
RUN --mount=type=bind,source=doc/requirements.txt,target=requirements-doc.txt \
    pip install --no-cache-dir -r requirements-doc.txt

FROM base-build-python AS build-wheel-lightning-qubit
WORKDIR /opt/pennylane-lightning
RUN wget https://github.com/PennyLaneAI/pennylane-lightning/archive/refs/tags/v${LIGHTNING_VERSION}.tar.gz
RUN tar -zxf v${LIGHTNING_VERSION}.tar.gz && mv pennylane-lightning-${LIGHTNING_VERSION}/* /opt/pennylane-lightning
RUN pip uninstall -y pennylane-lightning
RUN python setup.py build_ext
RUN python setup.py bdist_wheel

FROM base-runtime AS wheel-lightning-qubit
COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
RUN pip install --force-reinstall --no-cache-dir PennyLane_Lightning*.whl

FROM base-build as base-build-kokkos
WORKDIR /opt/kokkos-${KOKKOS_VERSION}
RUN wget https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VERSION}.tar.gz
RUN tar -zxf ${KOKKOS_VERSION}.tar.gz && mv /opt/kokkos-${KOKKOS_VERSION}/kokkos-${KOKKOS_VERSION}/* /opt/kokkos-${KOKKOS_VERSION}
RUN cmake -S /opt/kokkos-${KOKKOS_VERSION} -B /opt/kokkos-${KOKKOS_VERSION}/build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/kokkos \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DBUILD_TESTING:BOOL=OFF \
    -DKokkos_ENABLE_SERIAL:BOOL=ON \
    -DKokkos_ENABLE_${KOKKOS_BACKEND}:BOOL=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DKokkos_ENABLE_EXAMPLES:BOOL=OFF \
    -DKokkos_ENABLE_TESTS:BOOL=OFF \
    -DKokkos_ENABLE_LIBDL:BOOL=OFF
RUN cmake --build /opt/kokkos-${KOKKOS_VERSION}/build
RUN cmake --install /opt/kokkos-${KOKKOS_VERSION}/build

FROM base-build-python AS build-wheel-lightning-kokkos
WORKDIR /opt/pennylane-lightning
COPY --from=base-build-kokkos /opt/kokkos /opt/kokkos
ENV CMAKE_PREFIX_PATH=/opt/kokkos:$CMAKE_PREFIX_PATH
RUN wget https://github.com/PennyLaneAI/pennylane-lightning/archive/refs/tags/v${LIGHTNING_VERSION}.tar.gz
RUN tar -zxf v${LIGHTNING_VERSION}.tar.gz && mv pennylane-lightning-${LIGHTNING_VERSION}/* /opt/pennylane-lightning
RUN pip uninstall -y pennylane-lightning
RUN PL_BACKEND=lightning_kokkos python setup.py build_ext
RUN PL_BACKEND=lightning_kokkos python setup.py bdist_wheel

FROM base-runtime AS wheel-lightning-kokkos
COPY --from=base-build-kokkos /opt/kokkos /opt/kokkos
COPY --from=build-wheel-lightning-kokkos /opt/pennylane-lightning/dist/ /
RUN pip install --force-reinstall --no-cache-dir PennyLane_Lightning*.whl

FROM base-build as base-build-cuda
WORKDIR /opt/cuda-build
RUN curl -o cuda-12_2.run https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
RUN chmod a+x cuda-12_2.run
RUN ./cuda-12_2.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.2
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

FROM base-build-cuda AS build-wheel-lightning-gpu
WORKDIR /opt/pennylane-lightning
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN wget https://github.com/PennyLaneAI/pennylane-lightning-gpu/archive/refs/tags/v${LIGHTNING_VERSION}.tar.gz
RUN tar -zxf v${LIGHTNING_VERSION}.tar.gz && mv pennylane-lightning-gpu-${LIGHTNING_VERSION}/* /opt/pennylane-lightning
RUN pip uninstall -y pennylane-lightning-gpu
RUN pip install --no-cache-dir wheel custatevec-cu12
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --no-cache-dir -r requirements.txt
RUN python setup.py build_ext --cuquantum=$(python -c "import site; print( f'{site.getsitepackages()[0]}/cuquantum/lib')")
RUN python setup.py bdist_wheel

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS wheel-lightning-gpu
RUN apt-get update
RUN apt-get install --no-install-recommends -y \
    libgomp1 \
    python3 \
    python3-pip \
    python3-venv
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir custatevec-cu12
COPY --from=build-wheel-lightning-gpu /opt/pennylane-lightning/dist/ /
RUN pip install --no-cache-dir --force-reinstall PennyLane_Lightning*.whl
