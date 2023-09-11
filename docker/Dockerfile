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
RUN apt-get clean

FROM base AS build-utils
RUN DEBIAN_FRONTEND="noninteractive" \
    apt-get install --no-install-recommends -y \
    build-essential \
    ccache \
    cmake \
    curl \
    git \
    ninja-build \ 
    python3-dev \
    gcc-11 g++-11 cpp-11 libomp-dev
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-10
RUN /usr/sbin/update-ccache-symlinks 
RUN mkdir /opt/ccache 
RUN ccache --set-config=cache_dir=/opt/ccache 

FROM build-utils AS python-venv
ENV VIRTUAL_ENV=/opt/venv-build
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir wheel
RUN --mount=type=bind,source=requirements-dev.txt,target=requirements-dev.txt \
    pip install --no-cache-dir -r requirements-dev.txt
RUN --mount=type=bind,source=doc/requirements.txt,target=requirements-doc.txt \
    pip install --no-cache-dir -r requirements-doc.txt

FROM python-venv AS build-wheel-lightning-qubit
WORKDIR /opt/pennylane-lightning
RUN wget https://github.com/PennyLaneAI/pennylane-lightning/archive/refs/tags/v${LIGHTNING_VERSION}.tar.gz
RUN tar -zxf v${LIGHTNING_VERSION}.tar.gz && mv pennylane-lightning-${LIGHTNING_VERSION}/* /opt/pennylane-lightning
RUN pip uninstall -y pennylane-lightning
RUN python setup.py build_ext
RUN python setup.py bdist_wheel

FROM base AS wheel-lightning-qubit
RUN rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=build-wheel-lightning-qubit /opt/pennylane-lightning/dist/ /
RUN pip install --force-reinstall --no-cache-dir PennyLane_Lightning*.whl

FROM build-utils as build-kokkos
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

FROM python-venv AS build-wheel-lightning-kokkos
WORKDIR /opt/pennylane-lightning
COPY --from=build-kokkos /opt/kokkos /opt/kokkos
ENV CMAKE_PREFIX_PATH=/opt/kokkos:$CMAKE_PREFIX_PATH
RUN wget https://github.com/PennyLaneAI/pennylane-lightning/archive/refs/tags/v${LIGHTNING_VERSION}.tar.gz
RUN tar -zxf v${LIGHTNING_VERSION}.tar.gz && mv pennylane-lightning-${LIGHTNING_VERSION}/* /opt/pennylane-lightning
RUN pip uninstall -y pennylane-lightning
RUN PL_BACKEND=lightning_kokkos python setup.py build_ext
RUN PL_BACKEND=lightning_kokkos python setup.py bdist_wheel

FROM base AS wheel-lightning-kokkos
RUN rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=build-kokkos /opt/kokkos /opt/kokkos
COPY --from=build-wheel-lightning-kokkos /opt/pennylane-lightning/dist/ /
RUN pip install --force-reinstall --no-cache-dir PennyLane_Lightning*.whl

