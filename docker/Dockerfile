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

FROM ubuntu:jammy AS base

# Setup and install basic packages
RUN apt-get update
RUN apt-get install --no-install-recommends -y apt-utils
RUN DEBIAN_FRONTEND="noninteractive" \
    apt-get install --no-install-recommends -y tzdata \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    gcc-11 g++-11 cpp-11 
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-10
RUN /usr/sbin/update-ccache-symlinks 
RUN mkdir /opt/ccache 
RUN ccache --set-config=cache_dir=/opt/ccache 

ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Setup and build pennylane-lightning
ENV WORKDIR=/opt/pennylane-lightning
WORKDIR ${WORKDIR}
COPY . ${WORKDIR}

RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir -r requirements-dev.txt
RUN pip install --no-cache-dir -r doc/requirements.txt
RUN pip uninstall -y pennylane-lightning
RUN python setup.py build_ext
RUN python setup.py bdist_wheel
RUN python setup.py install
