# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:latest AS base

# Setup and install basic packages
RUN apt-get update \ 
    && apt-get install -y apt-utils --no-install-recommends \
    && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y tzdata \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/sbin/update-ccache-symlinks \
    && mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache \
    && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Setup and build pennylane-lightning
WORKDIR /opt/pennylane-lightning

COPY . .

RUN pip install --no-cache-dir wheel \
    pytest \
    pytest-cov \
    pytest-mock \
    flaky \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y PennyLane_Lightning \
    && make install \
    && make test-python \
    && make test-cpp
