# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

default_backend = "lightning_qubit"
supported_backends = {"lightning_kokkos", "lightning_qubit", "lightning_gpu", "lightning_tensor"}
supported_backends.update({sb.replace("_", ".") for sb in supported_backends})


def get_backend():
    """Return backend.

    The backend is ``lightning_qubit`` by default.
    Allowed values are: "lightning_kokkos", "lightning_qubit" and "lightning_gpu".
    A dot can also be used instead of an underscore.
    If the environment variable ``PL_BACKEND`` is defined, its value is used.
    Otherwise, if the environment variable ``CMAKE_ARGS`` is defined and it
    contains the CMake option ``PL_BACKEND``, its value is used.
    Dots are replaced by underscores upon exiting.
    """
    backend = None
    if "PL_BACKEND" in os.environ:
        backend = os.environ.get("PL_BACKEND", default_backend)
        backend = backend.replace(".", "_")
    if "CMAKE_ARGS" in os.environ:
        cmake_args = os.environ["CMAKE_ARGS"].split(" ")
        arg = [x for x in cmake_args if "PL_BACKEND" in x]
        if not arg and backend is not None:
            cmake_backend = backend
        else:
            cmake_backend = arg[0].split("=")[1].replace(".", "_") if arg else default_backend
        # CMake determined backend will always take precedence over PL_BACKEND.
        backend = cmake_backend
    if backend is None:
        backend = default_backend
    if backend not in supported_backends:
        raise ValueError(f"Invalid backend {backend}.")
    return backend


backend = get_backend()
device_name = backend.replace("_", ".")
