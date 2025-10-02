# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Internal logic to discover and set CUDA version

The CUDA version can be set in multiple ways, with the following priority:
1. Environment Variable: PL_CUDA_VERSION (e.g. "12" or "13")
2. Environment Variable: CMAKE_ARGS (containing PL_CUDA_VERSION=...)
3. Auto discover: nvcc --version
4. Default value: "12"

The allowed CUDA major versions are "12" and "13".
"""
import os
import re
import shutil
import subprocess

_ALLOWED_CUDA_MAJOR_VERSIONS = ("12", "13")
_DEFAULT_CUDA_VERSION = "12"


def cuda_version():
    """
    Finds the CUDA version from multiple sources in a specific order of priority.

    Priority:
    1. Environment Variable: PL_CUDA_VERSION
    2. Environment Variable: CMAKE_ARGS (containing PL_CUDA_VERSION=...)
    3. System command: nvcc --version
    4. Default value

    Returns:
        str: The determined CUDA version string (e.g., "12").
    """
    version = (
        os.environ.get("PL_CUDA_VERSION")
        or _get_cuda_version_from_cmake_args()
        or _get_cuda_version_from_nvcc()
        or _DEFAULT_CUDA_VERSION
    )

    if version not in _ALLOWED_CUDA_MAJOR_VERSIONS:
        raise ValueError(
            f"Invalid CUDA version {version}. Allowed major versions are: {_ALLOWED_CUDA_MAJOR_VERSIONS}."
        )

    return version


def _get_cuda_version_from_cmake_args():
    """
    Tries to extract the CUDA version from the CMAKE_ARGS environment variable.

    Returns:
        str: The CUDA version string or None if not found.
    """
    cmake_args = os.environ.get("CMAKE_ARGS")
    if cmake_args:
        # Find an argument like 'PL_CUDA_VERSION=12'
        match = re.search(r"PL_CUDA_VERSION=(\d+)", cmake_args)
        if match:
            return match.group(1)
    return None


def _get_cuda_version_from_nvcc():
    """
    Tries to get the CUDA Toolkit version by running 'nvcc --version'.

    Returns:
        str: The CUDA major version string (e.g., "12") or None if not found.
    """
    nvcc_path = shutil.which("nvcc")

    if nvcc_path is None:
        return None

    try:
        result = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        nvcc_output = result.stdout.strip()
        match = re.search(r"release (\d+)\.\d+", nvcc_output)
        if match:
            return match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return None
