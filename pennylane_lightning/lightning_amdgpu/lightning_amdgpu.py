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
This module contains the :class:`~.LightningAmdgpu` class, a Lightning simulator device that derives from :class:`~.LightningKokkos` specifically for AMD GPUs.
"""

import ctypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pennylane_lightning.lightning_kokkos import LightningKokkos


# pylint: disable=too-few-public-methods
class LightningAmdgpu(LightningKokkos):
    """PennyLane-Lightning AMDGPU device.

    A device alias for LightningKokkos targeting AMD GPU platforms.
    """

    def __init__(self, wires=None, *args, **kwargs):
        self._check_amd_gpu_resources()
        super().__init__(wires, *args, **kwargs)

    def _get_rocm_version_from_amd_smi(self):
        amd_smi_path = shutil.which("amd-smi")
        if not amd_smi_path:
            return None

        try:
            output = subprocess.check_output([amd_smi_path, "version"], text=True)
            match = re.search(r"ROCm version:\s*([\d.]+)", output)
            if match:
                return match.group(1)

        except (subprocess.CalledProcessError, OSError):
            return None

        return None

    def _get_rocm_version_from_hipconfig(self):
        hipconfig_path = shutil.which("hipconfig")
        if not hipconfig_path:
            return None

        try:
            output = subprocess.check_output([hipconfig_path, "--version"], text=True).strip()
            match = re.search(r"^([\d.]+)", output)
            if match:
                return match.group(1)

        except (subprocess.CalledProcessError, OSError):
            return None

        return None

    def _check_amd_gpu_resources(self):
        # Detect local system ROCm version
        local_version_str = (
            self._get_rocm_version_from_amd_smi() or self._get_rocm_version_from_hipconfig()
        )

        # Get the library path
        lib_path = None
        try:
            _, lib_path = self.get_c_interface()
        except Exception as e:
            raise RuntimeError(
                "Lightning AMDGPU shared library not found. Please check installation."
            ) from e

        # Try to load the library and query version
        lib_version_str = None
        num_devices = 0

        if lib_path and os.path.exists(lib_path):
            try:
                # If dependencies (like libamdhip64.so.7) are missing,
                # this line triggers OSError immediately.
                hip_lib = ctypes.CDLL(lib_path)

                # If we reached here, the library loaded. Now query the version.
                version_val = ctypes.c_int()
                status = hip_lib.hipRuntimeGetVersion(ctypes.byref(version_val))

                if status == 0:
                    val = version_val.value
                    # Logic to unpack the integer version
                    # ROCm uses: Major * 10^7 + Minor * 10^5 + Patch
                    # https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/initialization_and_version.html#_CPPv419hipDriverGetVersionPi
                    major = val // 10000000
                    minor = (val % 10000000) // 100000
                    patch = val % 100000
                    lib_version_str = f"{major}.{minor}.{patch}"

                dev_count_val = ctypes.c_int()
                status_d = hip_lib.hipGetDeviceCount(ctypes.byref(dev_count_val))

                if status_d == 0:
                    num_devices = dev_count_val.value

            except OSError as e:
                # The library exists, but dependencies (like .so.7) are missing.
                # This handle cases when e.g. library is built for ROCm 6.x but system has ROCm 7.x
                error_msg = str(e)

                print(f"\nERROR: Failed to load library at {lib_path}")
                print(f"OS Error Details: {error_msg}")

                if local_version_str:
                    print(f"Detected System ROCm Version: {local_version_str}")

                    if "cannot open shared object file" in error_msg:
                        print("-" * 60)
                        print(
                            "Possible Cause: The library was compiled for a different ROCm version"
                        )
                        print(f"than the one installed on this system ({local_version_str}).")
                        print("Please ensure your compiled binaries match the system ROCm version.")
                        print("-" * 60)

                raise RuntimeError(
                    "ROCm library loading failed due to missing dependencies."
                ) from e

        # If load succeeded, check Major Version Compatibility
        if local_version_str and lib_version_str:
            try:
                local_major = int(local_version_str.split(".", maxsplit=1)[0])
                lib_major = int(lib_version_str.split(".", maxsplit=1)[0])

                if local_major != lib_major:
                    raise RuntimeError(
                        f"ROCm major version mismatch: System is {local_version_str} (Major {local_major}), "
                        f"but library was compiled against {lib_version_str} (Major {lib_major})."
                    )
            except ValueError:
                print("Warning: Could not parse ROCm versions for integer comparison.")

        if num_devices == 0:
            raise RuntimeError(
                "No supported AMD GPU devices found. Please ensure that an AMD GPU is available and accessible."
            )

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        The difference with the base implementation is that for editable mode,
        the shared library is in `build_lightning_amdgpu` instead of following its actual
        device name (Lightning Kokkos).
        """
        lightning_device_name = "LightningKokkosSimulator"
        lightning_lib_name = "lightning_kokkos"

        # The shared object file extension varies depending on the underlying operating system
        file_extension = ""
        OS = sys.platform
        if OS == "linux":
            file_extension = ".so"
        elif OS == "darwin":
            file_extension = ".dylib"
        else:
            raise RuntimeError(
                f"'{lightning_device_name}' shared library not available for '{OS}' platform"
            )

        lib_name = "lib" + lightning_lib_name + "_catalyst" + file_extension
        package_root = Path(__file__).parent

        # The absolute path of the plugin shared object varies according to the installation mode.

        # Wheel mode:
        # Fixed location at the root of the project
        wheel_mode_location = package_root.parent / lib_name
        if wheel_mode_location.is_file():
            return lightning_device_name, wheel_mode_location.as_posix()

        # Editable mode:
        build_lightning_dir = "build_lightning_amdgpu"
        editable_mode_path = package_root.parent.parent / build_lightning_dir
        for path, _, files in os.walk(editable_mode_path):
            if lib_name in files:
                lib_location = (Path(path) / lib_name).as_posix()
                return lightning_device_name, lib_location

        raise RuntimeError(f"'{lightning_device_name}' shared library not found")
