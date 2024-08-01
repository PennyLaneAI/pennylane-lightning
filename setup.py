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
import os
import platform
import subprocess
import shutil
import sys
import toml

from pathlib import Path
from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext

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
        if backend is not None and backend != cmake_backend:
            raise ValueError(
                f"Backends {backend} and {cmake_backend} specified by PL_BACKEND and CMAKE_ARGS respectively do not match."
            )
        backend = cmake_backend
    if backend is None:
        backend = default_backend
    if backend not in supported_backends:
        raise ValueError(f"Invalid backend {backend}.")
    return backend


backend = get_backend()
device_name = backend.replace("_", ".")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = Path(sourcedir).absolute()


class CMakeBuild(build_ext):
    """
    This class is built upon https://github.com/diegoferigo/cmake-build-extension/blob/master/src/cmake_build_extension/build_extension.py and https://github.com/pybind/cmake_example/blob/master/setup.py
    """

    user_options = build_ext.user_options + [("define=", "D", "Define variables for CMake")]

    def initialize_options(self):
        super().initialize_options()
        self.define = None
        self.verbosity = ""

    def finalize_options(self):
        # Parse the custom CMake options and store them in a new attribute
        defines = [] if self.define is None else self.define.split(";")
        self.cmake_defines = [f"-D{define}" for define in defines]
        if self.verbosity != "":
            self.verbosity = "--verbose"

        super().finalize_options()

    def build_extension(self, ext: CMakeExtension):
        self.build_temp = f"build_{backend}"
        extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.absolute())
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        build_type = "Debug" if debug else "RelWithDebInfo"
        ninja_path = str(shutil.which("ninja"))

        build_args = ["--config", "Debug"] if debug else ["--config", "RelWithDebInfo"]
        configure_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={build_type}",  # not used on MSVC, but no harm
            "-DENABLE_WARNINGS=OFF",  # Ignore warnings
        ]
        configure_args += (
            [f"-DPYTHON_EXECUTABLE={sys.executable}"]
            if platform.system() == "Linux"
            else [f"-DPython_EXECUTABLE={sys.executable}"]
        )

        if platform.system() == "Windows":
            # As Ninja does not support long path for windows yet:
            #  (https://github.com/ninja-build/ninja/pull/2056)
            configure_args += [
                "-T clangcl",
            ]
        elif ninja_path:
            configure_args += [
                "-GNinja",
                f"-DCMAKE_MAKE_PROGRAM={ninja_path}",
            ]

        configure_args += [f"-DPL_BACKEND={backend}"]
        configure_args += self.cmake_defines

        # Add more platform dependent options
        if platform.system() == "Darwin":
            clang_path = Path(shutil.which("clang++")).parent.parent
            configure_args += [
                f"-DCMAKE_CXX_COMPILER={clang_path}/bin/clang++",
                f"-DCMAKE_LINKER={clang_path}/bin/lld",
                f"-DENABLE_GATE_DISPATCHER=OFF",
            ]
            if shutil.which("brew"):
                libomp_path = subprocess.run(
                    "brew --prefix libomp".split(" "),
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if not Path(libomp_path).exists():
                    libomp_path = ""
                configure_args += (
                    [f"-DOpenMP_ROOT={libomp_path}/"] if libomp_path else ["-DENABLE_OPENMP=OFF"]
                )
        elif platform.system() == "Windows":
            configure_args += ["-DENABLE_OPENMP=OFF", "-DENABLE_BLAS=OFF"]
        elif platform.system() not in ["Linux"]:
            raise RuntimeError(f"Unsupported '{platform.system()}' platform")

        if not Path(self.build_temp).exists():
            os.makedirs(self.build_temp)

        if "CMAKE_ARGS" in os.environ:
            configure_args += os.environ["CMAKE_ARGS"].split(" ")

        subprocess.check_call(
            ["cmake", str(ext.sourcedir)] + configure_args,
            cwd=self.build_temp,
            env=os.environ,
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--verbose"] + build_args,
            cwd=self.build_temp,
            env=os.environ,
        )


requirements = [
    "pennylane>=0.36",
]

packages_list = ["pennylane_lightning." + backend]

version = toml.load("pyproject.toml")["project"]["version"]
if backend == "lightning_qubit":
    packages_list += ["pennylane_lightning.core"]
else:
    requirements += ["pennylane_lightning==" + version]

suffix = backend.replace("lightning_", "")
if suffix == "gpu":
    suffix = suffix[0:].upper()
suffix = suffix[0].upper() + suffix[1:]

pennylane_plugins = [device_name + " = pennylane_lightning." + backend + ":Lightning" + suffix]

pkg_suffix = "" if suffix == "Qubit" else "_" + suffix

info = {
    "packages": find_namespace_packages(include=packages_list),
    "include_package_data": True,
    "entry_points": {"pennylane.plugins": pennylane_plugins},
    "install_requires": requirements,
    "ext_modules": (
        [] if os.environ.get("SKIP_COMPILATION", False) else [CMakeExtension(f"{backend}_ops")]
    ),
    "cmdclass": {"build_ext": CMakeBuild},
    "ext_package": "pennylane_lightning",
}

if backend == "lightning_qubit":
    info.update(
        {
            "package_data": {
                "pennylane_lightning.core": [
                    os.path.join("src", "*"),
                    os.path.join("src", "**", "*"),
                ]
            },
        }
    )

setup(**(info))
