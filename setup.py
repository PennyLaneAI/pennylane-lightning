# Copyright 2020 Xanadu Quantum Technologies Inc.

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
import setuptools
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

from distutils.util import get_platform


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

    def finalize_options(self):
        # Parse the custom CMake options and store them in a new attribute
        defines = [] if self.define is None else self.define.split(";")
        self.cmake_defines = [f"-D{define}" for define in defines]

        super().finalize_options()

    def build_extension(self, ext: CMakeExtension):
        extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.absolute())

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        ninja_path = str(shutil.which("ninja"))

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        configure_args = [
            "-GNinja",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_MAKE_PROGRAM={ninja_path}",
            "-DENABLE_WARNINGS=OFF",  # Ignore warnings
        ]

        if debug:
            configure_args += ["-DCMAKE_BUILD_TYPE=Debug"]
        configure_args += self.cmake_defines

        build_args = []

        # Add more platform dependent options
        if platform.system() == "Darwin":
            # Disable OpenMP in M1 Macs
            if os.environ.get("USE_OMP"):
                configure_args += []
            else:
                configure_args += ["-DENABLE_OPENMP=OFF"]
        elif platform.system() == "Linux":
            if platform.machine() == "x86_64":
                configure_args += ["-DENABLE_AVX=ON"]  # Enable AVX if x64 on Linux
        elif platform.system() == "Windows":
            configure_args += ["-DENABLE_OPENMP=OFF", "-DENABLE_BLAS=OFF"]
        else:
            raise RuntimeError(f"Unsupported '{platform.system()}' platform")

        if not Path(self.build_temp).exists():
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", str(ext.sourcedir)] + configure_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


with open("pennylane_lightning/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "ninja",
    "numpy",
    "pennylane>=0.19",
]

info = {
    "name": "PennyLane-Lightning",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/XanaduAI/pennylane-lightning",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "package_data": {"pennylane_lightning": ["src/*", "src/**/*"]},
    "include_package_data": True,
    "entry_points": {
        "pennylane.plugins": [
            "lightning.qubit = pennylane_lightning:LightningQubit",
        ],
    },
    "description": "PennyLane-Lightning plugin",
    "long_description": open("README.rst").read(),
    "long_description_content_type": "text/x-rst",
    "provides": ["pennylane_lightning"],
    "install_requires": requirements,
    "ext_modules": [CMakeExtension("lightning_qubit_ops")]
    if not os.environ.get("SKIP_COMPILATION", False)
    else [],
    "cmdclass": {"build_ext": CMakeBuild},
    "ext_package": "pennylane_lightning",
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
