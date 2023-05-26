# Copyright 2022 Xanadu Quantum Technologies Inc.

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
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


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
        extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.absolute())
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        build_type = "Debug" if debug else "RelWithDebInfo"
        ninja_path = str(shutil.which("ninja"))

        build_args = ["--config", "Debug"] if debug else ["--config", "RelWithDebInfo"]
        configure_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={build_type}",  # not used on MSVC, but no harm
            "-DENABLE_WARNINGS=OFF",  # Ignore warnings
            *(self.cmake_defines),
        ]

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

        configure_args += self.cmake_defines

        # Add more platform dependent options
        if platform.system() == "Darwin":
            clang_path = Path(shutil.which("clang++")).parent.parent
            configure_args += [
                f"-DCMAKE_CXX_COMPILER={clang_path}/bin/clang++",
                f"-DCMAKE_LINKER={clang_path}/bin/lld",
            ]
            if shutil.which("brew"):
                libomp_path = subprocess.run(
                    "brew --prefix libomp".split(" "),
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                configure_args += (
                    [f"-DOpenMP_ROOT={libomp_path}/"] if libomp_path else ["-DENABLE_OPENMP=OFF"]
                )
        elif platform.system() == "Windows":
            configure_args += ["-DENABLE_OPENMP=OFF", "-DENABLE_BLAS=OFF"]
        elif platform.system() not in ["Linux"]:
            raise RuntimeError(f"Unsupported '{platform.system()}' platform")

        if not Path(self.build_temp).exists():
            os.makedirs(self.build_temp)

        if "CMAKE_ARGS" not in os.environ.keys():
            os.environ["CMAKE_ARGS"] = ""

        subprocess.check_call(
            ["cmake"] + [str(ext.sourcedir)] + configure_args + os.environ["CMAKE_ARGS"].split(" "),
            cwd=self.build_temp,
            env=os.environ,
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--verbose"] + build_args,
            cwd=self.build_temp,
            env=os.environ,
        )


with open(os.path.join("pennylane_lightning", "_version.py")) as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "pennylane>=0.30",
]

info = {
    "name": "PennyLane-Lightning",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/XanaduAI/pennylane-lightning",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "package_data": {
        "pennylane_lightning": [
            os.path.join("src", "*"),
            os.path.join("src", "**", "*"),
        ]
    },
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
    "ext_modules": []
    if os.environ.get("SKIP_COMPILATION", False)
    else [CMakeExtension("lightning_qubit_ops")],
    "cmdclass": {"build_ext": CMakeBuild},
    "ext_package": "pennylane_lightning",
    "extras_require": {"gpu": ["pennylane-lightning-gpu"]},
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
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
