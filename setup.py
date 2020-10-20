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
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


with open("pennylane_lightning/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11

        return pybind11.get_include()


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        fname = f.name

    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass

    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["-EHsc", "-O2", "-Wall", "-std:c++11"],
        "unix": ["-O3", "-Wall", "-fPIC", "-shared", "-fopenmp"],
    }

    l_opts = {
        "msvc": [],
        "unix": ["-O3", "-Wall", "-fPIC", "-shared", "-fopenmp"],
    }

    if platform.system() == "Darwin":
        for opts in (c_opts, l_opts):
            opts["unix"].remove("-fopenmp")
            opts["unix"].remove("-shared")

        darwin_opts = [
            "-stdlib=libc++",
            "-Xpreprocessor",
            "-fopenmp",
            "-lomp",
            "-mmacosx-version-min=10.9",
        ]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        if ct == "unix":
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.define_macros = [("VERSION_INFO", '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


if not os.environ.get("MOCK_DOCS", False):

    include_dirs = [
        get_pybind_include(),
        os.environ.get("EIGEN_INCLUDE_DIR", ""),
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
        "./include",
    ]

    library_dirs = [i for i in os.environ.get("LD_LIBRARY_PATH", "").split(":") if i]
    libraries = []
    extra_compile_args = []
    extra_link_args = []

    if os.environ.get("USE_LAPACK", False):
        extra_compile_args += [" -llapacke -DLAPACKE=1"]
        libraries += ["lapacke"]
        extra_link_args += ["-llapacke"]

    if os.environ.get("USE_OPENBLAS", False):
        extra_compile_args += [" -lopenblas -DLAPACKE=1"]
        libraries += ["openblas"]
        extra_link_args += ["-lopenblas"]

    if platform.system() == "Darwin":
        include_dirs += ["/usr/local/opt/libomp/include"]
        library_dirs += ["/usr/local/opt/libomp/lib"]
        libraries += ["omp"]

    ext_modules = [
        Extension(
            "lightning_qubit_ops",
            sources=["pennylane_lightning/src/lightning_qubit.cpp"],
            depends=[
                "pennylane_lightning/src/lightning_qubit.hpp",
                "pennylane_lightning/src/operations.hpp",
            ],
            include_dirs=include_dirs,
            language="c++",
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]
else:
    ext_modules = []

requirements = [
    "numpy",
    "pennylane>=0.12.0",
    "pybind11",
]


info = {
    "name": "PennyLane-Lightning",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/XanaduAI/pennylane-lightning",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "package_data": {"pennylane_lightning": ["src/*"]},
    "entry_points": {
        "pennylane.plugins": ["lightning.qubit = pennylane_lightning:LightningQubit",],
    },
    "description": "PennyLane-Lightning plugin",
    "long_description": open("README.rst").read(),
    "long_description_content_type": "text/x-rst",
    "provides": ["pennylane_lightning"],
    "install_requires": requirements,
    "ext_modules": ext_modules,
    "ext_package": "pennylane_lightning",
    "cmdclass": {"build_ext": BuildExt},
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
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
