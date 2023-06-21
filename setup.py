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
from setuptools import find_packages
from skbuild import setup


with open(os.path.join("pennylane_lightning", "_version.py")) as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "pennylane>=0.30",
]

cmake_args = []
if "CMAKE_ARGS" in os.environ.keys():
    cmake_args += os.environ["CMAKE_ARGS"].split(" ")

info = {
    "name": "PennyLane-Lightning",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "https://github.com/XanaduAI/pennylane-lightning",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "package_data": {"pennylane_lightning": ["*.py", "*.hpp", "*.cpp", "*.txt", "*.md"]},
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
    "cmake_args": cmake_args,
    "cmake_languages": ("CXX",),
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
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
