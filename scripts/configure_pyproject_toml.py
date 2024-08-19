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
r"""
Project configuration script.
"""
import argparse
import os
from pathlib import Path

from importlib import import_module
from importlib.util import find_spec
from backend_support import backend, device_name

path_to_version = Path("pennylane_lightning").absolute() / "core" / "_version.py"
with open(path_to_version, encoding="utf-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


has_toml = False
toml_libs = ["tomllib", "tomli", "tomlkit"]
for pkg in toml_libs:
    spec = find_spec(pkg)
    if spec:
        tomllib = import_module(pkg)
        has_toml = True
        break

########################################################################
# Parsing arguments
########################################################################
def parse_args():
    """Parse external arguments provided to the script."""
    parser = argparse.ArgumentParser(
        prog="python configure_pyproject_toml.py",
        description="This module configures the pyproject.toml file for a Lightning backend (package).",
    )

    parser.add_argument(
        "--path",
        type=str,
        default="",
        nargs="?",
        help="pyproject.toml file path",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    pyproject_path = os.path.join(parsed_args.path, "pyproject.toml")

    if not has_toml:
        raise ImportError(
            "A TOML parser is required to configure 'pyproject.toml'. "
            "We support any of the following TOML parsers: [tomli, tomlkit, tomllib] "
            "You can install either tomli via `pip install tomli`, "
            "tomlkit via `pip install tomlkit`, or use Python 3.11 "
            "or above which natively offers the tomllib library."
        )

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    # ------------------------
    # Configure Build.
    # ------------------------
    requires = [
        "cmake~=3.24.0",
        "ninja; platform_system!='Windows'",
        "setuptools>=42",
        "toml",
    ]
    if backend == "lightning_gpu":
        requires.append("custatevec-cu12")
    if backend == "lightning_tensor":
        requires.append("cutensornet-cu12")

    pyproject["build-system"]["requires"] = requires

    # ------------------------
    # Configure Project.
    # ------------------------
    suffix = backend.replace("lightning_", "")
    suffix = suffix.upper() if suffix == "gpu" else suffix.title()

    plugin = "pennylane_lightning." + backend + ":Lightning" + suffix

    pkg_suffix = "" if suffix == "Qubit" else "_" + suffix

    # Specifying the project name.
    pyproject["project"]["name"] = f"PennyLane_Lightning{pkg_suffix}"

    # Project entry point.
    pyproject["project"]["entry-points"]["pennylane.plugins"] = {device_name: plugin}

    dependencies = [
        "pennylane>=0.37",
    ]

    if backend != "lightning_qubit":
        dependencies += ["pennylane_lightning==" + version]

    # Package requirements.
    pyproject["project"]["dependencies"] = dependencies

    with open(pyproject_path, "w", encoding="utf-8") as file:
        tomllib.dump(pyproject, file)
