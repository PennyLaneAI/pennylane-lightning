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
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from backend_support import backend, device_name

path_to_project = Path(__file__).parent.parent.absolute()

path_to_version = path_to_project / Path("pennylane_lightning") / "core" / "_version.py"
with open(path_to_version, encoding="utf-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


# ------------------------
# Find the toml parser.
# ------------------------
has_toml = False
toml_libs = ["tomlkit", "toml"]  # "tomllib" and "tomli" do not implement 'dump'.
for pkg in toml_libs:
    spec = find_spec(pkg)
    if spec:
        toml = import_module(pkg)
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

    if parsed_args.path.strip() == "":
        parsed_args.path = path_to_project.as_posix()

    pyproject_path = os.path.join(parsed_args.path, "pyproject.toml")

    if not has_toml:
        raise ImportError(
            "A TOML parser is required to configure 'pyproject.toml'. "
            f"We support any of the following TOML parsers: {toml_libs} "
            "You can install tomlkit via `pip install tomlkit`."
        )

    try:
        with open(pyproject_path, "rb") as f:
            pyproject = toml.load(f)
    except TypeError:
        # To support toml and tomli APIs
        pyproject = toml.load(pyproject_path)

    # ------------------------
    # Configure Build.
    # ------------------------
    requires = [
        "cmake",
        "ninja; platform_system!='Windows'",
        "setuptools>=75.8.1",
        "tomli",
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

    # Specifying the project name.
    pkg_suffix = "" if suffix == "qubit" else "_" + suffix

    pyproject["project"]["name"] = f"pennylane_lightning{pkg_suffix}"

    # Specifying the Lightning module name.
    module_suffix = suffix.upper() if suffix == "gpu" else suffix.title()

    module_name = "pennylane_lightning." + backend + ":Lightning" + module_suffix

    # Project entry point.
    pyproject["project"]["entry-points"]["pennylane.plugins"] = {device_name: module_name}

    dependencies = [
        "pennylane>=0.41",
        "scipy-openblas32>=0.3.26",
    ]

    if backend == "lightning_gpu":
        dependencies += ["custatevec-cu12"]

    if backend == "lightning_tensor":
        dependencies += ["cutensornet-cu12", "nvidia-cusolver-cu12"]

    if backend in ("lightning_gpu", "lightning_tensor"):
        dependencies += [
            "nvidia-nvjitlink-cu12",
            "nvidia-cusparse-cu12",
            "nvidia-cublas-cu12",
            "nvidia-cuda-runtime-cu12",
        ]

    if backend != "lightning_qubit":
        dependencies += ["pennylane_lightning==" + version]

    # Package requirements.
    pyproject["project"]["dependencies"] = dependencies

    # Edit Classifiers based on the backend.
    windows_classifier = "Operating System :: Microsoft :: Windows"
    classifiers = pyproject["project"]["classifiers"]

    if backend != "lightning_qubit":
        if windows_classifier in classifiers:
            classifiers.remove(windows_classifier)
    else:
        if windows_classifier not in classifiers:
            idx = classifiers.index("Operating System :: MacOS :: MacOS X")
            classifiers.insert(idx + 1, windows_classifier)

    with open(pyproject_path, "w", encoding="utf-8") as file:
        toml.dump(pyproject, file)
