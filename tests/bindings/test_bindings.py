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
"""Tests for PennyLane-Lightning Pybind11 bindings."""

import importlib
import inspect

import pytest
from conftest import (  # tested device
    LightningAdjointJacobian,
    LightningDevice,
    LightningStateVector,
    device_name,
)


def get_module_attributes(module_name):
    """Get classes and functions from a module."""
    try:
        module = importlib.import_module(module_name)

        classes = []
        functions = []

        for name in dir(module):
            if name.startswith("_") and not name.startswith("__"):
                continue

            attr = getattr(module, name)
            if inspect.isclass(attr):
                classes.append(name)
            elif inspect.isfunction(attr) or callable(attr):
                functions.append(name)

        return {"importable": True, "classes": classes, "functions": functions, "module": module}
    except ImportError as e:
        return {"importable": False, "error": str(e)}


@pytest.mark.parametrize(
    "module_name",
    [
        "pennylane_lightning.lightning_qubit",
        "pennylane_lightning.lightning_kokkos",
        "pennylane_lightning.lightning_gpu",
    ],
)
def test_module_importable(module_name):
    """Test if module can be imported."""
    result = get_module_attributes(module_name)

    # Skip test if module is not available instead of failing
    if not result["importable"]:
        pytest.skip(f"Module {module_name} not available: {result.get('error')}")

    assert result["importable"]
    assert len(result["classes"]) > 0 or len(result["functions"]) > 0


@pytest.mark.parametrize(
    "module_name, expected_class",
    [
        ("pennylane_lightning.lightning_qubit", "LightningQubit"),
        ("pennylane_lightning.lightning_kokkos", "LightningKokkos"),
        ("pennylane_lightning.lightning_gpu", "LightningGPU"),
    ],
)
def test_module_has_expected_attributes(module_name, expected_class):
    """Test if module has expected attributes."""
    result = get_module_attributes(module_name)

    # Skip test if module is not available
    if not result["importable"]:
        pytest.skip(f"Module {module_name} not available: {result.get('error')}")

    # Check for the main device class
    assert expected_class in result["classes"], f"{expected_class} not found in {module_name}"

    # Check for backend_info if available
    if hasattr(result["module"], "backend_info"):
        assert "backend_info" in result["functions"], f"backend_info not found in {module_name}"
