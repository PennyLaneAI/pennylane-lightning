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
"""Tests for PennyLane-Lightning Nanobind bindings."""

import importlib
import inspect

import numpy as np
import pytest


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
        "pennylane_lightning.lightning_qubit_nb",
        "pennylane_lightning.lightning_kokkos_nb",
        "pennylane_lightning.lightning_gpu_nb",
    ],
)
def test_nanobind_module_importable(module_name):
    """Test if Nanobind module can be imported."""
    result = get_module_attributes(module_name)

    # Skip test if module is not available instead of failing
    if not result["importable"]:
        pytest.skip(f"Module {module_name} not available: {result.get('error')}")

    assert result["importable"]
    assert len(result["classes"]) > 0 or len(result["functions"]) > 0


@pytest.mark.parametrize(
    "module_name",
    [
        "pennylane_lightning.lightning_qubit_nb",
        "pennylane_lightning.lightning_kokkos_nb",
        "pennylane_lightning.lightning_gpu_nb",
    ],
)
def test_nanobind_info_function(module_name):
    """Test if Nanobind module has nb_info function and it returns expected data."""
    result = get_module_attributes(module_name)

    # Skip test if module is not available
    if not result["importable"]:
        pytest.skip(f"Module {module_name} not available: {result.get('error')}")

    module = result["module"]

    # Check if nb_info function exists
    assert hasattr(module, "nb_info"), f"nb_info function not found in {module_name}"

    # Call nb_info and verify it returns a dictionary
    info = module.nb_info()
    assert isinstance(info, dict), f"nb_info should return a dictionary, got {type(info)}"
    assert len(info) > 0, "nb_info returned an empty dictionary"


@pytest.mark.parametrize(
    "module_name",
    [
        "pennylane_lightning.lightning_qubit_nb",
        "pennylane_lightning.lightning_kokkos_nb",
        "pennylane_lightning.lightning_gpu_nb",
    ],
)
def test_nanobind_vector_operations(module_name):
    """Test vector operations in Nanobind modules."""
    result = get_module_attributes(module_name)

    # Skip test if module is not available
    if not result["importable"]:
        pytest.skip(f"Module {module_name} not available: {result.get('error')}")

    module = result["module"]

    # Check if nb_add_vectors function exists
    if not hasattr(module, "nb_add_vectors"):
        pytest.skip(f"nb_add_vectors function not found in {module_name}")

    # Test array addition with NumPy arrays
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    result = module.nb_add_vectors(a, b)
    expected = np.array([5.0, 7.0, 9.0])

    # Use NumPy's testing functions for array comparison
    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-7,
        atol=1e-7,
        err_msg=f"Array addition failed: {a} + {b} = {result}",
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "pennylane_lightning.lightning_qubit_nb",
        "pennylane_lightning.lightning_kokkos_nb",
        "pennylane_lightning.lightning_gpu_nb",
    ],
)
def test_nanobind_int_vector_class(module_name):
    """Test IntVector class in Nanobind modules."""
    result = get_module_attributes(module_name)

    # Skip test if module is not available
    if not result["importable"]:
        pytest.skip(f"Module {module_name} not available: {result.get('error')}")

    module = result["module"]

    # Check if IntVector class exists
    if not hasattr(module, "IntVector"):
        pytest.skip(f"IntVector class not found in {module_name}")

    # Test IntVector functionality
    vec = module.IntVector()
    vec.append(1)
    vec.append(2)
    vec.append(3)

    assert len(vec) == 3, f"Expected length 3, got {len(vec)}"
    assert [vec[i] for i in range(len(vec))] == [
        1,
        2,
        3,
    ], "Vector elements don't match expected values"


@pytest.mark.parametrize(
    "module_name",
    [
        "pennylane_lightning.lightning_qubit_nb",
        "pennylane_lightning.lightning_kokkos_nb",
        "pennylane_lightning.lightning_gpu_nb",
    ],
)
def test_simple_statevector(module_name):
    """Test the SimpleStateVector class"""
    result = get_module_attributes(module_name)

    # Skip test if module is not available
    if not result["importable"]:
        pytest.skip(f"Module {module_name} not available: {result.get('error')}")

    module = result["module"]

    # Check if SimpleStateVector class exists
    if not hasattr(module, "SimpleStateVector"):
        pytest.skip(f"SimpleStateVector class not found in {module_name}")

    sv = module.SimpleStateVector(2)
    assert sv is not None

    # Test that we can call the applyMatrix method
    matrix = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    sv.applyMatrix(matrix, [0], False)
