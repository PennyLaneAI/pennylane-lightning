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
from conftest import device_module_name, device_name


def get_module_attributes(module):
    """Get classes and functions from a module."""
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

    return {"classes": classes, "functions": functions}


class TestNanobindBindings:
    """Tests for nanobind-based bindings."""

    @pytest.fixture(autouse=True)
    def setup_module_attributes(self, current_module):
        """Set up module attributes for all tests."""
        self.nb_module = current_module
        self.nb_module_attr = get_module_attributes(self.nb_module)

    def test_module_has_classes_and_functions(self):
        """Test if module has classes and functions."""
        assert len(self.nb_module_attr["classes"]) > 0 or len(self.nb_module_attr["functions"]) > 0

    def test_module_has_expected_info_dicts(self):
        """Test if module has expected info dicts."""
        # Check for info functions
        assert (
            "compile_info" in self.nb_module_attr["functions"]
        ), f"compile_info not found in module"
        assert (
            "runtime_info" in self.nb_module_attr["functions"]
        ), f"runtime_info not found in module"

    def test_statevector_classes_exists(self):
        """Test if StateVectorC classes exists in the module."""
        for precision in ["64", "128"]:
            if device_name == "lightning.tensor":
                for method in ["exact", "mps"]:
                    class_name = f"{method}TensorNetC{precision}"
                    assert (
                        class_name in self.nb_module_attr["classes"]
                    ), f"{class_name} not found in module"
            else:
                assert (
                    f"StateVectorC{precision}" in self.nb_module_attr["classes"]
                ), f"StateVectorC{precision} not found in module"

    def test_observables_submodule_exists(self, precision):
        """Test that the observables submodule exists and contains expected classes."""
        # Check if observables submodule exists
        assert hasattr(self.nb_module, "observables"), "Module does not have observables submodule"

        if device_name == "lightning.tensor":
            prefixes = ["exact", "mps"]
        else:
            prefixes = [""]

        for prefix in prefixes:
            # Check for NamedObs classes
            assert (
                f"{prefix}NamedObsC{precision}" in self.nb_module.observables.__dir__()
            ), f"{prefix}NamedObsC{precision} not found in observables submodule"

            # Check for HermitianObs classes
            assert (
                f"{prefix}HermitianObsC{precision}" in self.nb_module.observables.__dir__()
            ), f"{prefix}HermitianObsC{precision} not found in observables submodule"

            # Check for TensorProdObs classes
            assert (
                f"{prefix}TensorProdObsC{precision}" in self.nb_module.observables.__dir__()
            ), f"{prefix}TensorProdObsC{precision} not found in observables submodule"

            # Check for Hamiltonian classes
            assert (
                f"{prefix}HamiltonianC{precision}" in self.nb_module.observables.__dir__()
            ), f"{prefix}HamiltonianC{precision} not found in observables submodule"

    def test_algorithms_submodule_exists(self, precision):
        """Test that the algorithms submodule exists and contains expected classes."""
        if device_name == "lightning.tensor":
            pytest.skip("lightning.tensor does not have algorithms submodule")

        # Check if algorithms submodule exists
        assert hasattr(self.nb_module, "algorithms"), "Module does not have algorithms submodule"

        # Check for OpsStruct classes
        assert (
            f"OpsStructC{precision}" in self.nb_module.algorithms.__dir__()
        ), f"OpsStructC{precision} not found in algorithms submodule"

        assert (
            f"create_ops_listC{precision}" in self.nb_module.algorithms.__dir__()
        ), f"create_ops_listC{precision} not found in algorithms submodule"

    def test_runtime_info_binding_type(self):
        """Test if runtime_info contains the correct binding_type."""
        # Get runtime info
        info = self.nb_module.runtime_info()

        # Check binding type
        assert "binding_type" in info
        assert info["binding_type"] == "nanobind"

        # Check for CPU instruction set flags
        for key in ["AVX", "AVX2", "AVX512F"]:
            assert key in info

    def test_compile_info(self):
        """Test if compile_info contains expected keys."""
        # Get compile info
        info = self.nb_module.compile_info()

        # Check for expected keys
        for key in ["cpu.arch", "compiler.name", "compiler.version", "AVX2", "AVX512F"]:
            assert key in info


@pytest.mark.skipif(
    device_name == "lightning.tensor", reason="lightning.tensor does not support aligned arrays"
)
class TestAlignedArrayNB:
    """Tests for allocate_aligned_array function in nanobind-based modules."""

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.float32, np.float64])
    def test_allocate_aligned_array_basic(self, current_module, dtype):
        """Test basic functionality of allocate_aligned_array."""
        size = 1024
        arr = current_module.allocate_aligned_array(size, np.dtype(dtype), False)

        # Check array properties
        assert arr.size == size
        assert arr.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.float32, np.float64])
    def test_allocate_aligned_array_zero_init(self, current_module, dtype):
        """Test zero initialization of allocate_aligned_array."""
        size = 1024
        arr = current_module.allocate_aligned_array(size, np.dtype(dtype), True)

        # Check array is zero-initialized
        assert arr.size == size
        assert arr.dtype == dtype
        assert np.all(arr == 0)

    def test_allocate_aligned_array_invalid_dtype(self, current_module):
        """Test allocate_aligned_array with invalid dtype raises an error."""
        size = 1024
        with pytest.raises((TypeError, RuntimeError)):
            current_module.allocate_aligned_array(size, np.dtype(np.int32), False)

    def test_allocate_aligned_array_memory_alignment(self, current_module):
        """Test memory alignment of allocated array."""
        size = 1024
        arr = current_module.allocate_aligned_array(size, np.dtype(np.complex128), False)

        # Check array memory alignment
        # The pointer address should be divisible by 32 (for AVX2) or 64 (for AVX512)
        ptr_addr = arr.__array_interface__["data"][0]
        assert ptr_addr % 32 == 0, "Memory not aligned to 32-byte boundary"
