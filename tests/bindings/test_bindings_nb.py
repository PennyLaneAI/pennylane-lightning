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
from conftest import backend


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

    # Define the corresponding pybind module for comparison
    pb_module_name = f"pennylane_lightning.lightning_{backend}_ops"

    @pytest.fixture(autouse=True)
    def setup_module_attributes(self, current_nanobind_module):
        """Set up module attributes for all tests."""
        self.nb_module = current_nanobind_module
        self.nb_module_attr = get_module_attributes(self.nb_module)

        try:
            self.pb_module = importlib.import_module(self.pb_module_name)
            self.pb_module_attr = get_module_attributes(self.pb_module)
            self.pb_module_importable = True
        except ImportError:
            self.pb_module_importable = False

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
            assert (
                f"StateVectorC{precision}" in self.nb_module_attr["classes"]
            ), f"StateVectorC{precision} not found in module"

    def test_exception_class_exists(self):
        """Test if LightningException class exists in the module."""
        assert (
            "LightningException" in self.nb_module_attr["classes"]
        ), f"LightningException not found in module"

    @pytest.mark.xfail(reason="Expected to fail while we don't have backend-specific bindings.")
    def test_api_parity_with_pybind(self):
        """Test that nanobind modules have the same API as pybind modules."""
        if not self.pb_module_importable:
            pytest.skip(f"Pybind module {self.pb_module_name} not available")

        # Check that all classes in pybind module exist in nanobind module
        for cls in self.pb_module_attr["classes"]:
            assert (
                cls in self.nb_module_attr["classes"]
            ), f"Class {cls} exists in pybind module but not in nanobind module"

            # Get class objects
            pb_class = getattr(self.pb_module, cls)
            nb_class = getattr(self.nb_module, cls)

            # Check methods of the class
            pb_methods = [
                name
                for name in dir(pb_class)
                if not name.startswith("__") and callable(getattr(pb_class, name))
            ]

            for method_name in pb_methods:
                # Skip private methods and special methods
                if method_name.startswith("_"):
                    continue

                assert hasattr(
                    nb_class, method_name
                ), f"Method {cls}.{method_name} exists in pybind module but not in nanobind module"

                # Get method objects
                pb_method = getattr(pb_class, method_name)
                nb_method = getattr(nb_class, method_name)

                # Check if both are callable
                assert callable(pb_method), f"{cls}.{method_name} in pybind module is not callable"
                assert callable(
                    nb_method
                ), f"{cls}.{method_name} in nanobind module is not callable"

                # Try to get signatures if possible
                try:
                    pb_sig = inspect.signature(pb_method)
                    nb_sig = inspect.signature(nb_method)

                    # Compare parameter count
                    pb_param_count = len(pb_sig.parameters)
                    nb_param_count = len(nb_sig.parameters)

                    assert pb_param_count == nb_param_count, (
                        f"Method {cls}.{method_name} has {pb_param_count} parameters in pybind module "
                        f"but {nb_param_count} parameters in nanobind module"
                    )

                    # Compare parameter names and kinds
                    for pb_param_name, pb_param in pb_sig.parameters.items():
                        assert pb_param_name in nb_sig.parameters, (
                            f"Parameter '{pb_param_name}' of {cls}.{method_name} exists in pybind module "
                            f"but not in nanobind module"
                        )

                        nb_param = nb_sig.parameters[pb_param_name]
                        assert pb_param.kind == nb_param.kind, (
                            f"Parameter '{pb_param_name}' of {cls}.{method_name} has kind {pb_param.kind} in pybind module "
                            f"but kind {nb_param.kind} in nanobind module"
                        )
                except (ValueError, TypeError):
                    # Skip signature comparison if it's not possible to get signatures
                    pass

        # Check that all functions in pybind module exist in nanobind module
        for func in self.pb_module_attr["functions"]:
            assert (
                func in self.nb_module_attr["functions"]
            ), f"Function {func} exists in pybind module but not in nanobind module"

            # Get function objects
            pb_func = getattr(self.pb_module, func)
            nb_func = getattr(self.nb_module, func)

            # Check if both are callable
            assert callable(pb_func), f"{func} in pybind module is not callable"
            assert callable(nb_func), f"{func} in nanobind module is not callable"

            # Try to get signatures if possible
            try:
                pb_sig = inspect.signature(pb_func)
                nb_sig = inspect.signature(nb_func)

                # Compare parameter count
                pb_param_count = len(pb_sig.parameters)
                nb_param_count = len(nb_sig.parameters)

                assert pb_param_count == nb_param_count, (
                    f"Function {func} has {pb_param_count} parameters in pybind module "
                    f"but {nb_param_count} parameters in nanobind module"
                )

                # Compare parameter names and kinds
                for pb_param_name, pb_param in pb_sig.parameters.items():
                    assert pb_param_name in nb_sig.parameters, (
                        f"Parameter '{pb_param_name}' of function {func} exists in pybind module "
                        f"but not in nanobind module"
                    )

                    nb_param = nb_sig.parameters[pb_param_name]
                    assert pb_param.kind == nb_param.kind, (
                        f"Parameter '{pb_param_name}' of function {func} has kind {pb_param.kind} in pybind module "
                        f"but kind {nb_param.kind} in nanobind module"
                    )

                # Check return type annotation if available
                if (
                    pb_sig.return_annotation is not inspect.Signature.empty
                    and nb_sig.return_annotation is not inspect.Signature.empty
                ):
                    assert pb_sig.return_annotation == nb_sig.return_annotation, (
                        f"Function {func} has return type {pb_sig.return_annotation} in pybind module "
                        f"but {nb_sig.return_annotation} in nanobind module"
                    )
            except (ValueError, TypeError):
                # Skip signature comparison if it's not possible to get signatures
                pass

            # Test return type for array-returning functions
            if func in ["probs", "expval", "var", "generate_samples"]:
                try:
                    # Create minimal test data to check return types
                    if "StateVectorC64" in self.nb_module_attr["classes"]:
                        sv_class = getattr(self.nb_module, "StateVectorC64")
                        sv = sv_class(1)  # Create a 1-qubit state vector

                        if func == "generate_samples":
                            # For generate_samples, we need a Measurements object
                            if "MeasurementsC64" in self.nb_module_attr["classes"]:
                                meas_class = getattr(self.nb_module, "MeasurementsC64")
                                meas = meas_class(sv)
                                nb_result = getattr(meas, func)(10)  # Generate 10 samples
                                assert isinstance(
                                    nb_result, np.ndarray
                                ), f"{func} in nanobind module should return numpy.ndarray"
                        elif func in ["probs", "expval", "var"]:
                            # For measurement functions, we need a Measurements object
                            if "MeasurementsC64" in self.nb_module_attr["classes"]:
                                meas_class = getattr(self.nb_module, "MeasurementsC64")
                                meas = meas_class(sv)

                                if func == "probs":
                                    nb_result = meas.probs([0])
                                    assert isinstance(
                                        nb_result, np.ndarray
                                    ), f"{func} in nanobind module should return numpy.ndarray"
                except Exception:
                    # Skip if we can't easily test the return type
                    pass

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_observables_submodule_exists(self, precision):
        """Test that the observables submodule exists and contains expected classes."""
        # Check if observables submodule exists
        assert hasattr(self.nb_module, "observables"), "Module does not have observables submodule"

        # Check for NamedObs classes
        assert (
            f"NamedObsC{precision}" in self.nb_module.observables.__dir__()
        ), f"NamedObsC{precision} not found in observables submodule"

        # Check for HermitianObs classes
        assert (
            f"HermitianObsC{precision}" in self.nb_module.observables.__dir__()
        ), f"HermitianObsC{precision} not found in observables submodule"

        # Check for TensorProdObs classes
        assert (
            f"TensorProdObsC{precision}" in self.nb_module.observables.__dir__()
        ), f"TensorProdObsC{precision} not found in observables submodule"

        # Check for Hamiltonian classes
        assert (
            f"HamiltonianC{precision}" in self.nb_module.observables.__dir__()
        ), f"HamiltonianC{precision} not found in observables submodule"

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_algorithms_submodule_exists(self, precision):
        """Test that the algorithms submodule exists and contains expected classes."""
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


class TestAlignedArrayNB:
    """Tests for allocate_aligned_array function in nanobind-based modules."""

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.float32, np.float64])
    def test_allocate_aligned_array_basic(self, current_nanobind_module, dtype):
        """Test basic functionality of allocate_aligned_array."""
        size = 1024
        arr = current_nanobind_module.allocate_aligned_array(size, np.dtype(dtype), False)

        # Check array properties
        assert arr.size == size
        assert arr.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.float32, np.float64])
    def test_allocate_aligned_array_zero_init(self, current_nanobind_module, dtype):
        """Test zero initialization of allocate_aligned_array."""
        size = 1024
        arr = current_nanobind_module.allocate_aligned_array(size, np.dtype(dtype), True)

        # Check array is zero-initialized
        assert arr.size == size
        assert arr.dtype == dtype
        assert np.all(arr == 0)

    def test_allocate_aligned_array_invalid_dtype(self, current_nanobind_module):
        """Test allocate_aligned_array with invalid dtype raises an error."""
        size = 1024
        with pytest.raises((TypeError, RuntimeError)):
            current_nanobind_module.allocate_aligned_array(size, np.dtype(np.int32), False)

    def test_allocate_aligned_array_memory_alignment(self, current_nanobind_module):
        """Test memory alignment of allocated array."""
        size = 1024
        arr = current_nanobind_module.allocate_aligned_array(size, np.dtype(np.complex128), False)

        # Check array memory alignment
        # The pointer address should be divisible by 32 (for AVX2) or 64 (for AVX512)
        ptr_addr = arr.__array_interface__["data"][0]
        assert ptr_addr % 32 == 0, "Memory not aligned to 32-byte boundary"
