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


class TestNanobindBindings:
    """Tests for nanobind-based bindings."""

    # List of nanobind modules to test
    nb_modules = [
        "pennylane_lightning.lightning_qubit_nb",
        "pennylane_lightning.lightning_kokkos_nb",
        "pennylane_lightning.lightning_gpu_nb",
    ]

    # List of corresponding pybind modules for comparison
    pb_modules = [
        "pennylane_lightning.lightning_qubit",
        "pennylane_lightning.lightning_kokkos",
        "pennylane_lightning.lightning_gpu",
    ]
    # Expected main class for each module
    expected_classes = {
        "pennylane_lightning.lightning_qubit_nb": "LightningQubit",
        "pennylane_lightning.lightning_kokkos_nb": "LightningKokkos",
        "pennylane_lightning.lightning_gpu_nb": "LightningGPU",
    }
    # Store module attributes for each module
    module_attributes = {}

    @pytest.fixture(autouse=True, scope="class")
    def setup_module_attributes(self):
        """Set up module attributes for all tests."""
        for module_name in self.nb_modules:
            self.module_attributes[module_name] = get_module_attributes(module_name)

        for module_name in self.pb_modules:
            self.module_attributes[module_name] = get_module_attributes(module_name)

    def _skip_if_module_not_importable(self, module_name):
        """Skip test if module is not importable.

        This helper method checks if a module is importable and skips the current test
        if it's not. It returns the module attributes for use in the test.

        Each test that calls this will independently be skipped if the module is not available.
        """
        module_attr = self.module_attributes[module_name]
        if not module_attr["importable"]:
            pytest.skip(f"Module {module_name} not available: {module_attr.get('error')}")
        return module_attr

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_module_importable(self, module_name):
        """Test if module can be imported."""
        module_attr = self._skip_if_module_not_importable(module_name)
        assert module_attr["importable"]
        assert len(module_attr["classes"]) > 0 or len(module_attr["functions"]) > 0

    # Expected to fail for now.
    @pytest.mark.xfail(reason="Expected to fail while we don;t have backend-specific main classes.")
    @pytest.mark.parametrize("module_name", nb_modules)
    def test_module_has_expected_main_class(self, module_name):
        """Test if module has expected main class."""
        module_attr = self._skip_if_module_not_importable(module_name)

        # Check for the main device class
        expected_class = self.expected_classes.get(module_name)
        assert (
            expected_class in module_attr["classes"]
        ), f"{expected_class} not found in {module_name}"

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_module_has_expected_info_dicts(self, module_name):
        """Test if module has expected info dicts."""
        module_attr = self._skip_if_module_not_importable(module_name)

        # Check for backend_info if available
        assert (
            "compile_info" in module_attr["functions"]
        ), f"compile_info not found in {module_name}"
        assert (
            "runtime_info" in module_attr["functions"]
        ), f"runtime_info not found in {module_name}"

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_statevector_classes_exists(self, module_name):
        """Test if StateVectorC classes exists in the module."""
        module_attr = self._skip_if_module_not_importable(module_name)

        for precision in ["64", "128"]:
            assert (
                f"StateVectorC{precision}" in module_attr["classes"]
            ), f"StateVectorC{precision} not found in {module_name}"

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_exception_class_exists(self, module_name):
        """Test if LightningException class exists in the module."""
        module_attr = self._skip_if_module_not_importable(module_name)

        assert (
            "LightningException" in module_attr["classes"]
        ), f"LightningException not found in {module_name}"

    # Expected to fail for now.
    @pytest.mark.xfail(reason="Expected to fail while we don;t have backend-specific main classes.")
    @pytest.mark.parametrize("module_name, pybind_module_name", zip(nb_modules, pb_modules))
    def test_api_parity_with_pybind(self, module_name, pybind_module_name):
        """Test that nanobind modules have the same API as pybind modules."""
        nb_attr = self._skip_if_module_not_importable(module_name)
        pb_attr = self.module_attributes[pybind_module_name]

        if not pb_attr["importable"]:
            pytest.skip(f"Pybind module {pybind_module_name} not available: {pb_attr.get('error')}")

        # Check that all classes in pybind module exist in nanobind module
        for cls in pb_attr["classes"]:
            assert (
                cls in nb_attr["classes"]
            ), f"Class {cls} exists in {pybind_module_name} but not in {module_name}"

        # Check that all functions in pybind module exist in nanobind module
        for func in pb_attr["functions"]:
            assert (
                func in nb_attr["functions"]
            ), f"Function {func} exists in {pybind_module_name} but not in {module_name}"
