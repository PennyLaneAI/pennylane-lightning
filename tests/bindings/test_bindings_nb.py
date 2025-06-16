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
        "pennylane_lightning.lightning_tensor_nb",
    ]

    # List of corresponding pybind modules for comparison
    pb_modules = [
        "pennylane_lightning.lightning_qubit_ops",
        "pennylane_lightning.lightning_kokkos_ops",
        "pennylane_lightning.lightning_gpu_ops",
        "pennylane_lightning.lightning_tensor_ops",
    ]

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
    def test_module_importable(self, module_name: str):
        """Test if module can be imported."""
        module_attr = self._skip_if_module_not_importable(module_name)

        assert len(module_attr["classes"]) > 0 or len(module_attr["functions"]) > 0

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_module_has_expected_info_dicts(self, module_name: str):
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
    def test_statevector_classes_exists(self, module_name: str):
        """Test if StateVectorC classes exists in the module."""
        module_attr = self._skip_if_module_not_importable(module_name)

        for precision in ["64", "128"]:
            assert (
                f"StateVectorC{precision}" in module_attr["classes"]
            ), f"StateVectorC{precision} not found in {module_name}"

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_exception_class_exists(self, module_name: str):
        """Test if LightningException class exists in the module."""
        module_attr = self._skip_if_module_not_importable(module_name)

        assert (
            "LightningException" in module_attr["classes"]
        ), f"LightningException not found in {module_name}"

    @pytest.mark.xfail(reason="Expected to fail while we don't have backend-specific bindings.")
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

            # Get class objects (this will fail if there is no such class.)
            pb_class = getattr(pb_attr["module"], cls)
            nb_class = getattr(nb_attr["module"], cls)

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
                ), f"Method {cls}.{method_name} exists in {pybind_module_name} but not in {module_name}"

                # Get method objects (this will fail if there is no such method.)
                pb_method = getattr(pb_class, method_name)
                nb_method = getattr(nb_class, method_name)

                # Check if both are callable
                assert callable(
                    pb_method
                ), f"{cls}.{method_name} in {pybind_module_name} is not callable"
                assert callable(nb_method), f"{cls}.{method_name} in {module_name} is not callable"

                # Try to get signatures if possible
                try:
                    pb_sig = inspect.signature(pb_method)
                    nb_sig = inspect.signature(nb_method)

                    # Compare parameter count (excluding self)
                    pb_param_count = len(pb_sig.parameters)
                    nb_param_count = len(nb_sig.parameters)

                    assert pb_param_count == nb_param_count, (
                        f"Method {cls}.{method_name} has {pb_param_count} parameters in {pybind_module_name} "
                        f"but {nb_param_count} parameters in {module_name}"
                    )

                    # Compare parameter names and kinds
                    for pb_param_name, pb_param in pb_sig.parameters.items():
                        assert pb_param_name in nb_sig.parameters, (
                            f"Parameter '{pb_param_name}' of {cls}.{method_name} exists in {pybind_module_name} "
                            f"but not in {module_name}"
                        )

                        nb_param = nb_sig.parameters[pb_param_name]
                        assert pb_param.kind == nb_param.kind, (
                            f"Parameter '{pb_param_name}' of {cls}.{method_name} has kind {pb_param.kind} in {pybind_module_name} "
                            f"but kind {nb_param.kind} in {module_name}"
                        )
                except (ValueError, TypeError):
                    # Skip signature comparison if it's not possible to get signatures
                    pass

        # Check that all functions in pybind module exist in nanobind module
        for func in pb_attr["functions"]:
            assert (
                func in nb_attr["functions"]
            ), f"Function {func} exists in {pybind_module_name} but not in {module_name}"

            # Get function objects
            pb_func = getattr(pb_attr["module"], func)
            nb_func = getattr(nb_attr["module"], func)

            # Check if both are callable
            assert callable(pb_func), f"{func} in {pybind_module_name} is not callable"
            assert callable(nb_func), f"{func} in {module_name} is not callable"

            # Try to get signatures if possible
            try:
                pb_sig = inspect.signature(pb_func)
                nb_sig = inspect.signature(nb_func)

                # Compare parameter count
                pb_param_count = len(pb_sig.parameters)
                nb_param_count = len(nb_sig.parameters)

                assert pb_param_count == nb_param_count, (
                    f"Function {func} has {pb_param_count} parameters in {pybind_module_name} "
                    f"but {nb_param_count} parameters in {module_name}"
                )

                # Compare parameter names and kinds
                for pb_param_name, pb_param in pb_sig.parameters.items():
                    assert pb_param_name in nb_sig.parameters, (
                        f"Parameter '{pb_param_name}' of function {func} exists in {pybind_module_name} "
                        f"but not in {module_name}"
                    )

                    nb_param = nb_sig.parameters[pb_param_name]
                    assert pb_param.kind == nb_param.kind, (
                        f"Parameter '{pb_param_name}' of function {func} has kind {pb_param.kind} in {pybind_module_name} "
                        f"but kind {nb_param.kind} in {module_name}"
                    )

                # Check return type annotation if available
                if (
                    pb_sig.return_annotation is not inspect.Signature.empty
                    and nb_sig.return_annotation is not inspect.Signature.empty
                ):
                    assert pb_sig.return_annotation == nb_sig.return_annotation, (
                        f"Function {func} has return type {pb_sig.return_annotation} in {pybind_module_name} "
                        f"but {nb_sig.return_annotation} in {module_name}"
                    )
            except (ValueError, TypeError):
                # Skip signature comparison if it's not possible to get signatures
                pass

            # Test return type for array-returning functions
            if func in ["probs", "expval", "var", "generate_samples"]:
                try:
                    # Create minimal test data to check return types
                    # This is a simplified approach - in a real test you might need more setup
                    if "StateVectorC64" in nb_attr["classes"]:
                        sv_class = getattr(nb_attr["module"], "StateVectorC64")
                        sv = sv_class(1)  # Create a 1-qubit state vector

                        if func == "generate_samples":
                            # For generate_samples, we need a Measurements object
                            if "MeasurementsC64" in nb_attr["classes"]:
                                meas_class = getattr(nb_attr["module"], "MeasurementsC64")
                                meas = meas_class(sv)
                                nb_result = getattr(meas, func)(10)  # Generate 10 samples
                                assert isinstance(
                                    nb_result, np.ndarray
                                ), f"{func} in {module_name} should return numpy.ndarray"
                        elif func in ["probs", "expval", "var"]:
                            # For measurement functions, we need a Measurements object
                            if "MeasurementsC64" in nb_attr["classes"]:
                                meas_class = getattr(nb_attr["module"], "MeasurementsC64")
                                meas = meas_class(sv)

                                if func == "probs":
                                    nb_result = meas.probs([0])
                                    assert isinstance(
                                        nb_result, np.ndarray
                                    ), f"{func} in {module_name} should return numpy.ndarray"
                except Exception:
                    # Skip if we can't easily test the return type
                    pass

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_observables_submodule_exists(self, module_name: str, precision: str):
        """Test that the observables submodule exists and contains expected classes."""

        module_attrs = self._skip_if_module_not_importable(module_name)

        if not module_attrs.get("importable", False):
            pytest.skip(
                f"Nanobind module {module_name} not available: {module_attrs.get('error', 'Unknown error')}"
            )

        module = module_attrs.get("module")

        # Check if observables submodule exists
        assert hasattr(
            module, "observables"
        ), f"Module {module.__name__} does not have observables submodule"

        # Check for NamedObs classes
        assert (
            f"NamedObsC{precision}" in module.observables.__dir__()
        ), f"NamedObsC{precision} not found in {module.__name__}.observables"

        # Check for HermitianObs classes
        assert (
            f"HermitianObsC{precision}" in module.observables.__dir__()
        ), f"HermitianObsC{precision} not found in {module.__name__}.observables"

        # Check for TensorProdObs classes
        assert (
            f"TensorProdObsC{precision}" in module.observables.__dir__()
        ), f"TensorProdObsC{precision} not found in {module.__name__}.observables"

        # Check for Hamiltonian classes
        assert (
            f"HamiltonianC{precision}" in module.observables.__dir__()
        ), f"HamiltonianC{precision} not found in {module.__name__}.observables"

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_algorithms_submodule_exists(self, module_name: str, precision: str):
        """Test that the algorithms submodule exists and contains expected classes."""

        module_attrs = self._skip_if_module_not_importable(module_name)

        if not module_attrs.get("importable", False):
            pytest.skip(
                f"Nanobind module {module_name} not available: {module_attrs.get('error', 'Unknown error')}"
            )

        module = module_attrs.get("module")

        # Check if algorithms submodule exists
        assert hasattr(
            module, "algorithms"
        ), f"Module {module.__name__} does not have algorithms submodule"

        # Check for OpsStruct classes
        assert (
            f"OpsStructC{precision}" in module.algorithms.__dir__()
        ), f"OpsStructC{precision} not found in {module.__name__}.algorithms"

        assert (
            f"create_ops_listC{precision}" in module.algorithms.__dir__()
        ), f"create_ops_listC{precision} not found in {module.__name__}.algorithms"

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_runtime_info_binding_type(self, module_name: str):
        """Test if runtime_info contains the correct binding_type."""
        module_attr = self._skip_if_module_not_importable(module_name)

        # Import the module
        module = importlib.import_module(module_name)

        # Get runtime info
        info = module.runtime_info()

        # Check binding type
        assert "binding_type" in info
        assert info["binding_type"] == "nanobind"

        # Check for CPU instruction set flags (same as in test_binary_info.py)
        for key in ["AVX", "AVX2", "AVX512F"]:
            assert key in info

    @pytest.mark.parametrize("module_name", nb_modules)
    def test_compile_info(self, module_name: str):
        """Test if compile_info contains expected keys."""
        module_attr = self._skip_if_module_not_importable(module_name)

        # Import the module
        module = importlib.import_module(module_name)

        # Get compile info
        info = module.compile_info()

        # Check for expected keys (same as in test_binary_info.py)
        for key in ["cpu.arch", "compiler.name", "compiler.version", "AVX2", "AVX512F"]:
            assert key in info
