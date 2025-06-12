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
"""Tests for StateVector classes in nanobind-based modules."""

import importlib

import numpy as np
import pytest


class TestStateVectorNB:
    """Tests for StateVectorC64 and StateVectorC128 classes in nanobind-based modules."""

    # List of modules to test
    modules = [
        "pennylane_lightning.lightning_qubit_nb",
        "pennylane_lightning.lightning_kokkos_nb",
        "pennylane_lightning.lightning_gpu_nb",
    ]

    # Store module attributes for each module
    module_attributes = {}

    @pytest.fixture(autouse=True, scope="class")
    def setup_module_attributes(self):
        """Set up module attributes for all tests."""
        for module_name in self.modules:
            try:
                self.module_attributes[module_name] = {
                    "importable": True,
                    "module": importlib.import_module(module_name),
                }
            except ImportError as e:
                self.module_attributes[module_name] = {"importable": False, "error": str(e)}

    def _skip_if_module_not_importable(self, module_name):
        """Skip test if module is not importable."""
        module_attr = self.module_attributes[module_name]
        if not module_attr["importable"]:
            pytest.skip(f"Module {module_name} not available: {module_attr.get('error')}")
        return module_attr["module"]

    @pytest.fixture(params=["64", "128"])
    def precision(self, request):
        """Fixture to parametrize tests over different precision types."""
        return request.param

    @pytest.fixture
    def get_statevector_class(self, precision):
        """Fixture to get the appropriate StateVector class based on precision."""

        def _get_class(module):
            return getattr(module, f"StateVectorC{precision}")

        return _get_class

    @pytest.mark.parametrize("module_name", modules)
    def test_statevector_initialization(self, module_name, get_statevector_class):
        """Test initialization of StateVectorC64/128 classes."""
        module = self._skip_if_module_not_importable(module_name)
        StateVectorClass = get_statevector_class(module)

        # Test initialization with number of qubits and updateData
        num_qubits = 3
        # Create a numpy array representing |0> state with appropriate size
        state_data = np.zeros(2**num_qubits, dtype=np.complex128)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

        # Check that the size is correct
        assert sv.size() == 2**num_qubits

    @pytest.mark.parametrize("module_name", modules)
    def test_statevector_methods(self, module_name, get_statevector_class):
        """Test methods of StateVectorC64/128 classes."""
        module = self._skip_if_module_not_importable(module_name)
        StateVectorClass = get_statevector_class(module)

        num_qubits = 2
        # Create a numpy array representing |0> state with appropriate size
        state_data = np.zeros(2**num_qubits, dtype=np.complex128)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

        # Test size
        assert sv.size() == 2**num_qubits

    @pytest.mark.parametrize("module_name", modules)
    def test_statevector_gate_operations(self, module_name, get_statevector_class):
        """Test gate operations on StateVectorC64/128 classes."""
        module = self._skip_if_module_not_importable(module_name)
        StateVectorClass = get_statevector_class(module)

        num_qubits = 2
        # Create a numpy array representing |0> state with appropriate size
        state_data = np.zeros(2**num_qubits, dtype=np.complex128)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

        # Apply various gates - we can't check the state directly yet,
        # but we can verify the operations don't raise exceptions
        sv.PauliX([0], False, [])
        sv.Hadamard([0], True, [])
        sv.Hadamard([1], False, [])
        sv.CNOT([0, 1], True, [])

        # If we get here without exceptions, the test passes
        assert True

    @pytest.mark.parametrize("module_name", modules)
    def test_statevector_parametric_gates(self, module_name, get_statevector_class):
        """Test parametric gates on StateVectorC64/128 classes."""
        module = self._skip_if_module_not_importable(module_name)
        StateVectorClass = get_statevector_class(module)

        num_qubits = 1
        # Create a numpy array representing |0> state with appropriate size
        state_data = np.zeros(2**num_qubits, dtype=np.complex128)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

        # Apply parametric gates - we can't check the state directly yet,
        # but we can verify the operations don't raise exceptions
        sv.RX([0], False, [np.pi / 2])
        sv.RY([0], False, [np.pi])
        sv.RZ([0], False, [np.pi / 2])

        # If we get here without exceptions, the test passes
        assert True

    @pytest.mark.parametrize("module_name", modules)
    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_statevector_matrix_application(self, module_name, precision, get_statevector_class):
        """Test matrix application on StateVectorC64/128 classes for lightning.qubit_nb."""
        if module_name != "pennylane_lightning.lightning_qubit_nb":
            pytest.skip(f"Module {module_name} not supported")
        module = self._skip_if_module_not_importable(module_name)
        StateVectorClass = get_statevector_class(module)

        num_qubits = 1
        # Create a numpy array representing |0> state with appropriate size
        dtype = np.complex128 if precision == "128" else np.complex64
        state_data = np.zeros(2**num_qubits, dtype=dtype)
        state_data[0] = 1.0

        # Create state vector with just number of qubits
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

        # Apply a matrix to the state vector
        matrix = np.array([[0, 1], [1, 0]], dtype=dtype)  # X gate
        sv.applyMatrix(matrix, [0], False)

        # Get the result
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Expected: |1⟩ state
        expected = np.zeros(2**num_qubits, dtype=dtype)
        expected[1] = 1.0

        # Assert the result matches the expected state
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_statevector_update_data(self, precision, get_statevector_class):
        """Test updateData method on StateVectorC64/128 classes for lightning.qubit_nb."""
        module_name = "pennylane_lightning.lightning_qubit_nb"

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            pytest.skip(f"Module {module_name} not available")

        StateVectorClass = get_statevector_class(module)

        num_qubits = 2
        # Create a numpy array with custom data
        dtype = np.complex128 if precision == "128" else np.complex64
        state_data = np.zeros(2**num_qubits, dtype=dtype)
        state_data[1] = 1.0  # |01⟩ state

        # Create state vector with just number of qubits
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

        # Get the result
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Assert the result matches the input data
        np.testing.assert_allclose(result, state_data)

    @pytest.mark.parametrize("module_name", modules)
    def test_statevector_initialization_only(
        self, module_name, precision, get_statevector_class, capfd
    ):
        """Test that StateVector is correctly initialized without any operations."""
        module = self._skip_if_module_not_importable(module_name)
        StateVectorClass = get_statevector_class(module)

        # Test initialization with number of qubits
        num_qubits = 3
        sv = StateVectorClass(num_qubits)

        # Check that the size is correct
        assert sv.size() == 2**num_qubits

        # Get the state and verify it's |000⟩
        dtype = np.complex64 if precision == "64" else np.complex128
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Expected: |000⟩ state (first element is 1.0, rest are 0.0)
        expected = np.zeros(2**num_qubits, dtype=dtype)
        expected[0] = 1.0

        # Assert the result matches the expected state
        np.testing.assert_allclose(result, expected, atol=1e-6)
