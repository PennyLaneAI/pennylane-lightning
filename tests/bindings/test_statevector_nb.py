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

import numpy as np
import pytest
from conftest import backend

if backend != "qubit":
    pytest.skip("Skipping tests for binaries other than lightning_qubit .", allow_module_level=True)


class TestStateVectorNB:
    """Tests for StateVectorC64 and StateVectorC128 classes in nanobind-based modules."""

    @pytest.fixture
    def get_statevector_class(self):
        """Get StateVectorC64/128 class from module based on precision."""

        def _get_class(module, precision="64"):
            class_name = f"StateVectorC{precision}"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in module")

        return _get_class

    def test_statevector_initialization(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test initialization of StateVectorC64/128 classes."""
        module = current_nanobind_module

        # Get the appropriate StateVector class based on precision
        StateVectorClass = get_statevector_class(module, precision)

        # Test initialization with number of qubits
        num_qubits = 3

        # Initialize with number of qubits - this should already be in |0⟩ state
        sv = StateVectorClass(num_qubits)

        # Check that the size is correct
        assert sv.size() == 2**num_qubits

        # Verify the state is |0⟩
        dtype = np.complex128 if precision == "128" else np.complex64
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Expected: |0⟩ state
        expected = np.zeros(2**num_qubits, dtype=dtype)
        expected[0] = 1.0

        np.testing.assert_allclose(result, expected)

    def test_statevector_gate_operations(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test gate operations on StateVectorC64/128 classes."""
        module = current_nanobind_module

        StateVectorClass = get_statevector_class(module, precision)

        num_qubits = 2
        # Initialize with number of qubits - already in |0⟩ state
        sv = StateVectorClass(num_qubits)

        # Apply various gates - we can't check the state directly yet,
        # but we can verify the operations don't raise exceptions
        sv.PauliX([0], False, [])
        sv.Hadamard([0], True, [])
        sv.Hadamard([1], False, [])
        sv.CNOT([0, 1], True, [])

        # If we get here without exceptions, the test passes
        assert True

    def test_statevector_parametric_gates(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test parametric gates on StateVectorC64/128 classes."""
        module = current_nanobind_module

        StateVectorClass = get_statevector_class(module, precision)

        num_qubits = 1
        # Initialize with number of qubits - already in |0⟩ state
        sv = StateVectorClass(num_qubits)

        # Apply parametric gates - we can't check the state directly yet,
        # but we can verify the operations don't raise exceptions
        sv.RX([0], False, [np.pi / 2])
        sv.RY([0], False, [np.pi])
        sv.RZ([0], False, [np.pi / 2])

        # If we get here without exceptions, the test passes
        assert True

    def test_statevector_matrix_application(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test matrix application on StateVectorC64/128 classes."""
        module = current_nanobind_module

        StateVectorClass = get_statevector_class(module, precision)

        num_qubits = 1
        # Initialize with number of qubits - already in |0⟩ state
        sv = StateVectorClass(num_qubits)

        # Get the result
        dtype = np.complex128 if precision == "128" else np.complex64
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

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

    def test_statevector_reset(self, current_nanobind_module, precision, get_statevector_class):
        """Test resetStateVector method on StateVectorC64/128 classes."""
        module = current_nanobind_module

        StateVectorClass = get_statevector_class(module, precision)

        num_qubits = 2
        # Initialize with number of qubits - already in |0⟩ state
        sv = StateVectorClass(num_qubits)

        # Apply some operations to change the state
        sv.PauliX([0], False, [])

        # Reset the state vector
        sv.resetStateVector()

        # Get the result
        dtype = np.complex128 if precision == "128" else np.complex64
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Expected: |00⟩ state
        expected = np.zeros(2**num_qubits, dtype=dtype)
        expected[0] = 1.0

        assert np.allclose(result, expected)

    def test_statevector_set_basis_state(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test setBasisState method on StateVectorC64/128 classes."""
        module = current_nanobind_module

        StateVectorClass = get_statevector_class(module, precision)

        num_qubits = 3
        dtype = np.complex128 if precision == "128" else np.complex64
        # Initialize with number of qubits
        sv = StateVectorClass(num_qubits)

        # Set to basis state |101⟩
        sv.setBasisState([1, 0, 1], [0, 1, 2])

        # Get the result
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Expected: |101⟩ state (index 5)
        expected = np.zeros(2**num_qubits, dtype=dtype)
        expected[5] = 1.0

        assert np.allclose(result, expected)

        # Test with different wire ordering
        sv.resetStateVector()
        # Set to basis state |110⟩ using different wire order
        sv.setBasisState([1, 1, 0], [2, 1, 0])

        # Get the result
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Expected: |011⟩ state (index 3)
        expected = np.zeros(2**num_qubits, dtype=dtype)
        expected[3] = 1.0

        assert np.allclose(result, expected)

    def test_statevector_set_state_vector(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test setStateVector method on StateVectorC64/128 classes."""
        module = current_nanobind_module

        StateVectorClass = get_statevector_class(module, precision)

        num_qubits = 2
        # Initialize with number of qubits
        sv = StateVectorClass(num_qubits)

        # Create a superposition state for a single qubit
        dtype = np.complex128 if precision == "128" else np.complex64
        superposition = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=dtype)

        # Set qubit 0 to the superposition state
        sv.setStateVector(superposition, [0])

        # Get the result
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Expected: (|00⟩ + |10⟩)/√2
        expected = np.zeros(2**num_qubits, dtype=dtype)
        expected[0] = 1.0 / np.sqrt(2)
        expected[2] = 1.0 / np.sqrt(2)

        assert np.allclose(result, expected)

        # Reset and try with qubit 1
        sv.resetStateVector()

        # Set qubit 1 to the superposition state
        sv.setStateVector(superposition, [1])

        # Get the result
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Expected: (|00⟩ + |01⟩)/√2
        expected = np.zeros(2**num_qubits, dtype=dtype)
        expected[0] = 1.0 / np.sqrt(2)
        expected[1] = 1.0 / np.sqrt(2)

        assert np.allclose(result, expected)

    def test_statevector_len_and_size(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test __len__ and size methods on StateVectorC64/128 classes."""
        module = current_nanobind_module

        StateVectorClass = get_statevector_class(module, precision)

        # Test with different numbers of qubits
        for num_qubits in range(1, 5):
            sv = StateVectorClass(num_qubits)

            # Check that len() and size() return the correct value
            expected_size = 2**num_qubits
            assert len(sv) == expected_size
            assert sv.size() == expected_size

    def test_statevector_update_data_nontrivial(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test updateData method with non-trivial state vectors."""
        module = current_nanobind_module
        # Skip if updateData is not available
        if not hasattr(module, "StateVectorC" + precision):
            pytest.skip(f"Class StateVectorC{precision} not available in module")

        StateVectorClass = get_statevector_class(module, precision)

        num_qubits = 2
        # Initialize with number of qubits
        sv = StateVectorClass(num_qubits)

        # Create a non-trivial state vector
        dtype = np.complex128 if precision == "128" else np.complex64
        state_data = np.array([1, 2, 3, 4], dtype=dtype) / np.sqrt(30)
        sv.updateData(state_data)

        # Get the result
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        assert np.allclose(result, state_data)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_statevector_with_aligned_array(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test StateVector with aligned array."""
        StateVectorClass = get_statevector_class(current_nanobind_module, precision)

        # Create a state vector
        num_qubits = 2
        sv = StateVectorClass(num_qubits)

        # Create an aligned array
        dtype_str = np.complex64 if precision == "64" else np.complex128
        capsule = current_nanobind_module.allocate_aligned_array(
            2**num_qubits, np.dtype(dtype_str), True
        )

        # Convert the capsule to a numpy array using numpy's array interface
        # This approach works with both pybind11 and nanobind
        arr = np.asarray(capsule, dtype=dtype_str)

        # Set the first element to 1.0 to create a valid state
        arr[0] = 1.0

        # Update the state vector with the aligned array
        sv.updateData(arr)

        # Check that the state is correct
        result = np.zeros(2**num_qubits, dtype=dtype_str)
        sv.getState(result)

        expected = np.zeros(2**num_qubits, dtype=dtype_str)
        expected[0] = 1.0
        assert np.allclose(result, expected)
