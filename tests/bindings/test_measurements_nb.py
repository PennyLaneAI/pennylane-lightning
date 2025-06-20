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
"""Tests for Measurements classes in nanobind-based modules."""

import numpy as np
import pytest
from conftest import backend

if backend != "qubit":
    pytest.skip("Skipping tests for binaries other than lightning_qubit .", allow_module_level=True)


class TestMeasurementsNB:
    """Tests for MeasurementsC64 and MeasurementsC128 classes in nanobind-based modules."""

    num_qubits = 2

    @pytest.fixture
    def get_classes(self):
        """Get StateVector, Measurements, NamedObs, and HermitianObs classes from module based on precision."""

        def _get_classes(module, precision="64"):
            # Get StateVector class
            state_vector_class_name = f"StateVectorC{precision}"
            if hasattr(module, state_vector_class_name):
                state_vector_class = getattr(module, state_vector_class_name)
            else:
                pytest.skip(f"Class {state_vector_class_name} not available in module")

            # Get Measurements class
            measurements_class_name = f"MeasurementsC{precision}"
            if hasattr(module, measurements_class_name):
                measurements_class = getattr(module, measurements_class_name)
            else:
                pytest.skip(f"Class {measurements_class_name} not available in module")

            # Get NamedObs class
            named_obs_class_name = f"NamedObsC{precision}"
            if hasattr(module, named_obs_class_name):
                named_obs_class = getattr(module, named_obs_class_name)
            else:
                # Try to find it in the observables submodule if it exists
                if hasattr(module, "observables") and hasattr(
                    module.observables, named_obs_class_name
                ):
                    named_obs_class = getattr(module.observables, named_obs_class_name)
                else:
                    pytest.skip(
                        f"Class {named_obs_class_name} not available in module or module.observables"
                    )

            # Get HermitianObs class
            hermitian_obs_class_name = f"HermitianObsC{precision}"
            if hasattr(module, hermitian_obs_class_name):
                hermitian_obs_class = getattr(module, hermitian_obs_class_name)
            else:
                # Try to find it in the observables submodule if it exists
                if hasattr(module, "observables") and hasattr(
                    module.observables, hermitian_obs_class_name
                ):
                    hermitian_obs_class = getattr(module.observables, hermitian_obs_class_name)
                else:
                    hermitian_obs_class = None

            return state_vector_class, measurements_class, named_obs_class, hermitian_obs_class

        return _get_classes

    def test_state_initialization(self, current_nanobind_module, precision, get_classes):
        """Test state vector initialization."""
        module = current_nanobind_module

        StateVectorClass, MeasurementsClass, _, _ = get_classes(module, precision)

        # Initialize with number of qubits - already in |0⟩ state
        sv = StateVectorClass(self.num_qubits)

        # Check initial state probabilities
        meas = MeasurementsClass(sv)
        probs = meas.probs(list(range(self.num_qubits)))

        # The first element should be 1.0, the rest should be 0.0
        assert probs[0] == pytest.approx(1.0)
        assert all(p == pytest.approx(0.0) for p in probs[1:])

    def test_bell_state(self, current_nanobind_module, precision, get_classes):
        """Test Bell state creation."""
        module = current_nanobind_module

        StateVectorClass, MeasurementsClass, _, _ = get_classes(module, precision)

        # Initialize with number of qubits - already in |00⟩ state
        sv = StateVectorClass(self.num_qubits)

        # Check initial state
        initial_meas = MeasurementsClass(sv)
        initial_probs = initial_meas.probs([0, 1])
        assert np.allclose(initial_probs, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

        # Create Bell state step by step
        sv.Hadamard([0], False, [])
        sv.CNOT([0, 1], False, [])

        # Check final state
        bell_meas = MeasurementsClass(sv)
        bell_probs = bell_meas.probs([0, 1])
        assert np.allclose(bell_probs, [0.5, 0.0, 0.0, 0.5], atol=1e-6)

    def test_measurements_expval_named_obs(self, current_nanobind_module, precision, get_classes):
        """Test expectation value calculation with NamedObs."""
        module = current_nanobind_module

        StateVectorClass, MeasurementsClass, NamedObsClass, _ = get_classes(module, precision)

        num_qubits = 1

        # Initialize with number of qubits - already in |0⟩ state
        sv = StateVectorClass(num_qubits)

        sv.Hadamard([0], False, [])  # Put in |+⟩ state

        meas = MeasurementsClass(sv)

        # Create PauliX observable
        obs_x = NamedObsClass("PauliX", [0])
        expval_x = meas.expval(obs_x)
        assert np.isclose(expval_x, 1.0, atol=1e-6)

        # Create PauliZ observable
        obs_z = NamedObsClass("PauliZ", [0])
        expval_z = meas.expval(obs_z)
        assert np.isclose(expval_z, 0.0, atol=1e-6)

    def test_measurements_var_named_obs(self, current_nanobind_module, precision, get_classes):
        """Test variance calculation with NamedObs."""
        module = current_nanobind_module

        StateVectorClass, MeasurementsClass, NamedObsClass, _ = get_classes(module, precision)

        # Skip if NamedObsClass is None
        if NamedObsClass is None:
            pytest.skip(f"NamedObsClass not available for {backend} with precision {precision}")

        num_qubits = 1

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)

        sv.Hadamard([0], False, [])  # Put in |+⟩ state

        meas = MeasurementsClass(sv)

        # Create PauliX observable
        obs_x = NamedObsClass("PauliX", [0])
        var_x = meas.var(obs_x)
        assert np.isclose(var_x, 0.0, atol=1e-6)

        # Create PauliZ observable
        obs_z = NamedObsClass("PauliZ", [0])
        var_z = meas.var(obs_z)
        assert np.isclose(var_z, 1.0, atol=1e-6)

    def test_measurements_expval_hermitian_obs(
        self, current_nanobind_module, precision, get_classes
    ):
        """Test expectation value calculation with HermitianObs."""
        module = current_nanobind_module

        StateVectorClass, MeasurementsClass, _, HermitianObsClass = get_classes(module, precision)

        # Skip if HermitianObsClass is None
        if HermitianObsClass is None:
            pytest.skip(f"HermitianObsClass not available for {backend} with precision {precision}")

        num_qubits = 1

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)

        sv.Hadamard([0], False, [])  # Put in |+⟩ state

        meas = MeasurementsClass(sv)

        # Define Pauli-X matrix
        pauli_x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)

        # Create Hermitian observable with Pauli-X matrix
        obs_x = HermitianObsClass(pauli_x_matrix.flatten(), [0])
        expval_x = meas.expval(obs_x)
        assert np.isclose(expval_x, 1.0, atol=1e-6)

        # Define Pauli-Z matrix
        pauli_z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)

        # Create Hermitian observable with Pauli-Z matrix
        obs_z = HermitianObsClass(pauli_z_matrix.flatten(), [0])
        expval_z = meas.expval(obs_z)
        assert np.isclose(expval_z, 0.0, atol=1e-6)

    def test_measurements_var_hermitian_obs(self, current_nanobind_module, precision, get_classes):
        """Test variance calculation with HermitianObs."""
        module = current_nanobind_module

        StateVectorClass, MeasurementsClass, _, HermitianObsClass = get_classes(module, precision)

        # Skip if HermitianObsClass is None
        if HermitianObsClass is None:
            pytest.skip(f"HermitianObsClass not available for {backend} with precision {precision}")

        num_qubits = 1

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)

        sv.Hadamard([0], False, [])  # Put in |+⟩ state

        meas = MeasurementsClass(sv)

        # Define Pauli-X matrix
        pauli_x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)

        # Create Hermitian observable with Pauli-X matrix
        obs_x = HermitianObsClass(pauli_x_matrix.flatten(), [0])
        var_x = meas.var(obs_x)
        assert np.isclose(var_x, 0.0, atol=1e-6)

        # Define Pauli-Z matrix
        pauli_z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)

        # Create Hermitian observable with Pauli-Z matrix
        obs_z = HermitianObsClass(pauli_z_matrix.flatten(), [0])
        var_z = meas.var(obs_z)
        assert np.isclose(var_z, 1.0, atol=1e-6)

    def test_hadamard_gate(self, current_nanobind_module, precision, get_classes):
        """Test Hadamard gate application."""
        module = current_nanobind_module

        StateVectorClass, MeasurementsClass, _, _ = get_classes(module, precision)
        dtype = np.complex64 if precision == "64" else np.complex128

        num_qubits = 1
        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)  # Single qubit

        # Get the state
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Check initial probabilities
        meas = MeasurementsClass(sv)
        probs = meas.probs([0])

        # Should be [1.0, 0.0]
        assert np.allclose(probs, [1.0, 0.0], atol=1e-6)

        # Apply Hadamard
        sv.Hadamard([0], False, [])

        # Get the state
        result = np.zeros(2**num_qubits, dtype=dtype)
        sv.getState(result)

        # Check probabilities
        meas = MeasurementsClass(sv)
        probs = meas.probs([0])

        # Should be [0.5, 0.5]
        assert np.allclose(probs, [0.5, 0.5], atol=1e-6)

    def test_measurements_generate_samples(self, get_classes, precision, current_nanobind_module):
        """Test sample generation."""
        module = current_nanobind_module

        StateVectorClass, MeasurementsClass, _, _ = get_classes(module, precision)

        num_qubits = 2

        # Initialize with number of qubits - already in |00⟩ state
        sv = StateVectorClass(num_qubits)

        # Create Bell state
        sv.Hadamard([0], False, [])
        sv.CNOT([0, 1], False, [])

        meas = MeasurementsClass(sv)

        # Generate samples
        num_samples = 1000
        samples = meas.generate_samples(num_qubits, num_samples)

        # Check shape
        assert samples.shape == (num_samples, 2)

        # In a Bell state, the qubits should be perfectly correlated
        # Count how many times we get |00⟩ and |11⟩
        count_00 = np.sum(np.all(samples == [0, 0], axis=1))
        count_11 = np.sum(np.all(samples == [1, 1], axis=1))

        # We should have approximately 50% |00⟩ and 50% |11⟩
        # and very few or no |01⟩ and |10⟩
        assert count_00 + count_11 >= 0.95 * num_samples  # Allow for small statistical fluctuations
        assert 0.4 <= count_00 / num_samples <= 0.6  # Allow for statistical fluctuations
        assert 0.4 <= count_11 / num_samples <= 0.6  # Allow for statistical fluctuations

        # Test generate_samples for all wires
        all_samples = meas.generate_samples(num_qubits, num_samples)
        assert all_samples.shape == (num_samples, num_qubits)

        # The results should be the same as when specifying all wires
        count_00_all = np.sum(np.all(all_samples == [0, 0], axis=1))
        count_11_all = np.sum(np.all(all_samples == [1, 1], axis=1))

        # The counts should be similar to the previous test
        assert 0.4 * num_samples <= count_00_all <= 0.6 * num_samples
        assert 0.4 * num_samples <= count_11_all <= 0.6 * num_samples

    def test_measurements_with_controlled_gates(self, current_nanobind_module, precision, get_classes):
        """Test measurements with controlled gates."""
        module = current_nanobind_module

        StateVectorClass, MeasurementsClass, NamedObsClass, _ = get_classes(module, precision)

        num_qubits = 3

        # Initialize with number of qubits
        sv = StateVectorClass(num_qubits)

        # Create a GHZ state using Hadamard and CNOT
        sv.Hadamard([0], False, [])
        sv.CNOT([0, 1], False, [])
        sv.CNOT([1, 2], False, [])

        # Create measurements object
        meas = MeasurementsClass(sv)

        # Check probabilities - should be 50% |000⟩ and 50% |111⟩
        probs = meas.probs(list(range(num_qubits)))
        assert np.isclose(probs[0], 0.5, atol=1e-6)  # |000⟩
        assert np.isclose(probs[7], 0.5, atol=1e-6)  # |111⟩
        assert np.sum(probs) == pytest.approx(1.0)

        # Test controlled operations
        sv.resetStateVector()
        
        # Create |+⟩ state on qubit 0
        sv.Hadamard([0], False, [])
        
        # Apply CNOT with control=0, target=1
        sv.CNOT([0, 1], False, [])
        
        # Create measurements object
        meas = MeasurementsClass(sv)
        
        # Create PauliX observable on qubit 1
        obs_x = NamedObsClass("PauliX", [1])
        
        # For the state (|00⟩ + |11⟩)/√2, the expectation value of X₁ is 0
        # because X₁|00⟩ = |01⟩ and X₁|11⟩ = |10⟩, which are orthogonal to the original state
        expval_x = meas.expval(obs_x)
        assert np.isclose(expval_x, 0.0, atol=1e-6)
        
        # Create PauliZ observable on qubit 1
        obs_z = NamedObsClass("PauliZ", [1])
        
        # For the state (|00⟩ + |11⟩)/√2, the expectation value of Z₁ is 0
        # because Z₁|00⟩ = |00⟩ and Z₁|11⟩ = -|11⟩
        expval_z = meas.expval(obs_z)
        assert np.isclose(expval_z, 0.0, atol=1e-6)
        
        # Create a correlation observable Z₀⊗Z₁
        obs_z0 = NamedObsClass("PauliZ", [0])
        obs_z1 = NamedObsClass("PauliZ", [1])
        
        # For the Bell state, ⟨Z₀⊗Z₁⟩ = 1
        expval_z0 = meas.expval(obs_z0)
        expval_z1 = meas.expval(obs_z1)
        
        # Individual Z measurements should be 0
        assert np.isclose(expval_z0, 0.0, atol=1e-6)
        assert np.isclose(expval_z1, 0.0, atol=1e-6)
        
        # But their correlation should be 1
        # We can verify this by measuring in the computational basis
        probs = meas.probs([0, 1])
        assert np.isclose(probs[0], 0.5, atol=1e-6)  # |00⟩
        assert np.isclose(probs[3], 0.5, atol=1e-6)  # |11⟩
        assert np.isclose(probs[1], 0.0, atol=1e-6)  # |01⟩
        assert np.isclose(probs[2], 0.0, atol=1e-6)  # |10⟩

    def test_measurements_with_multi_controlled_gates(self, current_nanobind_module, precision, get_classes):
        """Test measurements with multi-controlled gates."""
        module = current_nanobind_module
        StateVectorClass, MeasurementsClass, NamedObsClass, _ = get_classes(module, precision)

        sv = StateVectorClass(3)
        
        # Test 1: Toffoli gate - multi-controlled operation
        # Start with |110⟩ state to activate the Toffoli
        sv.PauliX([0], False, [])  # |100⟩
        sv.PauliX([1], False, [])  # |110⟩
        
        # Apply Toffoli: should flip target qubit 2 since both controls are |1⟩
        sv.Toffoli([0, 1, 2], False, [])  # |110⟩ → |111⟩
        
        meas = MeasurementsClass(sv)
        probs = meas.probs([0, 1, 2])
        
        # Should be in |111⟩ state
        assert np.isclose(probs[7], 1.0, atol=1e-6)  # |111⟩ = index 7
        
        # Test 2: Multi-controlled phase gate behavior
        sv.resetStateVector()
        
        # Create entangled state: (|000⟩ + |111⟩)/√2
        sv.Hadamard([0], False, [])
        sv.CNOT([0, 1], False, [])
        sv.CNOT([1, 2], False, [])
        
        # Apply controlled-Z between qubits 0 and 2
        sv.CZ([0, 2], False, [])
        
        meas = MeasurementsClass(sv)
        
        # CZ only adds phase to |11⟩ component, doesn't change probabilities
        probs = meas.probs([0, 1, 2])
        assert np.isclose(probs[0], 0.5, atol=1e-6)  # |000⟩
        assert np.isclose(probs[7], 0.5, atol=1e-6)  # |111⟩
        
        # Test expectation values
        obs_z0 = NamedObsClass("PauliZ", [0])
        obs_z2 = NamedObsClass("PauliZ", [2])
        
        # Individual measurements are zero due to superposition
        assert np.isclose(meas.expval(obs_z0), 0.0, atol=1e-6)
        assert np.isclose(meas.expval(obs_z2), 0.0, atol=1e-6)
        
        # Variance should be 1 for maximally mixed single-qubit states
        assert np.isclose(meas.var(obs_z0), 1.0, atol=1e-6)