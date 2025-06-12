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

import importlib

import numpy as np
import pytest


class TestMeasurementsNB:
    """Tests for MeasurementsC64 and MeasurementsC128 classes in nanobind-based modules."""

    num_qubits = 2
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
                module = importlib.import_module(module_name)
                self.module_attributes[module_name] = {"importable": True, "module": module}
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
    def get_classes(self, precision):
        """Fixture to get the appropriate classes based on precision."""

        def _get_classes(module):
            sv_class = getattr(module, f"StateVectorC{precision}")
            meas_class = getattr(module, f"MeasurementsC{precision}")
            named_obs_class = getattr(module.observables, f"NamedObsC{precision}")
            hermitian_obs_class = getattr(module.observables, f"HermitianObsC{precision}")
            return sv_class, meas_class, named_obs_class, hermitian_obs_class

        return _get_classes

    @pytest.mark.parametrize("module_name", modules)
    def test_state_initialization(self, module_name, get_classes, precision):
        """Test state vector initialization."""
        module = self._skip_if_module_not_importable(module_name)

        StateVectorClass, MeasurementsClass, _, _ = get_classes(module)
        # Create a numpy array representing |0> state with appropriate size
        dtype = np.complex64 if precision == "64" else np.complex128
        state_data = np.zeros(2**self.num_qubits, dtype=dtype)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(self.num_qubits)
        # Then update data
        sv.updateData(state_data)

        # Check initial state probabilities
        meas = MeasurementsClass(sv)
        probs = meas.probs(list(range(self.num_qubits)))

        # The first element should be 1.0, all others 0.0
        expected = np.zeros(2**self.num_qubits)
        expected[0] = 1.0

        assert np.allclose(probs, expected, atol=1e-6)

    @pytest.mark.parametrize("module_name", modules)
    def test_bell_state(self, module_name, get_classes, precision):
        """Test Bell state creation."""
        module = self._skip_if_module_not_importable(module_name)

        StateVectorClass, MeasurementsClass, _, _ = get_classes(module)
        # Create a numpy array representing |0> state with appropriate size
        dtype = np.complex64 if precision == "64" else np.complex128
        state_data = np.zeros(2**self.num_qubits, dtype=dtype)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(self.num_qubits)
        # Then update data
        sv.updateData(state_data)

        # Check initial state
        initial_meas = MeasurementsClass(sv)
        initial_probs = initial_meas.probs([0, 1])
        assert np.allclose(initial_probs, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

        # Create Bell state step by step
        sv.Hadamard([0], False, [])

        # Check after Hadamard
        hadamard_meas = MeasurementsClass(sv)
        hadamard_probs = hadamard_meas.probs([0, 1])
        assert np.allclose(hadamard_probs, [0.5, 0.0, 0.5, 0.0], atol=1e-6)

        # Apply CNOT
        sv.CNOT([0, 1], False, [])

        # Final measurement
        meas = MeasurementsClass(sv)
        probs = meas.probs([0, 1])
        assert np.allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-6)

    @pytest.mark.parametrize("module_name", modules)
    def test_measurements_expval_named_obs(self, module_name, get_classes, precision):
        """Test expectation value calculation with NamedObs."""
        module = self._skip_if_module_not_importable(module_name)

        StateVectorClass, MeasurementsClass, NamedObsClass, _ = get_classes(module)

        num_qubits = 1
        # Create a numpy array representing |0> state with appropriate size
        dtype = np.complex64 if precision == "64" else np.complex128
        state_data = np.zeros(2**num_qubits, dtype=dtype)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

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

    @pytest.mark.parametrize("module_name", modules)
    def test_measurements_var_named_obs(self, module_name, get_classes, precision):
        """Test variance calculation with NamedObs."""
        module = self._skip_if_module_not_importable(module_name)

        StateVectorClass, MeasurementsClass, NamedObsClass, _ = get_classes(module)

        num_qubits = 1
        # Create a numpy array representing |0> state with appropriate size
        dtype = np.complex64 if precision == "64" else np.complex128
        state_data = np.zeros(2**num_qubits, dtype=dtype)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

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

    @pytest.mark.parametrize("module_name", modules)
    def test_measurements_expval_hermitian_obs(self, module_name, get_classes, precision):
        """Test expectation value calculation with HermitianObs."""
        module = self._skip_if_module_not_importable(module_name)

        StateVectorClass, MeasurementsClass, _, HermitianObsClass = get_classes(module)

        num_qubits = 1
        # Create a numpy array representing |0> state with appropriate size
        dtype = np.complex64 if precision == "64" else np.complex128
        state_data = np.zeros(2**num_qubits, dtype=dtype)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

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

    @pytest.mark.parametrize("module_name", modules)
    def test_measurements_var_hermitian_obs(self, module_name, get_classes, precision):
        """Test variance calculation with HermitianObs."""
        module = self._skip_if_module_not_importable(module_name)

        StateVectorClass, MeasurementsClass, _, HermitianObsClass = get_classes(module)

        num_qubits = 1
        # Create a numpy array representing |0> state with appropriate size
        dtype = np.complex64 if precision == "64" else np.complex128
        state_data = np.zeros(2**num_qubits, dtype=dtype)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

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

    @pytest.mark.parametrize("module_name", modules)
    def test_hadamard_gate(self, module_name, get_classes, precision):
        """Test Hadamard gate application."""
        module = self._skip_if_module_not_importable(module_name)

        StateVectorClass, MeasurementsClass, _, _ = get_classes(module)
        # Create a numpy array representing |0> state with appropriate size
        dtype = np.complex64 if precision == "64" else np.complex128
        state_data = np.zeros(2, dtype=dtype)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(1)  # Single qubit
        # Then update data
        sv.updateData(state_data)

        # Get the state
        result = np.zeros(2, dtype=dtype)
        sv.getState(result)

        # Check initial probabilities
        meas = MeasurementsClass(sv)
        probs = meas.probs([0])

        # Should be [1.0, 0.0]
        assert np.allclose(probs, [1.0, 0.0], atol=1e-6)

        # Apply Hadamard
        sv.Hadamard([0], False, [])

        # Get the state
        result = np.zeros(2, dtype=dtype)
        sv.getState(result)

        # Check probabilities
        meas = MeasurementsClass(sv)
        probs = meas.probs([0])

        # Should be [0.5, 0.5]
        assert np.allclose(probs, [0.5, 0.5], atol=1e-6)

    @pytest.mark.parametrize("module_name", modules)
    def test_measurements_generate_samples(self, module_name, get_classes, precision):
        """Test sample generation."""
        module = self._skip_if_module_not_importable(module_name)

        StateVectorClass, MeasurementsClass, _, _ = get_classes(module)

        num_qubits = 2
        # Create a numpy array representing |0> state with appropriate size
        dtype = np.complex64 if precision == "64" else np.complex128
        state_data = np.zeros(2**num_qubits, dtype=dtype)
        state_data[0] = 1.0

        # Initialize with number of qubits first
        sv = StateVectorClass(num_qubits)
        # Then update data
        sv.updateData(state_data)

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
        assert abs(count_00 - count_00_all) < 0.1 * num_samples
        assert abs(count_11 - count_11_all) < 0.1 * num_samples
