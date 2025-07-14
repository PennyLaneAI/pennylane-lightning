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
"""Tests for LightningQubit-specific Nanobind bindings."""

import numpy as np
import pytest
from conftest import device_name

# Skip all tests if not using lightning.qubit
if device_name != "lightning.qubit":
    pytest.skip("Skipping tests for binaries other than lightning_qubit.", allow_module_level=True)


class TestLQubitStateVectorBindings:
    """Tests for LightningQubit-specific StateVector bindings."""

    @pytest.fixture
    def get_statevector_class(self):
        """Get StateVector class based on precision."""

        def _get_class(module, precision="64"):
            class_name = f"StateVectorC{precision}"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in module")

        return _get_class

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_init(self, current_nanobind_module, precision, get_statevector_class):
        """Test initialization of state vector."""
        StateVectorClass = get_statevector_class(current_nanobind_module, precision)
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        # Check that we can get the state
        state = np.zeros(2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128)
        sv.getState(state)
        # Initial state should be |0...0>
        expected = np.zeros(
            2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128
        )
        expected[0] = 1.0
        assert np.allclose(state, expected)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_reset_state_vector(self, current_nanobind_module, precision, get_statevector_class):
        """Test resetting state vector to |0...0>."""
        StateVectorClass = get_statevector_class(current_nanobind_module, precision)
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        # Apply X gate to first qubit to change state
        sv.PauliX([0], False, [])
        # Reset state vector
        sv.resetStateVector()
        # Check state is |0...0>
        state = np.zeros(2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128)
        sv.getState(state)
        expected = np.zeros(
            2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128
        )
        expected[0] = 1.0
        assert np.allclose(state, expected)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_set_basis_state(self, current_nanobind_module, precision, get_statevector_class):
        """Test setting state vector to a basis state."""
        StateVectorClass = get_statevector_class(current_nanobind_module, precision)
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        # Set to |11>
        basis_state = [1, 1]
        sv.setBasisState(basis_state, [0, 1])
        # Check state
        state = np.zeros(2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128)
        sv.getState(state)
        expected = np.zeros(
            2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128
        )
        expected[3] = 1.0
        assert np.allclose(state, expected)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_set_state_vector(self, current_nanobind_module, precision, get_statevector_class):
        """Test setting state vector to a custom state."""
        StateVectorClass = get_statevector_class(current_nanobind_module, precision)
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        # Create a custom state (|00> + |11>)/sqrt(2)
        custom_state = np.zeros(
            2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128
        )
        custom_state[0] = 1.0 / np.sqrt(2)
        custom_state[3] = 1.0 / np.sqrt(2)
        sv.setStateVector(custom_state, [0, 1])
        # Check state
        state = np.zeros(2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128)
        sv.getState(state)
        assert np.allclose(state, custom_state)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_update_data(self, current_nanobind_module, precision, get_statevector_class):
        """Test updating state vector data."""
        StateVectorClass = get_statevector_class(current_nanobind_module, precision)
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        # Create a custom state (|01>)
        custom_state = np.zeros(
            2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128
        )
        custom_state[1] = 1.0
        sv.updateData(custom_state)
        # Check state
        state = np.zeros(2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128)
        sv.getState(state)
        assert np.allclose(state, custom_state)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_kernel_map(self, current_nanobind_module, precision, get_statevector_class):
        """Test getting kernel map."""
        StateVectorClass = get_statevector_class(current_nanobind_module, precision)
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        kernel_map = sv.kernel_map()
        # Check that kernel map contains expected keys
        assert "PauliX" in kernel_map
        assert "Hadamard" in kernel_map
        assert "CNOT" in kernel_map

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_normalize(self, current_nanobind_module, precision, get_statevector_class):
        """Test normalizing state vector."""
        StateVectorClass = get_statevector_class(current_nanobind_module, precision)
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        # Create a non-normalized state
        custom_state = np.zeros(
            2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128
        )
        custom_state[0] = 2.0
        sv.updateData(custom_state)
        # Normalize
        sv.normalize()
        # Check state
        state = np.zeros(2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128)
        sv.getState(state)
        expected = np.zeros(
            2**num_qubits, dtype=np.complex64 if precision == "64" else np.complex128
        )
        expected[0] = 1.0
        assert np.allclose(state, expected)


class TestLQubitMeasurementsBindings:
    """Tests for LightningQubit-specific Measurements bindings."""

    @pytest.fixture
    def get_classes(self):
        """Get StateVector and Measurements classes based on precision."""

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

            return state_vector_class, measurements_class

        return _get_classes

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_expval_named(self, current_nanobind_module, precision, get_classes):
        """Test expected value calculation with named observable."""
        StateVectorClass, MeasurementsClass = get_classes(current_nanobind_module, precision)
        num_qubits = 1
        sv = StateVectorClass(num_qubits)
        # Apply Hadamard
        sv.Hadamard([0], False, [])
        # Create measurements object
        meas = MeasurementsClass(sv)
        # Calculate expected value of PauliZ
        expval = meas.expval("PauliZ", [0])
        assert np.isclose(expval, 0.0)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_var_named(self, current_nanobind_module, precision, get_classes):
        """Test variance calculation with named observable."""
        StateVectorClass, MeasurementsClass = get_classes(current_nanobind_module, precision)
        num_qubits = 1
        sv = StateVectorClass(num_qubits)
        # Apply Hadamard
        sv.Hadamard([0], False, [])
        # Create measurements object
        meas = MeasurementsClass(sv)
        # Calculate variance of PauliZ
        var = meas.var("PauliZ", [0])
        assert np.isclose(var, 1.0)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_generate_samples(self, current_nanobind_module, precision, get_classes):
        """Test generating samples."""
        StateVectorClass, MeasurementsClass = get_classes(current_nanobind_module, precision)
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        # Apply Hadamard to both qubits
        sv.Hadamard([0], False, [])
        sv.Hadamard([1], False, [])
        # Create measurements object
        meas = MeasurementsClass(sv)
        # Generate samples
        num_shots = 1000
        samples = meas.generate_samples([0, 1], num_shots)
        # Check shape
        assert samples.shape == (num_shots, 2)
        # Check distribution (should be roughly uniform)
        counts = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0,
        }
        for sample in samples:
            counts[(sample[0], sample[1])] += 1

        # Each outcome should occur roughly 25% of the time
        for outcome, count in counts.items():
            assert 150 <= count <= 350, f"Outcome {outcome} occurred {count} times, expected ~250"

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_generate_mcmc_samples(self, current_nanobind_module, precision, get_classes):
        """Test generating MCMC samples."""
        StateVectorClass, MeasurementsClass = get_classes(current_nanobind_module, precision)
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        # Apply Hadamard to both qubits
        sv.Hadamard([0], False, [])
        sv.Hadamard([1], False, [])
        # Create measurements object
        meas = MeasurementsClass(sv)
        # Generate samples
        num_shots = 1000
        num_burnin = 100
        samples = meas.generate_mcmc_samples(num_qubits, "local", num_burnin, num_shots)
        # Check shape
        assert samples.shape == (num_shots, num_qubits)


class TestLQubitSparseHamiltonianBindings:
    """Tests for LightningQubit-specific SparseHamiltonian bindings."""

    @pytest.fixture
    def get_sparse_hamiltonian_class(self):
        """Get SparseHamiltonian class based on precision."""

        def _get_class(module, precision="64"):
            # SparseHamiltonian is in the observables submodule
            if hasattr(module, "observables"):
                observables_module = module.observables
                class_name = f"SparseHamiltonianC{precision}"
                if hasattr(observables_module, class_name):
                    return getattr(observables_module, class_name)
                else:
                    pytest.skip(f"Class {class_name} not available in module.observables")
            else:
                pytest.skip("observables submodule not available")

        return _get_class

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_init(self, current_nanobind_module, precision, get_sparse_hamiltonian_class):
        """Test initialization of SparseHamiltonian."""
        SparseHamiltonianClass = get_sparse_hamiltonian_class(current_nanobind_module, precision)

        # Create a simple sparse matrix for the Pauli-Z operator
        data = np.array([1.0, -1.0], dtype=np.complex64 if precision == "64" else np.complex128)
        indices = np.array([0, 1], dtype=np.uint64)
        indptr = np.array([0, 1, 2], dtype=np.uint64)
        wires = [0]

        # Create SparseHamiltonian
        sparse_ham = SparseHamiltonianClass(data, indices, indptr, wires)

        # Check wires
        assert sparse_ham.get_wires() == wires

        # Check string representation
        assert "SparseHamiltonian" in sparse_ham.__repr__()


class TestLQubitVJPBindings:
    """Tests for LightningQubit-specific VectorJacobianProduct bindings."""

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_vjp_init(self, current_nanobind_module, precision):
        """Test VectorJacobianProduct initialization."""
        # VectorJacobianProduct is in the algorithms submodule
        if hasattr(current_nanobind_module, "algorithms"):
            algorithms_module = current_nanobind_module.algorithms
            vjp_class_name = f"VectorJacobianProductC{precision}"
            if hasattr(algorithms_module, vjp_class_name):
                vjp_class = getattr(algorithms_module, vjp_class_name)
                vjp = vjp_class()
                assert vjp is not None
            else:
                pytest.skip(f"Class {vjp_class_name} not available in module.algorithms")
        else:
            pytest.skip("algorithms submodule not available")


class TestLQubitBackendInfoBindings:
    """Tests for LightningQubit-specific backend info bindings."""

    def test_backend_info(self, current_nanobind_module):
        """Test getting backend info."""
        if hasattr(current_nanobind_module, "backend_info"):
            info = current_nanobind_module.backend_info()
            assert info["NAME"] == "lightning.qubit"
        else:
            pytest.skip("backend_info function not available in module")
