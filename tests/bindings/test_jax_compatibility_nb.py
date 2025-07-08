"""Tests for JAX compatibility with nanobind-based bindings."""

import numpy as np
import pytest
from conftest import backend

try:
    import jax
    import jax.numpy as jnp

    # Enable 64-bit precision for tests that need it
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Skip all tests if JAX is not available
    pytestmark = pytest.mark.skip(reason="JAX not available")


class TestJAXCompatibility:
    """Tests for JAX compatibility with nanobind-based bindings."""

    @pytest.fixture
    def get_statevector_class_and_precision(self, current_nanobind_module, precision):
        """Get StateVectorC64/128 class from module based on precision."""
        if JAX_AVAILABLE:
            if precision == "128":
                jax.config.update("jax_enable_x64", True)

        def _get_class():
            module = current_nanobind_module
            class_name = f"StateVectorC{precision}"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in module {module}")

        dtype = np.complex128 if precision == "128" else np.complex64
        return _get_class(), dtype

    @pytest.fixture
    def get_measurements_class(self, current_nanobind_module):
        def _get_class(dtype):
            module = current_nanobind_module
            class_name = "MeasurementsC64" if dtype == np.complex64 else "MeasurementsC128"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in module {module}")

        return _get_class

    @pytest.fixture
    def get_named_obs_class(self, current_nanobind_module):
        def _get_class(dtype):
            module = current_nanobind_module
            class_name = "NamedObsC64" if dtype == np.complex64 else "NamedObsC128"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in module {module}")

        return _get_class

    def test_jax_array_initialization(self, get_statevector_class_and_precision):
        """Test initialization of StateVector with JAX arrays."""
        # Skip if module doesn't have StateVectorC class
        StateVectorClass, dtype = get_statevector_class_and_precision

        num_qubits = 3
        dim = 2**num_qubits

        # Create JAX array for |0> state
        jax_data = jnp.zeros(dim, dtype=dtype)
        jax_data = jax_data.at[0].set(1.0)  # Set to |000⟩ state

        # Create state vector and update with JAX data
        sv = StateVectorClass(num_qubits)
        sv.updateData(jax_data)

        # Get state back as numpy array and verify
        # Create a numpy array to receive the state data
        state_data = np.zeros(dim, dtype=dtype)
        sv.getState(state_data)

        assert np.allclose(state_data[0], 1.0)
        assert np.allclose(state_data[1:], 0.0)

    def test_jax_array_operations(self, get_statevector_class_and_precision):
        """Test operations on StateVector with JAX arrays."""
        # Skip if module doesn't have StateVectorC class
        StateVectorClass, dtype = get_statevector_class_and_precision

        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip(f"updateData method not available. ")

        num_qubits = 1
        dim = 2**num_qubits

        # Create JAX array for |0> state
        jax_data = jnp.zeros(dim, dtype=dtype)
        jax_data = jax_data.at[0].set(1.0)  # Set to |0⟩ state

        # Create state vector and update with JAX data
        sv = StateVectorClass(num_qubits)
        sv.updateData(jax_data)

        # Apply X gate using applyMatrix instead of applyPauliX
        # Create Pauli-X matrix
        x_matrix = jnp.array([[0, 1], [1, 0]], dtype=dtype)
        sv.applyMatrix(x_matrix, [0], False)

        # Get state back as numpy array and verify
        state_data = np.zeros(dim, dtype=dtype)
        sv.getState(state_data)

        assert np.allclose(state_data[0], 0.0)
        assert np.allclose(state_data[1], 1.0)

    def test_jax_array_with_measurements_class(
        self, get_statevector_class_and_precision, get_measurements_class
    ):
        """Test using the Measurements class with JAX arrays."""
        # Skip if module doesn't have StateVectorC class
        StateVectorClass, dtype = get_statevector_class_and_precision

        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip(f"updateData method not available.")

        # Check if the module has a Measurements class and get it.
        MeasurementsClass = get_measurements_class(dtype)

        num_qubits = 1
        dim = 2**num_qubits

        # Create a simple |0⟩ state
        jax_data = jnp.zeros(dim, dtype=dtype)
        jax_data = jax_data.at[0].set(1.0)

        # Create state vector and update with JAX data
        sv = StateVectorClass(num_qubits)
        sv.updateData(jax_data)

        # Create a Measurements object
        measurements = MeasurementsClass(sv)

        # Test probabilities method
        np_probs = measurements.probs([0])

        # Verify probabilities
        expected_probs = np.array([1.0, 0.0])
        assert np.allclose(np_probs, expected_probs)

        # Apply Hadamard gate to create |+⟩ state
        h_matrix = jnp.array([[1, 1], [1, -1]], dtype=dtype) / jnp.sqrt(2.0)
        sv.applyMatrix(h_matrix, [0], False)

        # Test probabilities again
        np_probs = measurements.probs([0])

        # Verify probabilities for |+⟩ state
        expected_probs = np.array([0.5, 0.5])
        assert np.allclose(np_probs, expected_probs)

    def test_jax_array_with_measurements_expval(
        self, get_statevector_class_and_precision, get_measurements_class, get_named_obs_class
    ):
        """Test expectation value calculation using the Measurements class with JAX arrays."""

        # Skip if module doesn't have StateVectorC class
        StateVectorClass, dtype = get_statevector_class_and_precision

        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip(f"updateData method not available.")

        # Check if the module has a Measurements class
        MeasurementsClass = get_measurements_class(dtype)

        # Check if the module has an observables submodule with NamedObs class
        NamedObsClass = get_named_obs_class(dtype)

        num_qubits = 1
        dim = 2**num_qubits

        # Create a |+⟩ state (equal superposition)
        jax_data = jnp.ones(dim, dtype=dtype) / jnp.sqrt(2.0)

        # Create state vector and update with JAX data
        sv = StateVectorClass(num_qubits)
        sv.updateData(jax_data)

        # Create a Measurements object
        measurements = MeasurementsClass(sv)

        # Create PauliX observable
        x_obs = NamedObsClass("PauliX", [0])

        # Calculate expectation value
        x_expval = measurements.expval(x_obs)

        # For |+⟩ state, ⟨X⟩ = 1
        assert np.allclose(x_expval, 1.0)

        # Create PauliZ observable
        z_obs = NamedObsClass("PauliZ", [0])

        # Calculate expectation value
        z_expval = measurements.expval(z_obs)

        # For |+⟩ state, ⟨Z⟩ = 0
        assert np.allclose(z_expval, 0.0)
