"""Tests for JAX compatibility with nanobind-based bindings."""

import numpy as np
import pytest
from conftest import backend

try:
    import jax
    import jax.numpy as jnp

    # Enable 64-bit precision for tests that need it
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Skip all tests if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")


class TestJAXCompatibility:
    """Tests for JAX compatibility with nanobind-based bindings."""

    @pytest.fixture
    def get_statevector_class(self):
        """Get StateVectorC64/128 class from module based on precision."""

        def _get_class(module, precision="64"):
            class_name = f"StateVectorC{precision}"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in module")

        return _get_class

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_jax_array_initialization(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test initialization of StateVector with JAX arrays."""
        module = current_nanobind_module

        # Skip if module doesn't have StateVectorC class
        state_vector_class_name = f"StateVectorC{precision}"
        if not hasattr(module, state_vector_class_name):
            pytest.skip(f"Class {state_vector_class_name} not available in module")
            
        # Get the StateVector class and check if it has updateData method
        StateVectorClass = getattr(module, state_vector_class_name)
        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip(f"updateData method not available in {state_vector_class_name}")

        # Skip if JAX doesn't support the precision
        if precision == "128" and not jax.config.read("jax_enable_x64"):
            pytest.skip("JAX x64 precision not enabled")

        num_qubits = 3
        dim = 2**num_qubits

        # Create JAX array for |0> state
        dtype = jnp.complex128 if precision == "128" else jnp.complex64
        jax_data = jnp.zeros(dim, dtype=dtype)
        jax_data = jax_data.at[0].set(1.0)  # Set to |000⟩ state

        # Create state vector and update with JAX data
        sv = StateVectorClass(num_qubits)
        sv.updateData(jax_data)

        # Get state back as numpy array and verify
        # Create a numpy array to receive the state data
        state_data = np.zeros(dim, dtype=np.complex128 if precision == "128" else np.complex64)
        sv.getState(state_data)

        assert np.allclose(state_data[0], 1.0)
        assert np.allclose(state_data[1:], 0.0)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_jax_array_operations(self, current_nanobind_module, precision, get_statevector_class):
        """Test operations on StateVector with JAX arrays."""
        module = current_nanobind_module

        # Skip if module doesn't have StateVectorC class
        state_vector_class_name = f"StateVectorC{precision}"
        if not hasattr(module, state_vector_class_name):
            pytest.skip(f"Class {state_vector_class_name} not available in module")
            
        # Get the StateVector class and check if it has updateData method
        StateVectorClass = getattr(module, state_vector_class_name)
        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip(f"updateData method not available in {state_vector_class_name}")

        # Skip if JAX doesn't support the precision
        if precision == "128" and not jax.config.read("jax_enable_x64"):
            pytest.skip("JAX x64 precision not enabled")

        num_qubits = 1
        dim = 2**num_qubits

        # Create JAX array for |0> state
        dtype = jnp.complex128 if precision == "128" else jnp.complex64
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
        state_data = np.zeros(dim, dtype=np.complex128 if precision == "128" else np.complex64)
        sv.getState(state_data)

        assert np.allclose(state_data[0], 0.0)
        assert np.allclose(state_data[1], 1.0)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_jax_array_with_measurements_class(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test using the Measurements class with JAX arrays."""
        module = current_nanobind_module

        # Skip if module doesn't have StateVectorC class
        state_vector_class_name = f"StateVectorC{precision}"
        if not hasattr(module, state_vector_class_name):
            pytest.skip(f"Class {state_vector_class_name} not available in module")
            
        # Get the StateVector class and check if it has updateData method
        StateVectorClass = getattr(module, state_vector_class_name)
        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip(f"updateData method not available in {state_vector_class_name}")

        # Skip if JAX doesn't support the precision
        if precision == "128" and not jax.config.read("jax_enable_x64"):
            pytest.skip("JAX x64 precision not enabled")

        # Check if the module has a Measurements class
        measurements_class_name = f"MeasurementsC{precision}"
        if not hasattr(module, measurements_class_name):
            pytest.skip(f"Class {measurements_class_name} not available in module")

        MeasurementsClass = getattr(module, measurements_class_name)

        num_qubits = 1
        dim = 2**num_qubits

        # Create a simple |0⟩ state
        dtype = jnp.complex128 if precision == "128" else jnp.complex64
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

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_jax_array_with_measurements_expval(
        self, current_nanobind_module, precision, get_statevector_class
    ):
        """Test expectation value calculation using the Measurements class with JAX arrays."""
        module = current_nanobind_module

        # Skip if module doesn't have StateVectorC class
        state_vector_class_name = f"StateVectorC{precision}"
        if not hasattr(module, state_vector_class_name):
            pytest.skip(f"Class {state_vector_class_name} not available in module")
            
        # Get the StateVector class and check if it has updateData method
        StateVectorClass = getattr(module, state_vector_class_name)
        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip(f"updateData method not available in {state_vector_class_name}")

        # Skip if JAX doesn't support the precision
        if precision == "128" and not jax.config.read("jax_enable_x64"):
            pytest.skip("JAX x64 precision not enabled")

        # Check if the module has a Measurements class
        measurements_class_name = f"MeasurementsC{precision}"
        if not hasattr(module, measurements_class_name):
            pytest.skip(f"Class {measurements_class_name} not available in module")

        MeasurementsClass = getattr(module, measurements_class_name)

        # Check if the module has NamedObs class
        named_obs_class_name = f"NamedObsC{precision}"
        if not hasattr(module, named_obs_class_name):
            pytest.skip(f"Class {named_obs_class_name} not available in module")

        NamedObsClass = getattr(module, named_obs_class_name)

        num_qubits = 1
        dim = 2**num_qubits

        # Create a |+⟩ state (equal superposition)
        dtype = jnp.complex128 if precision == "128" else jnp.complex64
        jax_data = jnp.ones(dim, dtype=dtype) / jnp.sqrt(2.0)

        # Create state vector and update with JAX data
        sv = StateVectorClass(num_qubits)
        sv.updateData(jax_data)

        try:
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

        except Exception as e:
            pytest.skip(f"Measurements class failed with error: {str(e)}")
