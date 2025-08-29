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
"""Tests for JAX compatibility with nanobind-based bindings."""

import numpy as np
import pytest

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
    def get_statevector_class_and_precision(self, precision, current_module):
        """Get StateVectorC64/128 class from module based on precision."""
        module = current_module
        if JAX_AVAILABLE:
            if precision == "128":
                jax.config.update("jax_enable_x64", True)

        def _get_class():
            class_name = f"StateVectorC{precision}"
            if hasattr(module, class_name):
                StateVectorClass = getattr(module, class_name)
                return StateVectorClass
            pytest.skip(f"Class {class_name} not available in module {module}")

        dtype = np.complex128 if precision == "128" else np.complex64
        return _get_class(), dtype

    @pytest.fixture
    def get_measurements_class(self, current_module):
        module = current_module

        def _get_class(dtype):
            class_name = "MeasurementsC64" if dtype == np.complex64 else "MeasurementsC128"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in module {module}")

        return _get_class

    @pytest.fixture
    def get_named_obs_class(self, current_module):
        """Get the NamedObs class from a module."""
        module = current_module.observables

        def _get_class(dtype):
            class_name = "NamedObsC64" if dtype == np.complex64 else "NamedObsC128"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in {module}.observables")

        return _get_class

    @pytest.fixture
    def get_ops_struct_class(self, current_module):
        """Get the OpsStruct class from a module."""
        module = current_module.algorithms

        def _get_class(dtype):
            class_name = "OpsStructC64" if dtype == np.complex64 else "OpsStructC128"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in {module}.algorithms")

        return _get_class

    @pytest.fixture
    def get_adjoint_jacobian_class(self, current_module):
        """Get the AdjointJacobian class from a module."""
        module = current_module.algorithms

        def _get_class(dtype):
            class_name = "AdjointJacobianC64" if dtype == np.complex64 else "AdjointJacobianC128"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in {module}.algorithms")

        return _get_class

    def test_adjoint_jacobian_jax(
        self,
        get_statevector_class_and_precision,
        get_adjoint_jacobian_class,
        get_named_obs_class,
        get_ops_struct_class,
    ):
        """Test adjoint Jacobian with JAX parameters."""

        StateVectorClass, dtype = get_statevector_class_and_precision

        # Check if the module has AdjointJacobian class
        AdjointJacobianClass = get_adjoint_jacobian_class(dtype)

        # Check if the module has NamedObs class
        NamedObsClass = get_named_obs_class(dtype)

        # Check if the module has OpsStruct class
        OpsStructClass = get_ops_struct_class(dtype)

        # Create a simple state vector
        num_qubits = 2
        sv = StateVectorClass(num_qubits)

        # Create a simple observable
        obs = NamedObsClass("PauliZ", [0])

        # Create a simple operation with JAX parameter
        param_value = jnp.array(0.5)
        ops_name = ["RX"]
        ops_params = [[param_value]]
        ops_wires = [[0]]
        ops_inverses = [False]
        ops_matrices = [[]]
        ops_controlled_wires = [[]]
        ops_controlled_values = [[]]

        # Create the operations structure
        ops = OpsStructClass(
            ops_name,
            ops_params,
            ops_wires,
            ops_inverses,
            ops_matrices,
            ops_controlled_wires,
            ops_controlled_values,
        )

        # Create the adjoint Jacobian
        adj = AdjointJacobianClass()

        # Calculate the Jacobian with the correct parameter order
        trainable_params = [0]
        result = adj(sv, [obs], ops, trainable_params)

        # The result should be a numpy array with shape (1,)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], -np.sin(param_value))

    def test_adjoint_jacobian_multiple_params_jax(
        self,
        get_statevector_class_and_precision,
        get_adjoint_jacobian_class,
        get_named_obs_class,
        get_ops_struct_class,
    ):
        """Test adjoint Jacobian with multiple JAX parameters."""

        StateVectorClass, dtype = get_statevector_class_and_precision

        # Get necessary classes
        AdjointJacobianClass = get_adjoint_jacobian_class(dtype)
        NamedObsClass = get_named_obs_class(dtype)
        OpsStructClass = get_ops_struct_class(dtype)

        # Create a simple state vector
        num_qubits = 2
        sv = StateVectorClass(num_qubits)

        # Create a simple observable
        obs = NamedObsClass("PauliZ", [0])

        # Create operations with JAX parameters
        params = jnp.array([0.1, 0.2], dtype=dtype)
        ops_name = ["RX", "RY"]
        ops_params = [[params[0]], [params[1]]]
        ops_wires = [[0], [0]]
        ops_inverses = [False, False]
        ops_matrices = [[], []]
        ops_controlled_wires = [[], []]
        ops_controlled_values = [[], []]

        # Create the operations structure
        ops = OpsStructClass(
            ops_name,
            ops_params,
            ops_wires,
            ops_inverses,
            ops_matrices,
            ops_controlled_wires,
            ops_controlled_values,
        )

        # Create the adjoint Jacobian
        adj = AdjointJacobianClass()

        # Calculate the Jacobian with the correct parameter order
        trainable_params = [0, 1]
        jacobian = adj(sv, [obs], ops, trainable_params)

        # Verify the shape of the Jacobian
        assert jacobian.shape == (1, 2)  # 1 expectation value, 2 parameters

        # Verify the correctness of the Jacobian
        # The expected Jacobian can be computed analytically
        expected_jacobian = np.array([[-np.sin(params[0]), 0]])
        assert np.allclose(jacobian, expected_jacobian, atol=1e-5)

    def test_jax_array_initialization(self, get_statevector_class_and_precision):
        """Test initialization of StateVector with JAX arrays."""
        StateVectorClass, dtype = get_statevector_class_and_precision

        # Check if it has updateData and getState methods
        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip("updateData method not available in this backend")
        if not hasattr(StateVectorClass, "getState"):
            pytest.skip("getState method not available in this backend")

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
        StateVectorClass, dtype = get_statevector_class_and_precision

        # Check if it has updateData and getState methods
        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip("updateData method not available in this backend")
        if not hasattr(StateVectorClass, "getState"):
            pytest.skip("getState method not available in this backend")

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
        StateVectorClass, dtype = get_statevector_class_and_precision

        # Check if the module has a Measurements class and get it.
        MeasurementsClass = get_measurements_class(dtype)

        # Check if it has updateData method
        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip("updateData method not available in this backend")

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

        StateVectorClass, dtype = get_statevector_class_and_precision

        # Check if the module has a Measurements class
        MeasurementsClass = get_measurements_class(dtype)

        # Check if the module has an observables submodule with NamedObs class
        NamedObsClass = get_named_obs_class(dtype)

        # Check if it has updateData method
        if not hasattr(StateVectorClass, "updateData"):
            pytest.skip("updateData method not available in this backend")

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

    def test_adjoint_jacobian_jax(
        self,
        get_statevector_class_and_precision,
        get_adjoint_jacobian_class,
        get_named_obs_class,
        get_ops_struct_class,
    ):
        """Test adjoint Jacobian with JAX parameters."""
        # Skip if JAX doesn't support the precision
        StateVectorClass, dtype = get_statevector_class_and_precision

        AdjointJacobianClass = get_adjoint_jacobian_class(dtype)
        NamedObsClass = get_named_obs_class(dtype)
        OpsStructClass = get_ops_struct_class(dtype)

        # Create a state vector
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        param_value = jnp.array(0.5)

        # Apply the operations to the state vector
        sv.RX([0], False, [param_value])

        # Create an observable
        obs = NamedObsClass("PauliZ", [0])

        # Create operations with JAX parameters
        ops_name = ["RX"]
        ops_params = [[param_value]]
        ops_wires = [[0]]
        ops_inverses = [False]
        ops_matrices = [[]]
        ops_controlled_wires = [[]]
        ops_controlled_values = [[]]

        # Create the operations structure
        ops = OpsStructClass(
            ops_name,
            ops_params,
            ops_wires,
            ops_inverses,
            ops_matrices,
            ops_controlled_wires,
            ops_controlled_values,
        )

        # Create the adjoint Jacobian
        adj = AdjointJacobianClass()

        # Calculate the Jacobian with the correct parameter order
        trainable_params = [0]
        result = adj(sv, [obs], ops, trainable_params)

        # The result should be a numpy array with shape (1,)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], -np.sin(param_value))

    def test_adjoint_jacobian_multiple_params_jax(
        self,
        get_statevector_class_and_precision,
        get_adjoint_jacobian_class,
        get_named_obs_class,
        get_ops_struct_class,
    ):
        """Test adjoint Jacobian with multiple JAX parameters."""

        StateVectorClass, dtype = get_statevector_class_and_precision
        AdjointJacobianClass = get_adjoint_jacobian_class(dtype)
        NamedObsClass = get_named_obs_class(dtype)
        OpsStructClass = get_ops_struct_class(dtype)

        # Create a state vector
        num_qubits = 2
        params = jnp.array([0.1, 0.2])

        sv = StateVectorClass(num_qubits)
        sv.RX([0], False, [params[0]])
        sv.RY([0], False, [params[1]])

        # Create an observable
        obs = NamedObsClass("PauliZ", [0])

        # Create operations with JAX parameters
        ops_name = ["RX", "RY"]
        ops_params = [[params[0]], [params[1]]]
        ops_wires = [[0], [0]]
        ops_inverses = [False, False]
        ops_matrices = [[], []]
        ops_controlled_wires = [[], []]
        ops_controlled_values = [[], []]

        # Create the operations structure
        ops = OpsStructClass(
            ops_name,
            ops_params,
            ops_wires,
            ops_inverses,
            ops_matrices,
            ops_controlled_wires,
            ops_controlled_values,
        )

        # Create the adjoint Jacobian
        adj = AdjointJacobianClass()

        # Calculate the Jacobian with the correct parameter order
        trainable_params = [0, 1]
        jacobian = adj(sv, [obs], ops, trainable_params)

        # Verify the shape of the Jacobian
        assert jacobian.shape == (2,)  # 1 observable * 2 parameters = 2 elements

        # Verify the correctness of the Jacobian
        # Expected values calculated analytically
        expected_0 = -np.cos(params[1]) * np.sin(params[0])
        expected_1 = -np.sin(params[1]) * np.cos(params[0])

        assert np.isclose(jacobian[0], expected_0, atol=1e-7)
        assert np.isclose(jacobian[1], expected_1, atol=1e-7)
