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
"""Tests for AdjointJacobian classes in nanobind-based modules."""

import numpy as np
import pytest


class TestAdjointJacobianNanobind:
    """Tests for adjoint Jacobian with nanobind-based bindings."""

    param_value = 0.7
    param_values = [0.5, 0.3]

    @pytest.fixture
    def get_statevector_class(self):
        """Get StateVector class from module based on precision."""

        def _get_class(module, precision="64"):
            class_name = f"StateVectorC{precision}"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            pytest.skip(f"Class {class_name} not available in module")

        return _get_class

    @pytest.fixture
    def get_adjoint_jacobian_class(self):
        """Get AdjointJacobian class from submodule algorithms based on precision."""

        def _get_class(module, precision="64"):
            if hasattr(module.algorithms, f"AdjointJacobianC{precision}"):
                return getattr(module.algorithms, f"AdjointJacobianC{precision}")
            pytest.skip(f"Class AdjointJacobianC{precision} not available in module")

        return _get_class

    @pytest.fixture
    def get_observable_classes(self):
        """Get observable classes from submodule observables based on precision."""

        def _get_classes(module, precision="64"):
            # Check if observables submodule exists
            if not hasattr(module, "observables"):
                pytest.skip("Submodule observables not available in module")

            classes = {}

            # Get NamedObs class
            named_obs_class_name = f"NamedObsC{precision}"
            if hasattr(module.observables, named_obs_class_name):
                classes["named"] = getattr(module.observables, named_obs_class_name)
            else:
                pytest.skip(f"Class {named_obs_class_name} not available in module")

            # Get HermitianObs class
            hermitian_obs_class_name = f"HermitianObsC{precision}"
            if hasattr(module.observables, hermitian_obs_class_name):
                classes["hermitian"] = getattr(module.observables, hermitian_obs_class_name)
            else:
                pytest.skip(f"Class {hermitian_obs_class_name} not available in module")

            return classes

        return _get_classes

    @pytest.fixture
    def get_ops_struct_class(self):
        """Get OpsStruct class from submodule algorithms based on precision."""

        def _get_class(module, precision="64"):
            if hasattr(module.algorithms, f"OpsStructC{precision}"):
                return getattr(module.algorithms, f"OpsStructC{precision}")
            pytest.skip(f"Class OpsStructC{precision} not available in module")

        return _get_class

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_adjoint_jacobian_call(
        self,
        current_nanobind_module,
        precision,
        get_statevector_class,
        get_adjoint_jacobian_class,
        get_observable_classes,
        get_ops_struct_class,
    ):
        """Test calling the adjoint Jacobian directly."""

        module = current_nanobind_module
        StateVectorClass = get_statevector_class(module, precision)
        AdjointJacobianClass = get_adjoint_jacobian_class(module, precision)
        ObservableClasses = get_observable_classes(module, precision)
        OpsStructClass = get_ops_struct_class(module, precision)

        # Create objects:
        num_qubits = 2
        sv = StateVectorClass(num_qubits)

        obs = ObservableClasses["named"]("PauliZ", [0])

        # Create operations structure with RX gate
        ops = OpsStructClass(
            ["RX"],  # Operation names
            [[self.param_value]],  # Parameters
            [[0]],  # Wires
            [False],  # Inverses
            [np.array([])],  # Matrices
            [[]],  # Control wires
            [[]],  # Control values
        )

        # Apply the operations to the state vector
        sv.RX([0], False, [self.param_value])

        adj = AdjointJacobianClass()

        # Use the adjoint jacobian with the correct parameter order
        trainable_params = [0]  # Index of the trainable parameter
        result = adj(sv, [obs], ops, trainable_params)

        # Check the result
        assert isinstance(result, np.ndarray)
        # Raw binding returns a 1D array of size (num_obs * num_params)
        assert result.shape == (1,)  # 1 observable * 1 parameter = 1 element
        assert np.isclose(result[0], -np.sin(self.param_value), atol=1e-7)

    @pytest.mark.parametrize("precision", ["64", "128"])
    @pytest.mark.parametrize(
        "operation, expected_values",
        [
            (
                "RX",
                (0.0, -np.cos(param_value), -np.sin(param_value)),
            ),  # (expected_x, expected_y, expected_z)
            ("RY", (np.cos(param_value), 0.0, -np.sin(param_value))),
            ("RZ", (0.0, 0.0, 0.0)),  # All zeros for RZ when starting from |0⟩
        ],
    )
    def test_adjoint_jacobian_multiple_observables(
        self,
        current_nanobind_module,
        precision,
        operation,
        expected_values,
        get_statevector_class,
        get_adjoint_jacobian_class,
        get_observable_classes,
        get_ops_struct_class,
    ):
        """Test adjoint Jacobian with multiple observables."""

        module = current_nanobind_module

        StateVectorClass = get_statevector_class(module, precision)
        AdjointJacobianClass = get_adjoint_jacobian_class(module, precision)
        ObservableClasses = get_observable_classes(module, precision)
        OpsStructClass = get_ops_struct_class(module, precision)

        # Create objects
        num_qubits = 2
        sv = StateVectorClass(num_qubits)

        # Create observables
        obs_x = ObservableClasses["named"]("PauliX", [0])
        obs_y = ObservableClasses["named"]("PauliY", [0])
        obs_z = ObservableClasses["named"]("PauliZ", [0])

        # Create operations structure
        ops = OpsStructClass(
            [operation],  # Operation names
            [[self.param_value]],  # Parameters
            [[0]],  # Wires
            [False],  # Inverses
            [[]],  # Control wires
            [[]],  # Control values
            [[]],  # Extra parameters
        )

        # Apply the operations to the state vector
        getattr(sv, operation)([0], False, [self.param_value])

        adj = AdjointJacobianClass()

        # Use the adjoint jacobian with all three observables
        trainable_params = [0]
        result = adj(sv, [obs_x, obs_y, obs_z], ops, trainable_params)

        # Check the result shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)  # 3 observables * 1 parameter = 3 elements

        # Check each element
        for i, expected in enumerate(expected_values):
            assert np.isclose(
                result[i], expected, atol=1e-7
            ), f"Expected {expected}, got {result[i]}"

        # Try with observables in reverse order to check if order matters
        result_reversed = adj(sv, [obs_z, obs_y, obs_x], ops, trainable_params)

        # Check the result shape
        assert isinstance(result_reversed, np.ndarray)
        assert result_reversed.shape == (3,)  # 3 observables * 1 parameter = 3 elements

        # Check each element
        for i, expected in enumerate(expected_values[::-1]):
            assert np.isclose(
                result_reversed[i], expected, atol=1e-7
            ), f"Expected {expected}, got {result_reversed[i]}"

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_adjoint_jacobian_multiple_params(
        self,
        current_nanobind_module,
        precision,
        get_statevector_class,
        get_adjoint_jacobian_class,
        get_observable_classes,
        get_ops_struct_class,
    ):
        """Test adjoint Jacobian with multiple parameters."""

        module = current_nanobind_module
        StateVectorClass = get_statevector_class(module, precision)
        AdjointJacobianClass = get_adjoint_jacobian_class(module, precision)
        ObservableClasses = get_observable_classes(module, precision)
        OpsStructClass = get_ops_struct_class(module, precision)

        # Create objects
        num_qubits = 2
        sv = StateVectorClass(num_qubits)

        obs = ObservableClasses["named"]("PauliZ", [0])

        # Create operations structure with multiple gates
        ops = OpsStructClass(
            ["RX", "RY"],  # Operation names
            [[self.param_values[0]], [self.param_values[1]]],  # Parameters
            [[0], [0]],  # Wires
            [False, False],  # Inverses
            [[], []],  # Control wires
            [[], []],  # Control values
            [[], []],  # Extra parameters
        )

        # Apply the operations to the state vector
        sv.RX([0], False, [self.param_values[0]])
        sv.RY([0], False, [self.param_values[1]])

        adj = AdjointJacobianClass()

        # Use the adjoint jacobian with multiple trainable parameters
        trainable_params = [0, 1]  # Indices of trainable parameters
        result = adj(sv, [obs], ops, trainable_params)

        # Check the result
        assert isinstance(result, np.ndarray)
        # Raw binding returns a 1D array of size (num_obs * num_params)
        assert result.shape == (2,)  # 1 observable * 2 parameters = 2 elements

        # Expected values calculated analytically
        expected_0 = -np.cos(self.param_values[1]) * np.sin(self.param_values[0])
        expected_1 = -np.sin(self.param_values[1]) * np.cos(self.param_values[0])

        assert np.isclose(result[0], expected_0, atol=1e-7)
        assert np.isclose(result[1], expected_1, atol=1e-7)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_adjoint_jacobian_hermitian_observable(
        self,
        current_nanobind_module,
        precision,
        get_statevector_class,
        get_adjoint_jacobian_class,
        get_observable_classes,
        get_ops_struct_class,
    ):
        """Test adjoint Jacobian with Hermitian observable."""

        module = current_nanobind_module
        StateVectorClass = get_statevector_class(module, precision)
        AdjointJacobianClass = get_adjoint_jacobian_class(module, precision)
        ObservableClasses = get_observable_classes(module, precision)
        OpsStructClass = get_ops_struct_class(module, precision)

        # Create objects
        num_qubits = 1
        sv = StateVectorClass(num_qubits)

        # Create Hermitian observable (equivalent to PauliZ)
        hermitian_matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        if precision == "64":
            hermitian_matrix = hermitian_matrix.astype(np.complex64)

        obs = ObservableClasses["hermitian"](hermitian_matrix, [0])

        param_value = 0.5

        # Create operations structure
        ops = OpsStructClass(
            ["RX"],  # Operation names
            [[param_value]],  # Parameters
            [[0]],  # Wires
            [False],  # Inverses
            [[]],  # Control wires
            [[]],  # Control values
            [[]],  # Extra parameters
        )

        # Apply the operations to the state vector
        sv.RX([0], False, [param_value])

        adj = AdjointJacobianClass()

        # Use the adjoint jacobian
        trainable_params = [0]
        result = adj(sv, [obs], ops, trainable_params)

        # Check the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

        # Expected value calculated analytically (same as PauliZ)
        expected = -np.sin(param_value)

        assert np.isclose(result[0], expected, atol=1e-7)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_create_ops_list_function(self, current_nanobind_module, precision):
        """Test the create_ops_list function has the same signature as pybind11 version."""
        module = current_nanobind_module

        # Get the create_ops_list function
        create_ops_list_func_name = f"create_ops_listC{precision}"
        if not hasattr(module.algorithms, create_ops_list_func_name):
            pytest.skip(f"Function {create_ops_list_func_name} not available in module")

        create_ops_list = getattr(module.algorithms, create_ops_list_func_name)

        hermitian_matrix = np.array(
            [[1, 0], [0, -1]], dtype=np.complex64 if precision == "64" else np.complex128
        )

        # Create simple test data
        op_names = ["RX"]
        op_params = [[0.5]]
        op_wires = [[0]]
        op_inverses = [False]
        op_matrices = [hermitian_matrix]
        op_controlled_wires = [[]]
        op_controlled_values = [[]]

        # Test that the function works with the expected signature
        ops_struct = create_ops_list(
            op_names,
            op_params,
            op_wires,
            op_inverses,
            op_matrices,  # This is the new argument
            op_controlled_wires,
            op_controlled_values,
        )

        # Check that the result is the expected type
        ops_struct_class_name = f"OpsStructC{precision}"
        assert ops_struct.__class__.__name__ == ops_struct_class_name
