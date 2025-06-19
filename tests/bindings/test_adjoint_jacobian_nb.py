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

from collections.abc import Sequence

import numpy as np
import pytest


class TestAdjointJacobianNanobind:
    """Tests for adjoint Jacobian with nanobind-based bindings."""

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
        if precision == "128" and not hasattr(np, "complex256"):
            pytest.skip("NumPy doesn't support 128-bit complex numbers")

        module = current_nanobind_module
        StateVectorClass = get_statevector_class(module, precision)
        AdjointJacobianClass = get_adjoint_jacobian_class(module, precision)
        ObservableClasses = get_observable_classes(module, precision)
        OpsStructClass = get_ops_struct_class(module, precision)

        # Create objects
        num_qubits = 2
        sv = StateVectorClass(num_qubits)
        obs = ObservableClasses["named"]("PauliZ", [0])

        param_value = 0.5
        ops = OpsStructClass(["RX"], [[param_value]], [[0]], [False], [[]], [[]], [[]])

        adj = AdjointJacobianClass()

        # Use the adjoint jacobian directly with the correct parameter order
        trainable_params = [0]
        result = adj(sv, [obs], ops, trainable_params)

        # Check the result
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], -np.sin(param_value), atol=1e-7)
