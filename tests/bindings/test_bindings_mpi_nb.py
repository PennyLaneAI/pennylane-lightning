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
"""Tests for PennyLane-Lightning MPI Nanobind bindings."""

import importlib
import inspect

import numpy as np
import pytest
from conftest import device_module_name

try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

# Skip if not running with MPI
if MPI_AVAILABLE:
    if MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip("Skipping MPI tests. Run with at least 2 processes.", allow_module_level=True)

# Try to import the MPI module
try:
    module_name = f"pennylane_lightning.{device_module_name}_nb"
    mpi_module = importlib.import_module(module_name)
except ImportError:
    mpi_module = None

# Skip all tests if MPI module is not available
pytestmark = pytest.mark.skipif(
    not MPI_AVAILABLE or mpi_module is None, reason="MPI module not available"
)


def get_module_attributes(module):
    """Get classes and functions from a module."""
    classes = []
    functions = []

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            classes.append(name)
        elif inspect.isfunction(obj):
            functions.append(name)
        elif inspect.ismodule(obj):
            submodule_classes = []
            submodule_functions = []
            for subname, subobj in inspect.getmembers(obj):
                if inspect.isclass(subobj):
                    submodule_classes.append(subname)
                elif inspect.isfunction(subobj):
                    submodule_functions.append(subname)

            if submodule_classes:
                classes.append((name, submodule_classes))
            if submodule_functions:
                functions.append((name, submodule_functions))

    return {"classes": classes, "functions": functions}


class TestMPINanobindBindings:
    """Tests for MPI nanobind-based bindings."""

    @pytest.fixture
    def mpi_module_attributes(self):
        """Get module attributes for the MPI module."""
        return get_module_attributes(mpi_module)

    def test_module_has_mpi_classes(self, mpi_module_attributes):
        """Test if module has MPI-specific classes."""

        # Check for StateVectorMPI classes
        for precision in ["64", "128"]:
            class_name = f"StateVectorMPIC{precision}"
            assert class_name in mpi_module_attributes["classes"], f"{class_name} not found"

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_measurements_mpi(self, precision):
        """Test MeasurementsMPI class."""
        # Skip if the classes don't exist
        sv_class_name = f"StateVectorMPIC{precision}"
        meas_class_name = f"MeasurementsMPIC{precision}"

        if not hasattr(mpi_module, sv_class_name) or not hasattr(mpi_module, meas_class_name):
            pytest.skip(f"Classes {sv_class_name} or {meas_class_name} not available in module")

        StateVectorMPIClass = getattr(mpi_module, sv_class_name)
        MeasurementsMPIClass = getattr(mpi_module, meas_class_name)

        # Create a simple state vector
        num_qubits = 2
        sv = StateVectorMPIClass(num_qubits)

        # Apply Hadamard to create superposition
        sv.applyHadamard(0)

        # Create measurements object
        meas = MeasurementsMPIClass(sv)

        # Test probabilities
        probs = meas.probs([0])

        # Should be approximately [0.5, 0.5] for the first qubit
        assert len(probs) == 2
        assert np.isclose(probs[0], 0.5, atol=1e-6)
        assert np.isclose(probs[1], 0.5, atol=1e-6)

    @pytest.mark.parametrize("precision", ["64", "128"])
    def test_adjoint_jacobian_mpi(self, precision):
        """Test AdjointJacobianMPI class."""
        # Skip if the classes don't exist
        sv_class_name = f"StateVectorMPIC{precision}"
        adj_class_name = f"AdjointJacobianMPIC{precision}"

        if not hasattr(mpi_module, sv_class_name) or not hasattr(
            mpi_module.algorithmsMPI, adj_class_name
        ):
            pytest.skip(f"Classes {sv_class_name} or {adj_class_name} not available in module")

        StateVectorMPIClass = getattr(mpi_module, sv_class_name)
        AdjointJacobianMPIClass = getattr(mpi_module.algorithmsMPI, adj_class_name)

        # Get necessary classes for observables and operations
        NamedObsClass = getattr(mpi_module.observablesMPI, f"NamedObsMPIC{precision}")
        OpsStructClass = getattr(mpi_module.algorithmsMPI, f"OpsStructMPIC{precision}")

        # Create a simple state vector
        num_qubits = 2
        sv = StateVectorMPIClass(num_qubits)

        # Create a simple observable
        obs = NamedObsClass("PauliZ", [0])

        # Create a simple operation
        param_value = 0.5
        ops = OpsStructClass(["RX"], [[param_value]], [[0]], [False], [[]], [[]], [[]])

        # Create the adjoint Jacobian
        adj = AdjointJacobianMPIClass()

        # Calculate the Jacobian
        trainable_params = [0]
        result = adj(sv, [obs], ops, trainable_params)

        # The result should be a numpy array with shape (1,)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], -np.sin(param_value), atol=1e-6)

        # Test the batched method
        batched_result = adj.batched(sv, [obs], ops, trainable_params)
        assert isinstance(batched_result, np.ndarray)
        assert batched_result.shape == (1,)
        assert np.isclose(batched_result[0], -np.sin(param_value), atol=1e-6)
