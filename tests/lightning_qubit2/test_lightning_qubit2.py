# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains unit tests for the LightningQubit2 class
"""

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice  # tested device
from pennylane.devices import DefaultQubit
from pennylane.tape import QuantumScript

from pennylane_lightning.lightning_qubit import LightningQubit, LightningQubit2
from pennylane_lightning.lightning_qubit._measurements import LightningMeasurements
from pennylane_lightning.lightning_qubit._state_vector import LightningStateVector
from pennylane_lightning.lightning_qubit.lightning_qubit2 import (
    accepted_observables,
    jacobian,
    simulate,
    simulate_and_jacobian,
    stopping_condition,
)

if LightningDevice != LightningQubit:
    pytest.skip("Exclusive tests for lightning.qubit. Skipping.", allow_module_level=True)

if not LightningQubit2._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


class TestHelpers:
    """Unit tests for the simulate function"""

    # Test simulate
    # Test jacobian + xfail tests
    # Test simulate_and_jacobian + xfail tests
    # Test stopping_condition
    # Test accepted_observables

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningQubit2(wires=3, c_dtype=request.param)

    @staticmethod
    def calculate_reference(tape):
        dev = DefaultQubit(max_workers=1)
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    @staticmethod
    def process_and_execute(dev, tape):
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    class DummyOperator(qml.operation.Operation, qml.operation.Observable):
        """Dummy operator"""

        num_wires = 1

    def test_stopping_condition(self):
        """Test that stopping_condition returns whether or not an operation
        is supported by the device."""
        valid_op = qml.RX(1.23, 0)
        invalid_op = self.DummyOperator(0)

        assert stopping_condition(valid_op) is True
        assert stopping_condition(invalid_op) is False

    def test_accepted_observables(self):
        """Test that accepted_observables returns whether or not an observable
        is supported by the device."""
        valid_obs = qml.Projector([0], 0)
        invalid_obs = self.DummyOperator(0)

        assert accepted_observables(valid_obs) is True
        assert accepted_observables(invalid_obs) is False

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_simulate_single_measurement(self, theta, phi, dev):
        """Test that simulate returns the correct results with a single measurement."""
        return

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_simulate_multi_measurement(self, theta, phi, dev):
        """Test that simulate returns the correct results with multiple measurements."""
        return

    @pytest.mark.parametrize("theta", THETA)
    def test_jacobian_returns_zero(self, theta):
        """Test that jacobian always returns zero."""
        tape = QuantumScript([qml.RX(theta, wires=0)], [qml.expval(qml.Z(0))])
        assert np.allclose(jacobian(tape), 0)

    @pytest.mark.xfail
    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_jacobian_single_expval(self, theta, phi, dev):
        """Test that the jacobian is correct when a tape has a single expectation value"""
        return

    @pytest.mark.xfail
    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_jacobian_multi_expval(self, theta, phi, dev):
        """Test that the jacobian is correct when a tape has multiple expectation values"""
        return

    @pytest.mark.xfail
    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_simulate_and_jacobian_single_expval(self, theta, phi, dev):
        """Test that the result and jacobian is correct when a tape has a single
        expectation value"""
        return

    @pytest.mark.xfail
    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_simulate_and_jacobian_multi_expval(self, theta, phi, dev):
        """Test that the result and jacobian is correct when a tape has multiple
        expectation values"""
        return


class TestInitialization:
    """Unit tests for LightningQubit2 initialization"""

    # Test __init__ errors: invalid num_burnin, kernel name

    def test_invalid_num_burnin_error(self):
        """Test that an error is raised when num_burnin is more than number of shots"""
        n_shots = 10
        num_burnin = 11

        with pytest.raises(ValueError, match="Shots should be greater than num_burnin."):
            _ = LightningQubit2(wires=2, shots=n_shots, mcmc=True, num_burnin=num_burnin)

    def test_invalid_kernel_name(self):
        """Test that an error is raised when the kernel_name is not "Local" or "NonZeroRandom"."""

        _ = LightningQubit2(wires=2, shots=1000, mcmc=True, kernel_name="Local")
        _ = LightningQubit2(wires=2, shots=1000, mcmc=True, kernel_name="NonZeroRandom")

        with pytest.raises(
            NotImplementedError, match="only 'Local' and 'NonZeroRandom' kernels are supported"
        ):
            _ = LightningQubit2(wires=2, shots=1000, mcmc=True, kernel_name="bleh")


class TestExecution:
    """Unit tests for executing quantum tapes on LightningQubit2"""

    # Test preprocess
    # Test execute


class TestDerivatives:
    """Unit tests for calculating derivatives with LightningQubit2"""

    # Test supports derivative + xfail tests
    # Test compute_derivatives + xfail tests
    # Test execute_and_compute_derivatives + xfail tests
