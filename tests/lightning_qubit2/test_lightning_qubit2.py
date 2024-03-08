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

import pytest

import numpy as np
import pennylane as qml
from pennylane_lightning.lightning_qubit import LightningQubit, LightningQubit2
from pennylane_lightning.lightning_qubit.lightning_qubit2 import (
    accepted_observables,
    jacobian,
    simulate,
    simulate_and_jacobian,
    stopping_condition,
    decompose,
    validate_device_wires,
    decompose,
    validate_measurements,
    validate_observables,
    no_sampling,
)
from pennylane_lightning.lightning_qubit._state_vector import LightningStateVector
from pennylane_lightning.lightning_qubit._measurements import LightningMeasurements
from pennylane.devices import DefaultQubit, ExecutionConfig, DefaultExecutionConfig
from pennylane.tape import QuantumScript

from conftest import LightningDevice  # tested device

# TODO: Change this to point to LightningQubit2 after it's available as an installable
# device separate from LightningQubit
if LightningDevice != LightningQubit:
    pytest.skip("Exclusive tests for lightning.qubit. Skipping.", allow_module_level=True)

if not LightningQubit2._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


class TestHelpers:
    """Unit tests for helper functions"""

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


class TestInitialization:
    """Unit tests for LightningQubit2 initialization"""

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

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningQubit2(wires=3, c_dtype=request.param)

    @staticmethod
    def calculate_reference(tape):
        device = DefaultQubit(max_workers=1)
        program, _ = device.preprocess()
        tapes, transf_fn = program([tape])
        results = device.execute(tapes)
        return transf_fn(results)

    @staticmethod
    def process_and_execute(dev, tape):
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    _default_device_options = {
        "c_dtype": np.complex128,
        "batch_obs": False,
        "mcmc": False,
        "kernel_name": None,
        "num_burnin": None,
    }

    @pytest.mark.parametrize(
        "config, expected_config",
        [
            (
                DefaultExecutionConfig,
                ExecutionConfig(
                    grad_on_execution=True,
                    use_device_gradient=False,
                    device_options=_default_device_options,
                ),
            ),
            (
                ExecutionConfig(gradient_method="best"),
                ExecutionConfig(
                    gradient_method="adjoint",
                    grad_on_execution=True,
                    use_device_gradient=True,
                    device_options=_default_device_options,
                ),
            ),
            (
                ExecutionConfig(
                    device_options={
                        "c_dtype": np.complex64,
                        "mcmc": True,
                    }
                ),
                ExecutionConfig(
                    grad_on_execution=True,
                    use_device_gradient=False,
                    device_options={
                        "c_dtype": np.complex64,
                        "batch_obs": False,
                        "mcmc": True,
                        "kernel_name": None,
                        "num_burnin": None,
                    },
                ),
            ),
            (
                ExecutionConfig(
                    gradient_method="backprop", use_device_gradient=False, grad_on_execution=False
                ),
                ExecutionConfig(
                    gradient_method="backprop",
                    use_device_gradient=False,
                    grad_on_execution=False,
                    device_options=_default_device_options,
                ),
            ),
        ],
    )
    def test_preprocess_correct_config_setup(self, config, expected_config):
        """Test that the execution config is set up correctly in preprocess"""
        device = LightningQubit2(wires=2)
        _, new_config = device.preprocess(config)
        del new_config.device_options["rng"]

        assert new_config == expected_config

    def test_preprocess_correct_transforms(self):
        """Test that the transform program returned by preprocess is correct"""
        device = LightningQubit2(wires=2)

        expected_program = qml.transforms.core.TransformProgram()
        expected_program.add_transform(validate_measurements, name="LightningQubit2")
        expected_program.add_transform(no_sampling)
        expected_program.add_transform(
            validate_observables, accepted_observables, name="LightningQubit2"
        )
        expected_program.add_transform(validate_device_wires, device.wires, name="LightningQubit2")
        expected_program.add_transform(qml.defer_measurements, device=device)
        expected_program.add_transform(
            decompose, stopping_condition=stopping_condition, name="LightningQubit2"
        )
        expected_program.add_transform(qml.transforms.broadcast_expand)

        actual_program, _ = device.preprocess(DefaultExecutionConfig)
        assert actual_program == expected_program

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "mp",
        [
            qml.probs(wires=[1, 2]),
            qml.probs(op=qml.PauliZ(2)),
            qml.expval(qml.PauliZ(2)),
            qml.var(qml.PauliX(2)),
        ],
    )
    def test_execute_single_measurement(self, theta, phi, mp, dev):
        """Test that execute returns the correct results with a single measurement."""
        qs = QuantumScript(
            [
                qml.RX(phi, 0),
                qml.CNOT([0, 2]),
                qml.RZ(theta, 1),
                qml.CNOT([1, 2]),
            ],
            [mp],
        )
        res = self.process_and_execute(dev, qs)[0]
        expected = self.calculate_reference(qs)[0]
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "mp1",
        [
            qml.probs(wires=[1, 2]),
            qml.expval(qml.PauliZ(2)),
            qml.var(qml.PauliX(2)),
        ],
    )
    @pytest.mark.parametrize(
        "mp2",
        [
            qml.probs(op=qml.PauliX(2)),
            qml.expval(qml.PauliY(2)),
            qml.var(qml.PauliY(2)),
        ],
    )
    def test_execute_multi_measurement(self, theta, phi, dev, mp1, mp2):
        """Test that execute returns the correct results with multiple measurements."""
        qs = QuantumScript(
            [
                qml.RX(phi, 0),
                qml.CNOT([0, 2]),
                qml.RZ(theta, 1),
                qml.CNOT([1, 2]),
            ],
            [mp1, mp2],
        )
        res = self.process_and_execute(dev, qs)[0]
        expected = self.calculate_reference(qs)[0]
        assert len(res) == 2
        for r, e in zip(res, expected):
            assert np.allclose(r, e)

    @pytest.mark.parametrize("phi", PHI)
    def test_basic_circuit(self, dev, phi):
        """Test execution with a basic circuit without preprocessing"""
        qs = QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.parametrize("phi", PHI)
    def test_execute_tape_batch(self, phi):
        """Test that results are expected with a batch of tapes wiht custom wire labels"""
        device = LightningQubit2(wires=["a", "b", "target", -3])

        ops = [
            qml.PauliX("a"),
            qml.PauliX("b"),
            qml.ctrl(qml.RX(phi, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
        ]

        qs1 = qml.tape.QuantumScript(
            ops,
            [
                qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
            ],
        )

        ops = [qml.Hadamard("a"), qml.IsingXX(phi, wires=("a", "b"))]
        qs2 = qml.tape.QuantumScript(ops, [qml.probs(wires=("a", "b"))])

        results = device.execute((qs1, qs2))

        expected1 = (-qml.math.sin(phi) - 1, 3 * qml.math.cos(phi))
        x1 = qml.math.cos(phi / 2) ** 2 / 2
        x2 = qml.math.sin(phi / 2) ** 2 / 2
        expected2 = x1 * np.array([1, 0, 1, 0]) + x2 * np.array([0, 1, 0, 1])
        expected = (expected1, expected2)

        assert len(results) == len(expected)
        assert len(results[0]) == len(expected[0])
        assert qml.math.allclose(results[0][0], expected[0][0])
        assert qml.math.allclose(results[0][1], expected[0][1])
        assert qml.math.allclose(results[1], expected[1])


class TestDerivatives:
    """Unit tests for calculating derivatives with LightningQubit2"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return LightningQubit2(wires=3, c_dtype=request.param)

    @staticmethod
    def calculate_reference(tape):
        device = DefaultQubit(max_workers=1)
        program, _ = device.preprocess()
        tapes, transf_fn = program([tape])
        results = device.execute(tapes)
        return transf_fn(results)

    @staticmethod
    def process_and_execute(dev, tape):
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    # Test supports derivative + xfail tests

    @pytest.mark.parametrize(
        "config, tape, expected",
        [
            (None, None, True),
            (DefaultExecutionConfig, None, False),
            (ExecutionConfig(gradient_method="backprop"), None, False),
            (
                ExecutionConfig(gradient_method="backprop"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.PauliZ(0))]),
                False,
            ),
            (ExecutionConfig(gradient_method="best"), None, True),
            (ExecutionConfig(gradient_method="adjoint"), None, True),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.PauliZ(0))]),
                True,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.var(qml.PauliZ(0))]),
                False,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.PauliZ(0))], shots=10),
                False,
            ),
        ],
    )
    def test_supports_derivatives(self, dev, config, tape, expected):
        """Test that supports_derivative returns the correct boolean value."""
        assert dev.supports_derivatives(config, tape) == expected

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_derivative_single_expval(self, theta, phi, dev):
        """Test that the jacobian is correct when a tape has a single expectation value"""
        assert True

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_derivative_multi_expval(self, theta, phi, dev):
        """Test that the jacobian is correct when a tape has multiple expectation values"""
        assert True

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_execute_and_derivative_single_expval(self, theta, phi, dev):
        """Test that the result and jacobian is correct when a tape has a single
        expectation value"""
        assert True

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    def test_execute_and_derivative_multi_expval(self, theta, phi, dev):
        """Test that the result and jacobian is correct when a tape has multiple
        expectation values"""
        assert True
