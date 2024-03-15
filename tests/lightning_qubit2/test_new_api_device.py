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
This module contains unit tests for new device API Lightning classes.
"""
# pylint: disable=too-many-arguments

import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA, VARPHI, LightningDevice
from pennylane.devices import DefaultExecutionConfig, DefaultQubit, ExecutionConfig
from pennylane.devices.default_qubit import adjoint_ops
from pennylane.tape import QuantumScript

from pennylane_lightning.lightning_qubit.lightning_qubit2 import (
    _add_adjoint_transforms,
    _supports_adjoint,
    accepted_observables,
    adjoint_measurements,
    decompose,
    no_sampling,
    stopping_condition,
    validate_adjoint_trainable_params,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)

if not LightningDevice._new_API:
    pytest.skip("Exclusive tests for new device API. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture(params=[np.complex64, np.complex128])
def dev(request):
    return LightningDevice(wires=3, c_dtype=request.param)


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

    def test_add_adjoint_transforms(self):
        """Test that the correct transforms are added to the program by _add_adjoint_transforms"""
        expected_program = qml.transforms.core.TransformProgram()

        name = "adjoint + lightning.qubit"
        expected_program.add_transform(no_sampling, name=name)
        expected_program.add_transform(
            decompose,
            stopping_condition=adjoint_ops,
            name=name,
        )
        expected_program.add_transform(validate_observables, accepted_observables, name=name)
        expected_program.add_transform(
            validate_measurements,
            analytic_measurements=adjoint_measurements,
            name=name,
        )
        expected_program.add_transform(qml.transforms.broadcast_expand)
        expected_program.add_transform(validate_adjoint_trainable_params)

        actual_program = qml.transforms.core.TransformProgram()
        _add_adjoint_transforms(actual_program)
        assert actual_program == expected_program

    @pytest.mark.parametrize(
        "circuit, expected",
        [
            (None, True),
            (QuantumScript([], [qml.state()]), False),
            (QuantumScript([qml.RX(1.23, 0)], [qml.expval(qml.Z(0))]), True),
            (QuantumScript([qml.CRot(1.23, 4.56, 7.89, [0, 1])], [qml.expval(qml.Z(0))]), True),
            (QuantumScript([qml.Rot(1.23, 4.56, 7.89, 1)], [qml.var(qml.X(0))]), False),
        ],
    )
    def test_supports_adjoint(self, circuit, expected):
        """Test that _supports_adjoint returns the correct boolean value."""
        assert _supports_adjoint(circuit) == expected


class TestInitialization:
    """Unit tests for device initialization"""

    def test_invalid_num_burnin_error(self):
        """Test that an error is raised when num_burnin is more than number of shots"""
        n_shots = 10
        num_burnin = 11

        with pytest.raises(ValueError, match="Shots should be greater than num_burnin."):
            _ = LightningDevice(wires=2, shots=n_shots, mcmc=True, num_burnin=num_burnin)

    def test_invalid_kernel_name(self):
        """Test that an error is raised when the kernel_name is not "Local" or "NonZeroRandom"."""

        _ = LightningDevice(wires=2, shots=1000, mcmc=True, kernel_name="Local")
        _ = LightningDevice(wires=2, shots=1000, mcmc=True, kernel_name="NonZeroRandom")

        with pytest.raises(
            NotImplementedError, match="only 'Local' and 'NonZeroRandom' kernels are supported"
        ):
            _ = LightningDevice(wires=2, shots=1000, mcmc=True, kernel_name="bleh")


class TestExecution:
    """Unit tests for executing quantum tapes on a device"""

    @staticmethod
    def calculate_reference(tape):
        device = DefaultQubit(max_workers=1)
        program, _ = device.preprocess()
        tapes, transf_fn = program([tape])
        results = device.execute(tapes)
        return transf_fn(results)

    @staticmethod
    def process_and_execute(device, tape):
        program, _ = device.preprocess()
        tapes, transf_fn = program([tape])
        results = device.execute(tapes)
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
        device = LightningDevice(wires=2)
        _, new_config = device.preprocess(config)
        del new_config.device_options["rng"]

        assert new_config == expected_config

    @pytest.mark.parametrize("adjoint", [True, False])
    def test_preprocess(self, adjoint):
        """Test that the transform program returned by preprocess is correct"""
        device = LightningDevice(wires=2)

        expected_program = qml.transforms.core.TransformProgram()
        expected_program.add_transform(validate_measurements, name=device.name)
        expected_program.add_transform(no_sampling)
        expected_program.add_transform(validate_observables, accepted_observables, name=device.name)
        expected_program.add_transform(validate_device_wires, device.wires, name=device.name)
        expected_program.add_transform(qml.defer_measurements, device=device)
        expected_program.add_transform(
            decompose, stopping_condition=stopping_condition, name=device.name
        )
        expected_program.add_transform(qml.transforms.broadcast_expand)

        if adjoint:
            name = "adjoint + lightning.qubit"
            expected_program.add_transform(no_sampling, name=name)
            expected_program.add_transform(
                decompose,
                stopping_condition=adjoint_ops,
                name=name,
            )
            expected_program.add_transform(validate_observables, accepted_observables, name=name)
            expected_program.add_transform(
                validate_measurements,
                analytic_measurements=adjoint_measurements,
                name=name,
            )
            expected_program.add_transform(qml.transforms.broadcast_expand)
            expected_program.add_transform(validate_adjoint_trainable_params)

        gradient_method = "adjoint" if adjoint else None
        config = ExecutionConfig(gradient_method=gradient_method)
        actual_program, _ = device.preprocess(config)
        assert actual_program == expected_program

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "mp",
        [
            qml.probs(wires=[1, 2]),
            qml.probs(op=qml.Z(2)),
            qml.expval(qml.Z(2)),
            qml.var(qml.X(2)),
            qml.expval(qml.sum(qml.X(0), qml.Z(0))),
            qml.expval(qml.Hamiltonian([-0.5, 1.5], [qml.Y(1), qml.X(1)])),
            qml.expval(qml.s_prod(2.5, qml.Z(0))),
            qml.expval(qml.prod(qml.Z(0), qml.X(1))),
            qml.expval(qml.sum(qml.Z(1), qml.X(1))),
            qml.expval(
                qml.SparseHamiltonian(
                    qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]).sparse_matrix(
                        wire_order=[0, 1, 2]
                    ),
                    wires=[0, 1, 2],
                )
            ),
            qml.expval(qml.Projector([1], wires=2)),
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
            qml.expval(qml.Z(2)),
            qml.var(qml.X(2)),
            qml.var(qml.Hermitian(qml.Hadamard.compute_matrix(), 0)),
        ],
    )
    @pytest.mark.parametrize(
        "mp2",
        [
            qml.probs(op=qml.X(2)),
            qml.expval(qml.Y(2)),
            qml.var(qml.Y(2)),
            qml.expval(qml.Hamiltonian([-0.5, 1.5, -1.1], [qml.Y(1), qml.X(1), qml.Z(0)])),
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

    @pytest.mark.parametrize("phi, theta", list(zip(PHI, THETA)))
    @pytest.mark.parametrize("wires", (["a", "b", -3], [0, "target", "other_target"]))
    def test_custom_wires(self, phi, theta, wires):
        """Test execution with custom wires"""
        device = LightningDevice(wires=wires)
        qs = QuantumScript(
            [
                qml.RX(phi, wires[0]),
                qml.RY(theta, wires[2]),
                qml.CNOT([wires[0], wires[1]]),
                qml.CNOT([wires[1], wires[2]]),
            ],
            [qml.expval(qml.Z(wires[0])), qml.expval(qml.Z(wires[2]))],
        )

        result = device.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], np.cos(phi))
        assert np.allclose(result[1], np.cos(phi) * np.cos(theta))


@pytest.mark.parametrize("batch_obs", [True, False])
class TestDerivatives:
    """Unit tests for calculating derivatives with a device"""

    @staticmethod
    def calculate_reference(tape, execute_and_derivatives=False):
        device = DefaultQubit(max_workers=1)
        program, config = device.preprocess(ExecutionConfig(gradient_method="adjoint"))
        tapes, transf_fn = program([tape])

        if execute_and_derivatives:
            results, jac = device.execute_and_compute_derivatives(tapes, config)
        else:
            results = device.execute(tapes, config)
            jac = device.compute_derivatives(tapes, config)
        return transf_fn(results), jac

    @staticmethod
    def process_and_execute(device, tape, execute_and_derivatives=False, obs_batch=False):
        program, config = device.preprocess(
            ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": obs_batch})
        )
        tapes, transf_fn = program([tape])

        if execute_and_derivatives:
            results, jac = device.execute_and_compute_derivatives(tapes, config)
        else:
            results = device.execute(tapes, config)
            jac = device.compute_derivatives(tapes, config)
        return transf_fn(results), jac

    # Test supports derivative + xfail tests

    @pytest.mark.parametrize(
        "config, tape, expected",
        [
            (None, None, True),
            (DefaultExecutionConfig, None, False),
            (ExecutionConfig(gradient_method="backprop"), None, False),
            (
                ExecutionConfig(gradient_method="backprop"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))]),
                False,
            ),
            (ExecutionConfig(gradient_method="best"), None, True),
            (ExecutionConfig(gradient_method="adjoint"), None, True),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))]),
                True,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.var(qml.Z(0))]),
                False,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.state()]),
                False,
            ),
            (
                ExecutionConfig(gradient_method="adjoint"),
                QuantumScript([qml.RX(0.123, 0)], [qml.expval(qml.Z(0))], shots=10),
                False,
            ),
        ],
    )
    def test_supports_derivatives(self, dev, config, tape, expected, batch_obs):
        """Test that supports_derivative returns the correct boolean value."""
        assert dev.supports_derivatives(config, tape) == expected

    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "obs",
        [
            qml.Z(1),
            qml.s_prod(2.5, qml.Z(0)),
            qml.prod(qml.Z(0), qml.X(1)),
            qml.sum(qml.Z(1), qml.X(1)),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.Hermitian(qml.Hadamard.compute_matrix(), 0),
            qml.Projector([1], 1),
            qml.operation.Tensor(qml.Z(0), qml.X(1)),
        ],
    )
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_derivatives_single_expval(
        self, theta, phi, dev, obs, execute_and_derivatives, batch_obs
    ):
        """Test that the jacobian is correct when a tape has a single expectation value"""
        qs = QuantumScript(
            [qml.RX(theta, 0), qml.CNOT([0, 1]), qml.RY(phi, 1)],
            [qml.expval(obs)],
            trainable_params=[0, 1],
        )

        res, jac = self.process_and_execute(
            dev, qs, execute_and_derivatives=execute_and_derivatives, obs_batch=batch_obs
        )
        if isinstance(obs, qml.Hamiltonian):
            qs = QuantumScript(
                qs.operations,
                [qml.expval(qml.Hermitian(qml.matrix(obs), wires=obs.wires))],
                trainable_params=qs.trainable_params,
            )
        expected, expected_jac = self.calculate_reference(
            qs, execute_and_derivatives=execute_and_derivatives
        )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert len(res) == len(jac) == 1
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta, phi, omega", list(zip(THETA, PHI, VARPHI)))
    @pytest.mark.parametrize(
        "obs1",
        [
            qml.Z(1),
            qml.s_prod(2.5, qml.Y(2)),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.operation.Tensor(qml.Z(0), qml.X(1)),
        ],
    )
    @pytest.mark.parametrize(
        "obs2",
        [
            qml.prod(qml.Y(0), qml.X(2)),
            qml.sum(qml.Z(1), qml.X(1)),
            qml.SparseHamiltonian(
                qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]).sparse_matrix(
                    wire_order=[0, 1, 2]
                ),
                wires=[0, 1, 2],
            ),
            qml.Projector([1], wires=2),
        ],
    )
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_derivatives_multi_expval(
        self, theta, phi, omega, dev, obs1, obs2, execute_and_derivatives, batch_obs
    ):
        """Test that the jacobian is correct when a tape has multiple expectation values"""
        qs = QuantumScript(
            [
                qml.RX(theta, 0),
                qml.CNOT([0, 1]),
                qml.RY(phi, 1),
                qml.CNOT([1, 2]),
                qml.RZ(omega, 2),
            ],
            [qml.expval(obs1), qml.expval(obs2)],
            trainable_params=[0, 1, 2],
        )

        res, jac = self.process_and_execute(
            dev, qs, execute_and_derivatives=execute_and_derivatives, obs_batch=batch_obs
        )
        if isinstance(obs1, qml.Hamiltonian):
            qs = QuantumScript(
                qs.operations,
                [qml.expval(qml.Hermitian(qml.matrix(obs1), wires=obs1.wires)), qml.expval(obs2)],
                trainable_params=qs.trainable_params,
            )
        expected, expected_jac = self.calculate_reference(
            qs, execute_and_derivatives=execute_and_derivatives
        )
        res = res[0]
        jac = jac[0]

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert len(res) == len(jac) == 2
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_derivatives_no_trainable_params(self, dev, execute_and_derivatives, batch_obs):
        """Test that the derivatives are empty with there are no trainable parameters."""
        qs = QuantumScript(
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.S(1), qml.T(1)], [qml.expval(qml.Z(1))]
        )
        res, jac = self.process_and_execute(
            dev, qs, execute_and_derivatives=execute_and_derivatives, obs_batch=batch_obs
        )
        expected, _ = self.calculate_reference(qs, execute_and_derivatives=execute_and_derivatives)

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert len(jac) == 1
        assert qml.math.shape(jac[0]) == (0,)

    def test_state_jacobian_not_supported(self, dev, batch_obs):
        """Test that an error is raised if derivatives are requested for state measurement"""
        qs = QuantumScript([qml.RX(1.23, 0)], [qml.state()], trainable_params=[0])
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        with pytest.raises(
            qml.QuantumFunctionError, match="This method does not support statevector return type"
        ):
            _ = dev.compute_derivatives(qs, config)

        with pytest.raises(
            qml.QuantumFunctionError, match="This method does not support statevector return type"
        ):
            _ = dev.execute_and_compute_derivatives(qs, config)

    def test_shots_error_with_derivatives(self, dev, batch_obs):
        """Test that an error is raised if the gradient method is adjoint when the tape has shots"""
        qs = QuantumScript(
            [qml.RX(1.23, 0)], [qml.expval(qml.Z(0))], shots=10, trainable_params=[0]
        )
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})
        program, _ = dev.preprocess(config)

        with pytest.raises(qml.DeviceError, match="Finite shots are not supported"):
            _, _ = program([qs])

    @pytest.mark.parametrize("phi", PHI)
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_derivatives_tape_batch(self, phi, execute_and_derivatives, batch_obs):
        """Test that results are correct when we execute and compute derivatives for a batch of
        tapes."""
        device = LightningDevice(wires=4, batch_obs=batch_obs)

        ops = [
            qml.X(0),
            qml.X(1),
            qml.ctrl(qml.RX(phi, 2), (0, 1, 3), control_values=[1, 1, 0]),
        ]

        qs1 = QuantumScript(
            ops,
            [
                qml.expval(qml.sum(qml.Y(2), qml.Z(1))),
                qml.expval(qml.s_prod(3, qml.Z(2))),
            ],
            trainable_params=[0],
        )

        ops = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        qs2 = QuantumScript(ops, [qml.expval(qml.prod(qml.Z(0), qml.Z(1)))], trainable_params=[0])

        if execute_and_derivatives:
            results, jacs = device.execute_and_compute_derivatives((qs1, qs2))
        else:
            results = device.execute((qs1, qs2))
            jacs = device.compute_derivatives((qs1, qs2))

        # Assert results
        expected1 = (-np.sin(phi) - 1, 3 * np.cos(phi))
        x1 = np.cos(phi / 2) ** 2 / 2
        x2 = np.sin(phi / 2) ** 2 / 2
        expected2 = sum([x1, -x2, -x1, x2])  # zero
        expected = (expected1, expected2)

        assert len(results) == len(expected)
        assert len(results[0]) == len(expected[0])
        assert np.allclose(results[0][0], expected[0][0])
        assert np.allclose(results[0][1], expected[0][1])
        assert np.allclose(results[1], expected[1])

        # Assert derivatives
        expected_jac1 = (-np.cos(phi), -3 * np.sin(phi))
        x1_jac = -np.cos(phi / 2) * np.sin(phi / 2) / 2
        x2_jac = np.sin(phi / 2) * np.cos(phi / 2) / 2
        expected_jac2 = sum([x1_jac, -x2_jac, -x1_jac, x2_jac])  # zero
        expected_jac = (expected_jac1, expected_jac2)

        assert len(jacs) == len(expected_jac)
        assert len(jacs[0]) == len(expected_jac[0])
        assert np.allclose(jacs[0], expected_jac[0])
        assert np.allclose(jacs[1], expected_jac[1])
