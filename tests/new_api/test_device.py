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
# pylint: disable=too-many-arguments, unused-argument

import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA, VARPHI, LightningDevice, device_name
from pennylane.devices import DefaultExecutionConfig, DefaultQubit, ExecutionConfig, MCMConfig
from pennylane.devices.default_qubit import adjoint_ops
from pennylane.tape import QuantumScript

if device_name == "lightning.qubit":
    from pennylane_lightning.lightning_qubit.lightning_qubit import (
        _add_adjoint_transforms,
        _supports_adjoint,
        accepted_observables,
        adjoint_measurements,
        adjoint_observables,
        decompose,
        mid_circuit_measurements,
        no_sampling,
        stopping_condition,
        stopping_condition_shots,
        validate_adjoint_trainable_params,
        validate_device_wires,
        validate_measurements,
        validate_observables,
    )

if device_name == "lightning.tensor":
    from pennylane_lightning.lightning_tensor.lightning_tensor import (
        accepted_observables,
        stopping_condition,
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
        result = True if device_name != "lightning.tensor" else False
        assert accepted_observables(valid_obs) is result
        assert accepted_observables(invalid_obs) is False

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support adjoint_observables",
    )
    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.operation.Tensor(qml.Projector([0], 0), qml.PauliZ(1)), False),
            (qml.prod(qml.Projector([0], 0), qml.PauliZ(1)), False),
            (qml.s_prod(1.5, qml.Projector([0], 0)), False),
            (qml.sum(qml.Projector([0], 0), qml.Hadamard(1)), False),
            (qml.sum(qml.prod(qml.Projector([0], 0), qml.Y(1)), qml.PauliX(1)), False),
            (qml.operation.Tensor(qml.Y(0), qml.Z(1)), True),
            (qml.prod(qml.Y(0), qml.PauliZ(1)), True),
            (qml.s_prod(1.5, qml.Y(1)), True),
            (qml.sum(qml.Y(1), qml.Hadamard(1)), True),
            (qml.X(0), True),
            (qml.Hermitian(np.eye(4), [0, 1]), True),
        ],
    )
    def test_adjoint_observables(self, obs, expected):
        """Test that adjoint_observables returns the expected boolean result for
        a given observable"""
        assert adjoint_observables(obs) == expected

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support adjoint",
    )
    def test_add_adjoint_transforms(self):
        """Test that the correct transforms are added to the program by _add_adjoint_transforms"""
        expected_program = qml.transforms.core.TransformProgram()

        name = "adjoint + lightning.qubit"
        expected_program.add_transform(no_sampling, name=name)
        expected_program.add_transform(
            decompose,
            stopping_condition=adjoint_ops,
            stopping_condition_shots=stopping_condition_shots,
            name=name,
            skip_initial_state_prep=False,
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

    @pytest.mark.skipif(
        device_name == "lightning.tensor",
        reason="lightning.tensor device does not support adjoint",
    )
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


@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support shots or mcmc",
)
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


@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support adjoint_observables",
)
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
        "num_burnin": 0,
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
                        "num_burnin": 0,
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
        expected_program.add_transform(validate_observables, accepted_observables, name=device.name)
        expected_program.add_transform(validate_device_wires, device.wires, name=device.name)
        expected_program.add_transform(
            mid_circuit_measurements, device=device, mcm_config=MCMConfig()
        )
        expected_program.add_transform(
            decompose,
            stopping_condition=device._stopping_condition,
            stopping_condition_shots=stopping_condition_shots,
            skip_initial_state_prep=True,
            name=device.name,
        )
        expected_program.add_transform(qml.transforms.broadcast_expand)

        if adjoint:
            name = "adjoint + lightning.qubit"
            expected_program.add_transform(no_sampling, name=name)
            expected_program.add_transform(
                decompose,
                stopping_condition=adjoint_ops,
                stopping_condition_shots=stopping_condition_shots,
                name=name,
                skip_initial_state_prep=False,
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

    @pytest.mark.parametrize(
        "op, is_trainable",
        [
            (qml.StatePrep([1 / np.sqrt(2), 1 / np.sqrt(2)], wires=0), False),
            (qml.StatePrep(qml.numpy.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), wires=0), True),
            (qml.StatePrep(np.array([1, 0]), wires=0), False),
            (qml.BasisState([1, 1], wires=[0, 1]), False),
            (qml.BasisState(qml.numpy.array([1, 1]), wires=[0, 1]), True),
        ],
    )
    def test_preprocess_state_prep_first_op_decomposition(self, op, is_trainable):
        """Test that state prep ops in the beginning of a tape are decomposed with adjoint
        but not otherwise."""
        tape = qml.tape.QuantumScript([op, qml.RX(1.23, wires=0)], [qml.expval(qml.PauliZ(0))])
        device = LightningDevice(wires=3)

        if is_trainable:
            # Need to decompose twice as the state prep ops we use first decompose into a template
            decomp = op.decomposition()[0].decomposition()
        else:
            decomp = [op]

        config = ExecutionConfig(gradient_method="best" if is_trainable else None)
        program, _ = device.preprocess(config)
        [new_tape], _ = program([tape])
        expected_tape = qml.tape.QuantumScript([*decomp, qml.RX(1.23, wires=0)], tape.measurements)
        assert qml.equal(new_tape, expected_tape)

    @pytest.mark.parametrize(
        "op, decomp_depth",
        [
            (qml.StatePrep([1 / np.sqrt(2), 1 / np.sqrt(2)], wires=0), 1),
            (qml.StatePrep(np.array([1, 0]), wires=0), 1),
            (qml.BasisState([1, 1], wires=[0, 1]), 1),
            (qml.BasisState(qml.numpy.array([1, 1]), wires=[0, 1]), 1),
            (qml.AmplitudeEmbedding([1 / np.sqrt(2), 1 / np.sqrt(2)], wires=0), 2),
            (qml.MottonenStatePreparation([1 / np.sqrt(2), 1 / np.sqrt(2)], wires=0), 0),
        ],
    )
    def test_preprocess_state_prep_middle_op_decomposition(self, op, decomp_depth):
        """Test that state prep ops in the middle of a tape are always decomposed."""
        tape = qml.tape.QuantumScript(
            [qml.RX(1.23, wires=0), op, qml.CNOT([0, 1])], [qml.expval(qml.PauliZ(0))]
        )
        device = LightningDevice(wires=3)

        for _ in range(decomp_depth):
            op = op.decomposition()[0]
        decomp = op.decomposition()

        program, _ = device.preprocess()
        [new_tape], _ = program([tape])
        expected_tape = qml.tape.QuantumScript(
            [qml.RX(1.23, wires=0), *decomp, qml.CNOT([0, 1])], tape.measurements
        )
        assert qml.equal(new_tape, expected_tape)

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "mp",
        [
            qml.probs(wires=[1, 2]),
            qml.probs(op=qml.Z(2)),
            qml.expval(qml.Z(2)),
            qml.var(qml.X(2)),
            qml.expval(qml.X(0) + qml.Z(0)),
            qml.expval(qml.Hamiltonian([-0.5, 1.5], [qml.Y(1), qml.X(1)])),
            qml.expval(2.5 * qml.Z(0)),
            qml.expval(qml.Z(0) @ qml.X(1)),
            qml.expval(qml.operation.Tensor(qml.Z(0), qml.X(1))),
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
        if isinstance(mp.obs, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            mp.obs = qml.operation.convert_to_legacy_H(mp.obs)

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

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
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
        if isinstance(mp2.obs, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            mp2.obs = qml.operation.convert_to_legacy_H(mp2.obs)

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

    @pytest.mark.parametrize(
        "wires, wire_order", [(3, (0, 1, 2)), (("a", "b", "c"), ("a", "b", "c"))]
    )
    def test_probs_different_wire_orders(self, wires, wire_order):
        """Test that measuring probabilities works with custom wires."""

        dev = LightningDevice(wires=wires)

        op = qml.Hadamard(wire_order[1])

        tape = QuantumScript([op], [qml.probs(wires=(wire_order[0], wire_order[1]))])

        res = dev.execute(tape)
        assert qml.math.allclose(res, np.array([0.5, 0.5, 0.0, 0.0]))

        tape2 = QuantumScript([op], [qml.probs(wires=(wire_order[1], wire_order[2]))])
        res2 = dev.execute(tape2)
        assert qml.math.allclose(res2, np.array([0.5, 0.0, 0.5, 0.0]))

        tape3 = QuantumScript([op], [qml.probs(wires=(wire_order[1], wire_order[0]))])
        res3 = dev.execute(tape3)
        assert qml.math.allclose(res3, np.array([0.5, 0.0, 0.5, 0.0]))


@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support derivatives",
)
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

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "obs",
        [
            qml.Z(1),
            2.5 * qml.Z(0),
            qml.Z(0) @ qml.X(1),
            qml.Z(1) + qml.X(1),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.Hermitian(qml.Hadamard.compute_matrix(), 0),
            qml.Projector([1], 1),
        ],
    )
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_derivatives_single_expval(
        self, theta, phi, dev, obs, execute_and_derivatives, batch_obs
    ):
        """Test that the jacobian is correct when a tape has a single expectation value"""
        if isinstance(obs, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            obs = qml.operation.convert_to_legacy_H(obs)

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
            2.5 * qml.Y(2),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.Z(0) @ qml.X(1),
        ],
    )
    @pytest.mark.parametrize(
        "obs2",
        [
            qml.Y(0) @ qml.X(2),
            qml.Z(1) + qml.X(1),
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
        if isinstance(obs1, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            obs1 = qml.operation.convert_to_legacy_H(obs1)
        if isinstance(obs2, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            obs2 = qml.operation.convert_to_legacy_H(obs2)

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

    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    @pytest.mark.parametrize(
        "state_prep, params, wires",
        [
            (qml.BasisState, [1, 1], [0, 1]),
            (qml.StatePrep, [0.0, 0.0, 0.0, 1.0], [0, 1]),
            (qml.StatePrep, qml.numpy.array([0.0, 1.0]), [1]),
        ],
    )
    @pytest.mark.parametrize(
        "trainable_params",
        [(0, 1, 2), (1, 2)],
    )
    def test_state_prep_ops(
        self, dev, state_prep, params, wires, execute_and_derivatives, batch_obs, trainable_params
    ):
        """Test that a circuit containing state prep operations is differentiated correctly."""
        qs = QuantumScript(
            [state_prep(params, wires), qml.RX(1.23, 0), qml.CNOT([0, 1]), qml.RX(4.56, 1)],
            [qml.expval(qml.PauliZ(1))],
        )

        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})
        program, new_config = dev.preprocess(config)
        tapes, fn = program([qs])
        tapes[0].trainable_params = trainable_params
        if execute_and_derivatives:
            res, jac = dev.execute_and_compute_derivatives(tapes, new_config)
            res = fn(res)
        else:
            res, jac = (
                fn(dev.execute(tapes, new_config)),
                dev.compute_derivatives(tapes, new_config),
            )

        dev_ref = DefaultQubit(max_workers=1)
        config = ExecutionConfig(gradient_method="adjoint")
        program, new_config = dev_ref.preprocess(config)
        tapes, fn = program([qs])
        tapes[0].trainable_params = trainable_params
        if execute_and_derivatives:
            expected, expected_jac = dev_ref.execute_and_compute_derivatives(tapes, new_config)
            expected = fn(expected)
        else:
            expected, expected_jac = (
                fn(dev_ref.execute(tapes, new_config)),
                dev_ref.compute_derivatives(tapes, new_config),
            )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    def test_state_jacobian_not_supported(self, dev, batch_obs):
        """Test that an error is raised if derivatives are requested for state measurement"""
        qs = QuantumScript([qml.RX(1.23, 0)], [qml.state()], trainable_params=[0])
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
        ):
            _ = dev.compute_derivatives(qs, config)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation method does not support measurement StateMP.",
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


@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support vjp",
)
@pytest.mark.parametrize("batch_obs", [True, False])
class TestVJP:
    """Unit tests for VJP computation with the new device API."""

    @staticmethod
    def calculate_reference(tape, dy, execute_and_derivatives=False):
        device = DefaultQubit(max_workers=1)
        program, config = device.preprocess(ExecutionConfig(gradient_method="adjoint"))
        tapes, transf_fn = program([tape])
        dy = [dy]

        if execute_and_derivatives:
            results, jac = device.execute_and_compute_vjp(tapes, dy, config)
        else:
            results = device.execute(tapes, config)
            jac = device.compute_vjp(tapes, dy, config)
        return transf_fn(results), jac

    @staticmethod
    def process_and_execute(device, tape, dy, execute_and_derivatives=False, obs_batch=False):
        program, config = device.preprocess(
            ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": obs_batch})
        )
        tapes, transf_fn = program([tape])
        dy = [dy]

        if execute_and_derivatives:
            results, jac = device.execute_and_compute_vjp(tapes, dy, config)
        else:
            results = device.execute(tapes, config)
            jac = device.compute_vjp(tapes, dy, config)
        return transf_fn(results), jac

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
    def test_supports_vjp(self, dev, config, tape, expected, batch_obs):
        """Test that supports_vjp returns the correct boolean value."""
        assert dev.supports_vjp(config, tape) == expected

    @pytest.mark.usefixtures("use_legacy_and_new_opmath")
    @pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
    @pytest.mark.parametrize(
        "obs",
        [
            qml.Z(1),
            2.5 * qml.Z(0),
            qml.Z(0) @ qml.X(1),
            qml.Z(1) + qml.X(1),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.Hermitian(qml.Hadamard.compute_matrix(), 0),
            qml.Projector([1], 1),
        ],
    )
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_vjp_single_expval(self, theta, phi, dev, obs, execute_and_derivatives, batch_obs):
        """Test that the VJP is correct when a tape has a single expectation value"""
        if isinstance(obs, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            obs = qml.operation.convert_to_legacy_H(obs)

        qs = QuantumScript(
            [qml.RX(theta, 0), qml.CNOT([0, 1]), qml.RY(phi, 1)],
            [qml.expval(obs)],
            trainable_params=[0, 1],
        )

        dy = 1.0
        res, jac = self.process_and_execute(
            dev, qs, dy, execute_and_derivatives=execute_and_derivatives, obs_batch=batch_obs
        )
        if isinstance(obs, qml.Hamiltonian):
            qs = QuantumScript(
                qs.operations,
                [qml.expval(qml.Hermitian(qml.matrix(obs), wires=obs.wires))],
                trainable_params=qs.trainable_params,
            )
        expected, expected_jac = self.calculate_reference(
            qs, dy, execute_and_derivatives=execute_and_derivatives
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
            2.5 * qml.Y(2),
            qml.Hamiltonian([-1.0, 1.5], [qml.Z(1), qml.X(1)]),
            qml.Z(0) @ qml.X(1),
        ],
    )
    @pytest.mark.parametrize(
        "obs2",
        [
            qml.Y(0) @ qml.X(2),
            qml.Z(1) + qml.X(1),
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
    def test_vjp_multi_expval(
        self, theta, phi, omega, dev, obs1, obs2, execute_and_derivatives, batch_obs
    ):
        """Test that the VJP is correct when a tape has multiple expectation values"""
        if isinstance(obs1, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            obs1 = qml.operation.convert_to_legacy_H(obs1)
        if isinstance(obs2, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            obs2 = qml.operation.convert_to_legacy_H(obs2)

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
        dy = (1.0, 2.0)

        res, jac = self.process_and_execute(
            dev, qs, dy, execute_and_derivatives=execute_and_derivatives, obs_batch=batch_obs
        )
        if isinstance(obs1, qml.Hamiltonian):
            qs = QuantumScript(
                qs.operations,
                [qml.expval(qml.Hermitian(qml.matrix(obs1), wires=obs1.wires)), qml.expval(obs2)],
                trainable_params=qs.trainable_params,
            )
        expected, expected_jac = self.calculate_reference(
            qs, dy, execute_and_derivatives=execute_and_derivatives
        )
        res = res[0]
        jac = jac[0]

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert len(res) == 2
        assert len(jac) == 3
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_vjp_no_trainable_params(self, dev, execute_and_derivatives, batch_obs):
        """Test that the VJP is empty with there are no trainable parameters."""
        qs = QuantumScript(
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.S(1), qml.T(1)], [qml.expval(qml.Z(1))]
        )
        dy = 1.0

        res, jac = self.process_and_execute(
            dev, qs, dy, execute_and_derivatives=execute_and_derivatives, obs_batch=batch_obs
        )
        expected, _ = self.calculate_reference(
            qs, dy, execute_and_derivatives=execute_and_derivatives
        )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert len(jac) == 1
        assert qml.math.shape(jac[0]) == (0,)

    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    @pytest.mark.parametrize(
        "state_prep, params, wires",
        [
            (qml.BasisState, [1, 1], [0, 1]),
            (qml.StatePrep, [0.0, 0.0, 0.0, 1.0], [0, 1]),
            (qml.StatePrep, qml.numpy.array([0.0, 1.0]), [1]),
        ],
    )
    @pytest.mark.parametrize(
        "trainable_params",
        [(0, 1, 2), (1, 2)],
    )
    def test_state_prep_ops(
        self, dev, state_prep, params, wires, execute_and_derivatives, batch_obs, trainable_params
    ):
        """Test that a circuit containing state prep operations is differentiated correctly."""
        qs = QuantumScript(
            [state_prep(params, wires), qml.RX(1.23, 0), qml.CNOT([0, 1]), qml.RX(4.56, 1)],
            [qml.expval(qml.PauliZ(1))],
        )
        dy = [1.0]

        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})
        program, new_config = dev.preprocess(config)
        tapes, fn = program([qs])
        tapes[0].trainable_params = trainable_params
        if execute_and_derivatives:
            res, jac = dev.execute_and_compute_vjp(tapes, dy, new_config)
            res = fn(res)
        else:
            res, jac = (
                fn(dev.execute(tapes, new_config)),
                dev.compute_vjp(tapes, dy, new_config),
            )

        dev_ref = DefaultQubit(max_workers=1)
        config = ExecutionConfig(gradient_method="adjoint")
        program, new_config = dev_ref.preprocess(config)
        tapes, fn = program([qs])
        tapes[0].trainable_params = trainable_params
        if execute_and_derivatives:
            expected, expected_jac = dev_ref.execute_and_compute_vjp(tapes, dy, new_config)
            expected = fn(expected)
        else:
            expected, expected_jac = (
                fn(dev_ref.execute(tapes, new_config)),
                dev_ref.compute_vjp(tapes, dy, new_config),
            )

        tol = 1e-5 if dev.c_dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)

    def test_state_vjp_not_supported(self, dev, batch_obs):
        """Test that an error is raised if VJP are requested for state measurement"""
        qs = QuantumScript([qml.RX(1.23, 0)], [qml.state()], trainable_params=[0])
        config = ExecutionConfig(gradient_method="adjoint", device_options={"batch_obs": batch_obs})
        dy = 1.0

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation does not support State measurements",
        ):
            _ = dev.compute_vjp(qs, dy, config)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Adjoint differentiation does not support State measurements",
        ):
            _ = dev.execute_and_compute_vjp(qs, dy, config)

    @pytest.mark.parametrize("phi", PHI)
    @pytest.mark.parametrize("execute_and_derivatives", [True, False])
    def test_vjp_tape_batch(self, phi, execute_and_derivatives, batch_obs):
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
        dy = [(1.5, 2.5), 1.0]

        if execute_and_derivatives:
            results, jacs = device.execute_and_compute_vjp((qs1, qs2), dy)
        else:
            results = device.execute((qs1, qs2))
            jacs = device.compute_vjp((qs1, qs2), dy)

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
        expected_jac1 = -1.5 * np.cos(phi) - 2.5 * 3 * np.sin(phi)
        x1_jac = -np.cos(phi / 2) * np.sin(phi / 2) / 2
        x2_jac = np.sin(phi / 2) * np.cos(phi / 2) / 2
        expected_jac2 = sum([x1_jac, -x2_jac, -x1_jac, x2_jac])  # zero
        expected_jac = (expected_jac1, expected_jac2)

        assert len(jacs) == len(expected_jac) == 2
        assert np.allclose(jacs[0], expected_jac[0])
        assert np.allclose(jacs[1], expected_jac[1])
