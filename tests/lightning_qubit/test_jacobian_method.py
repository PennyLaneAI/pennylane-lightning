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


import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA, LightningDevice, device_name  # tested device
from pennylane.devices import DefaultExecutionConfig, DefaultQubit, ExecutionConfig
from pennylane.tape import QuantumScript

if not LightningDevice._new_API:
    pytest.skip(
        "Exclusive tests for new API backends LightningAdjointJacobian class. Skipping.",
        allow_module_level=True,
    )

if device_name == "lightning.gpu":
    pytest.skip("LGPU new API in WIP.  Skipping.", allow_module_level=True)

if device_name == "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestJacobian:
    """Unit tests for the jacobian method with the new device API."""

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
    def process_and_execute(statevector, tape, execute_and_derivatives=False):

        wires = statevector.num_wires
        device = LightningDevice(wires)
        if execute_and_derivatives:
            results, jac = device.simulate_and_jacobian(tape, statevector)
        else:
            results = device.simulate(tape, statevector)
            jac = device.jacobian(tape, statevector)
        return results, jac

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
        self, theta, phi, obs, execute_and_derivatives, lightning_sv
    ):
        """Test that the jacobian is correct when a tape has a single expectation value"""
        if isinstance(obs, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            obs = qml.operation.convert_to_legacy_H(obs)

        qs = QuantumScript(
            [qml.RX(theta, 0), qml.CNOT([0, 1]), qml.RY(phi, 1)],
            [qml.expval(obs)],
            trainable_params=[0, 1],
        )

        statevector = lightning_sv(num_wires=3)
        res, jac = self.process_and_execute(statevector, qs, execute_and_derivatives)

        if isinstance(obs, qml.Hamiltonian):
            qs = QuantumScript(
                qs.operations,
                [qml.expval(qml.Hermitian(qml.matrix(obs), wires=obs.wires))],
                trainable_params=qs.trainable_params,
            )
        expected, expected_jac = self.calculate_reference(
            qs, execute_and_derivatives=execute_and_derivatives
        )

        tol = 1e-5 if statevector.dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)


@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support vjp",
)
class TestVJP:
    """Unit tests for the vjp method with the new device API."""

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
    def process_and_execute(statevector, tape, dy, execute_and_derivatives=False):
        dy = [dy]

        wires = statevector.num_wires
        device = LightningDevice(wires)
        if execute_and_derivatives:
            results, jac = device.simulate_and_vjp(tape, dy, statevector)
        else:
            results = device.simulate(tape, statevector)
            jac = device.vjp(tape, dy, statevector)
        return results, jac

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
    def test_vjp_single_expval(self, theta, phi, obs, execute_and_derivatives, lightning_sv):
        """Test that the VJP is correct when a tape has a single expectation value"""
        if isinstance(obs, qml.ops.LinearCombination) and not qml.operation.active_new_opmath():
            obs = qml.operation.convert_to_legacy_H(obs)

        qs = QuantumScript(
            [qml.RX(theta, 0), qml.CNOT([0, 1]), qml.RY(phi, 1)],
            [qml.expval(obs)],
            trainable_params=[0, 1],
        )

        dy = 1.0
        statevector = lightning_sv(num_wires=3)
        res, jac = self.process_and_execute(
            statevector, qs, dy, execute_and_derivatives=execute_and_derivatives
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

        tol = 1e-5 if statevector.dtype == np.complex64 else 1e-7
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert np.allclose(jac, expected_jac, atol=tol, rtol=0)
