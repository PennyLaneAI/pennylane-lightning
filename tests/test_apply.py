# Copyright 2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane_lightning.LightningQubit` device.
"""
# pylint: disable=protected-access,cell-var-from-loop
import math

import numpy as np
import pennylane as qml
import pytest
from pennylane import DeviceError

from pennylane_lightning import LightningQubit
from pennylane_lightning.lightning_qubit import CPP_BINARY_AVAILABLE

U2 = np.array(
    [
        [
            -0.07843244 - 3.57825948e-01j,
            0.71447295 - 5.38069384e-02j,
            0.20949966 + 6.59100734e-05j,
            -0.50297381 + 2.35731613e-01j,
        ],
        [
            -0.26626692 + 4.53837083e-01j,
            0.27771991 - 2.40717436e-01j,
            0.41228017 - 1.30198687e-01j,
            0.01384490 - 6.33200028e-01j,
        ],
        [
            -0.69254712 - 2.56963068e-02j,
            -0.15484858 + 6.57298384e-02j,
            -0.53082141 + 7.18073414e-02j,
            -0.41060450 - 1.89462315e-01j,
        ],
        [
            -0.09686189 - 3.15085273e-01j,
            -0.53241387 - 1.99491763e-01j,
            0.56928622 + 3.97704398e-01j,
            -0.28671074 - 6.01574497e-02j,
        ],
    ]
)


U_toffoli = np.diag([1 for i in range(8)])
U_toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])

U_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

U_cswap = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)


H = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


class TestApply:
    """Tests that operations of certain operations are applied correctly or
    that the proper errors are raised.
    """

    from pennylane_lightning import LightningQubit as lq

    @pytest.mark.skipif(
        not hasattr(np, "complex256"), reason="Numpy only defines complex256 in Linux-like system"
    )
    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_apply_operation_raise_type_error(self, qubit_device_1_wire):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        with pytest.raises(TypeError, match="Unsupported complex Type: complex256"):
            qubit_device_1_wire._state = np.array([1, 0]).astype(np.complex256)
            qubit_device_1_wire.apply([qml.PauliX(wires=[0])])

    test_data_no_parameters = [
        (qml.PauliX, [1, 0], np.array([0, 1])),
        (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        (qml.PauliY, [1, 0], [0, 1j]),
        (qml.PauliY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1j / math.sqrt(2), 1j / math.sqrt(2)]),
        (qml.PauliZ, [1, 0], [1, 0]),
        (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]),
        (qml.S, [1, 0], [1, 0]),
        (qml.S, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1j / math.sqrt(2)]),
        (qml.T, [1, 0], [1, 0]),
        (
            qml.T,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), np.exp(1j * np.pi / 4) / math.sqrt(2)],
        ),
        (qml.Hadamard, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        (qml.Hadamard, [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_apply_operation_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output, C
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        qubit_device_1_wire._state = np.array(input).astype(C)
        qubit_device_1_wire.apply([operation(wires=[0])])

        assert np.allclose(qubit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)

    test_data_two_wires_no_parameters = [
        (qml.CNOT, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.CNOT, [0, 0, 1, 0], [0, 0, 0, 1]),
        (
            qml.CNOT,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
        ),
        (qml.SWAP, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.SWAP, [0, 0, 1, 0], [0, 1, 0, 0]),
        (
            qml.SWAP,
            [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
            [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0],
        ),
        (qml.CZ, [1, 0, 0, 0], [1, 0, 0, 0]),
        (qml.CZ, [0, 0, 0, 1], [0, 0, 0, -1]),
        (
            qml.CZ,
            [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)],
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        ),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_two_wires_no_parameters)
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_apply_operation_two_wires_no_parameters(
        self, qubit_device_2_wires, tol, operation, input, expected_output, C
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have no parameters."""
        qubit_device_2_wires._state = np.array(input).reshape(2 * [2]).astype(C)
        qubit_device_2_wires.apply([operation(wires=[0, 1])])

        assert np.allclose(qubit_device_2_wires.state, np.array(expected_output), atol=tol, rtol=0)

    test_data_three_wires_no_parameters = [
        (qml.CSWAP, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
        (qml.CSWAP, [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]),
        (qml.CSWAP, [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0]),
        (qml.Toffoli, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
        (qml.Toffoli, [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
        (qml.Toffoli, [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]),
        (qml.Toffoli, [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_parameters)
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_apply_operation_three_wires_no_parameters(
        self, qubit_device_3_wires, tol, operation, input, expected_output, C
    ):
        """Tests that applying an operation yields the expected output state for three wire
        operations that have no parameters."""

        qubit_device_3_wires._state = np.array(input).reshape(3 * [2]).astype(C)
        qubit_device_3_wires.apply([operation(wires=[0, 1, 2])])

        assert np.allclose(qubit_device_3_wires.state, np.array(expected_output), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "operation,expected_output,par",
        [
            (qml.BasisState, [0, 0, 1, 0], [1, 0]),
            (qml.BasisState, [0, 0, 1, 0], [1, 0]),
            (qml.BasisState, [0, 0, 0, 1], [1, 1]),
            (qml.QubitStateVector, [0, 0, 1, 0], [0, 0, 1, 0]),
            (qml.QubitStateVector, [0, 0, 1, 0], [0, 0, 1, 0]),
            (qml.QubitStateVector, [0, 0, 0, 1], [0, 0, 0, 1]),
            (
                qml.QubitStateVector,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
            ),
            (
                qml.QubitStateVector,
                [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
                [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
            ),
        ],
    )
    def test_apply_operation_state_preparation(
        self, qubit_device_2_wires, tol, operation, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        par = np.array(par)
        qubit_device_2_wires.reset()
        qubit_device_2_wires.apply([operation(par, wires=[0, 1])])

        assert np.allclose(qubit_device_2_wires.state, np.array(expected_output), atol=tol, rtol=0)

    """ operation,input,expected_output,par """
    test_data_single_wire_with_parameters = [
        (qml.PhaseShift, [1, 0], [1, 0], [math.pi / 2]),
        (qml.PhaseShift, [0, 1], [0, 1j], [math.pi / 2]),
        (
            qml.PhaseShift,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 1 / 2 + 1j / 2],
            [math.pi / 4],
        ),
        (qml.RX, [1, 0], [1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RX, [1, 0], [0, -1j], [math.pi]),
        (
            qml.RX,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 - 1j / 2, 1 / 2 - 1j / 2],
            [math.pi / 2],
        ),
        (qml.RY, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.RY, [1, 0], [0, 1], [math.pi]),
        (qml.RY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [math.pi / 2]),
        (qml.RZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.RZ, [0, 1], [0, 1j], [math.pi]),
        (
            qml.RZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
            [math.pi / 2],
        ),
        (qml.MultiRZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
        (qml.MultiRZ, [0, 1], [0, 1j], [math.pi]),
        (
            qml.MultiRZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
            [math.pi / 2],
        ),
        (qml.Rot, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2, 0, 0]),
        (qml.Rot, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
        (
            qml.Rot,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
            [0, 0, math.pi / 2],
        ),
        (
            qml.Rot,
            [1, 0],
            [-1j / math.sqrt(2), -1 / math.sqrt(2)],
            [math.pi / 2, -math.pi / 2, math.pi / 2],
        ),
        (
            qml.Rot,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / 2 + 1j / 2, -1 / 2 + 1j / 2],
            [-math.pi / 2, math.pi, math.pi],
        ),
    ]

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_single_wire_with_parameters
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_apply_operation_single_wire_with_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output, par, C
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        qubit_device_1_wire._state = np.array(input).astype(C)

        qubit_device_1_wire.apply([operation(*par, wires=[0])])

        assert np.allclose(qubit_device_1_wire.state, np.array(expected_output), atol=tol, rtol=0)

    """ operation,input,expected_output,par """
    test_data_two_wires_with_parameters = [
        (qml.IsingXX, [1, 0, 0, 0], [1 / math.sqrt(2), 0, 0, -1j / math.sqrt(2)], [math.pi / 2]),
        (
            qml.IsingXX,
            [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
            [-0.5j, 0.5, -0.5j, 0.5],
            [math.pi / 2],
        ),
        (qml.IsingYY, [1, 0, 0, 0], [1 / math.sqrt(2), 0, 0, 1j / math.sqrt(2)], [math.pi / 2]),
        (
            qml.IsingYY,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [0.5 + 0.5j, 0, 0, 0.5 + 0.5j],
            [math.pi / 2],
        ),
        (qml.IsingZZ, [1, 0, 0, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0, 0, 0], [math.pi / 2]),
        (
            qml.IsingZZ,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [0.5 - 0.5j, 0, 0, 0.5 - 0.5j],
            [math.pi / 2],
        ),
        (qml.MultiRZ, [1, 0, 0, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0, 0, 0], [math.pi / 2]),
        (
            qml.MultiRZ,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [0.5 - 0.5j, 0, 0, 0.5 - 0.5j],
            [math.pi / 2],
        ),
        (qml.CRX, [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
        (qml.CRX, [0, 0, 0, 1], [0, 0, -1j, 0], [math.pi]),
        (
            qml.CRX,
            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
            [0, 1 / math.sqrt(2), 1 / 2, -1j / 2],
            [math.pi / 2],
        ),
        (qml.CRY, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
        (qml.CRY, [0, 0, 0, 1], [0, 0, -1, 0], [math.pi]),
        (
            qml.CRY,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [math.pi / 2],
        ),
        (qml.CRZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2]),
        (qml.CRZ, [0, 0, 0, 1], [0, 0, 0, 1j], [math.pi]),
        (
            qml.CRZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [math.pi / 2],
        ),
        (
            qml.CRot,
            [0, 0, 0, 1],
            [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
            [math.pi / 2, 0, 0],
        ),
        (qml.CRot, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
        (
            qml.CRot,
            [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
            [0, 0, 1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
            [0, 0, math.pi / 2],
        ),
        (
            qml.CRot,
            [0, 0, 0, 1],
            [0, 0, 1 / math.sqrt(2), 1j / math.sqrt(2)],
            [math.pi / 2, -math.pi / 2, math.pi / 2],
        ),
        (
            qml.CRot,
            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
            [0, 1 / math.sqrt(2), 0, -1 / 2 + 1j / 2],
            [-math.pi / 2, math.pi, math.pi],
        ),
        (
            qml.ControlledPhaseShift,
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [math.pi / 2],
        ),
        (
            qml.ControlledPhaseShift,
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [math.pi / 2],
        ),
        (
            qml.ControlledPhaseShift,
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [math.pi / 2],
        ),
        (
            qml.ControlledPhaseShift,
            [0, 0, 0, 1],
            [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
            [math.pi / 4],
        ),
        (
            qml.ControlledPhaseShift,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 1 / math.sqrt(2), 1 / math.sqrt(2), 1 / 2 + 1j / 2],
            [math.pi / 4],
        ),
    ]

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_two_wires_with_parameters
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_apply_operation_two_wires_with_parameters(
        self, qubit_device_2_wires, tol, operation, input, expected_output, par, C
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        qubit_device_2_wires._state = np.array(input).reshape(2 * [2]).astype(C)
        qubit_device_2_wires.apply([operation(*par, wires=[0, 1])])

        assert np.allclose(qubit_device_2_wires.state, np.array(expected_output), atol=tol, rtol=0)

    def test_apply_errors_qubit_state_vector(self, qubit_device_2_wires):
        """Test that apply fails for incorrect state preparation, and > 2 qubit gates"""
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            qubit_device_2_wires.apply([qml.QubitStateVector(np.array([1, -1]), wires=[0])])

        with pytest.raises(ValueError, match=r"State vector must be of length 2\*\*wires."):
            p = np.array([1, 0, 1, 1, 0]) / np.sqrt(3)
            qubit_device_2_wires.apply([qml.QubitStateVector(p, wires=[0, 1])])

        with pytest.raises(
            DeviceError,
            match="Operation QubitStateVector cannot be used after other Operations have already been applied ",
        ):
            qubit_device_2_wires.reset()
            qubit_device_2_wires.apply(
                [qml.RZ(0.5, wires=[0]), qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[0, 1])]
            )

    def test_apply_errors_basis_state(self, qubit_device_2_wires):
        with pytest.raises(
            ValueError, match="BasisState parameter must consist of 0 or 1 integers."
        ):
            qubit_device_2_wires.apply([qml.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])])

        with pytest.raises(
            ValueError, match="BasisState parameter and wires must be of equal length."
        ):
            qubit_device_2_wires.apply([qml.BasisState(np.array([0, 1]), wires=[0])])

        with pytest.raises(
            DeviceError,
            match="Operation BasisState cannot be used after other Operations have already been applied ",
        ):
            qubit_device_2_wires.reset()
            qubit_device_2_wires.apply(
                [qml.RZ(0.5, wires=[0]), qml.BasisState(np.array([1, 1]), wires=[0, 1])]
            )


class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            (qml.PauliX, [1 / math.sqrt(2), -1 / math.sqrt(2)], -1),
            (qml.PauliX, [1, 0], 0),
            (qml.PauliY, [1 / math.sqrt(2), 1j / math.sqrt(2)], 1),
            (qml.PauliY, [1 / math.sqrt(2), -1j / math.sqrt(2)], -1),
            (qml.PauliY, [1, 0], 0),
            (qml.PauliZ, [1, 0], 1),
            (qml.PauliZ, [0, 1], -1),
            (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            (qml.Hadamard, [1, 0], 1 / math.sqrt(2)),
            (qml.Hadamard, [0, 1], -1 / math.sqrt(2)),
            (qml.Hadamard, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / math.sqrt(2)),
            (qml.Identity, [1, 0], 1),
            (qml.Identity, [0, 1], 1),
            (qml.Identity, [1 / math.sqrt(2), -1 / math.sqrt(2)], 1),
        ],
    )
    def test_expval_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.expval(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_expval_estimate(self):
        """Test that the expectation value is not analytically calculated"""
        dev = qml.device("lightning.qubit", wires=1, shots=3)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliX(0))

        expval = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance an an analytically calculated one
        assert expval != 0.0


class TestVar:
    """Tests that variances are properly calculated."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            (qml.PauliX, [1 / math.sqrt(2), -1 / math.sqrt(2)], 0),
            (qml.PauliX, [1, 0], 1),
            (qml.PauliY, [1 / math.sqrt(2), 1j / math.sqrt(2)], 0),
            (qml.PauliY, [1 / math.sqrt(2), -1j / math.sqrt(2)], 0),
            (qml.PauliY, [1, 0], 1),
            (qml.PauliZ, [1, 0], 0),
            (qml.PauliZ, [0, 1], 0),
            (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            (qml.Hadamard, [1, 0], 1 / 2),
            (qml.Hadamard, [0, 1], 1 / 2),
            (qml.Hadamard, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / 2),
            (qml.Identity, [1, 0], 0),
            (qml.Identity, [0, 1], 0),
            (qml.Identity, [1 / math.sqrt(2), -1 / math.sqrt(2)], 0),
        ],
    )
    def test_var_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that variances are properly calculated for single-wire observables without parameters."""

        obs = operation(wires=[0])

        qubit_device_1_wire.reset()
        qubit_device_1_wire.apply(
            [qml.QubitStateVector(np.array(input), wires=[0])], obs.diagonalizing_gates()
        )
        res = qubit_device_1_wire.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_var_estimate(self):
        """Test that the variance is not analytically calculated"""

        dev = qml.device("lightning.qubit", wires=1, shots=3)

        @qml.qnode(dev)
        def circuit():
            return qml.var(qml.PauliX(0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance and an analytically calculated one
        assert var != 1.0


class TestSample:
    """Tests that samples are properly calculated."""

    def test_sample_dimensions(self, qubit_device_2_wires):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        qubit_device_2_wires.reset()

        qubit_device_2_wires.apply([qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])])

        qubit_device_2_wires.shots = 10
        qubit_device_2_wires._wires_measured = {0}
        qubit_device_2_wires._samples = qubit_device_2_wires.generate_samples()
        s1 = qubit_device_2_wires.sample(qml.PauliZ(wires=[0]))
        assert np.array_equal(s1.shape, (10,))

        qubit_device_2_wires.reset()
        qubit_device_2_wires.shots = 12
        qubit_device_2_wires._wires_measured = {1}
        qubit_device_2_wires._samples = qubit_device_2_wires.generate_samples()
        s2 = qubit_device_2_wires.sample(qml.PauliZ(wires=[1]))
        assert np.array_equal(s2.shape, (12,))

        qubit_device_2_wires.reset()
        qubit_device_2_wires.shots = 17
        qubit_device_2_wires._wires_measured = {0, 1}
        qubit_device_2_wires._samples = qubit_device_2_wires.generate_samples()
        s3 = qubit_device_2_wires.sample(qml.PauliX(0) @ qml.PauliZ(1))
        assert np.array_equal(s3.shape, (17,))

    def test_sample_values(self, qubit_device_2_wires, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        qubit_device_2_wires.reset()

        qubit_device_2_wires.shots = 1000
        qubit_device_2_wires.apply([qml.RX(1.5708, wires=[0])])
        qubit_device_2_wires._wires_measured = {0}
        qubit_device_2_wires._samples = qubit_device_2_wires.generate_samples()

        s1 = qubit_device_2_wires.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)


class TestLightningQubitIntegration:
    """Integration tests for lightning.qubit. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    from pennylane_lightning import LightningQubit as lq

    def test_load_default_qubit_device(self):
        """Test that the default plugin loads correctly"""

        dev = qml.device("lightning.qubit", wires=2)
        assert dev.num_wires == 2
        assert dev.shots is None
        assert dev.short_name == "lightning.qubit"

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_load_default_qubit_device_with_valid_kernel(self):
        """Test that lightning.qubit works with valid kernel_for_ops argument."""
        for gate in ["PauliX", "CRot", "CSWAP", "Matrix"]:
            dev = qml.device("lightning.qubit", kernel_for_ops={gate: "PI"}, wires=2)

            assert dev.num_wires == 2
            assert dev.shots is None
            assert dev.short_name == "lightning.qubit"

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_load_default_qubit_device_with_invalid_kernel(self):
        """Test that lightning.qubit raises error for unsupported gate/kernel pair."""

        for gate in ["PauliX", "CRot", "CSWAP", "Matrix"]:
            with pytest.raises(
                ValueError, match=f"The given kernel Unknown does not implement {gate} gate."
            ):
                dev = qml.device("lightning.qubit", kernel_for_ops={gate: "Unknown"}, wires=2)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_load_default_qubit_device_with_invalid_param(self):
        """Test that lightning.qubit does not support kernel_for_ops type list."""
        with pytest.raises(ValueError, match=f"Argument kernel_for_ops must be a dictionary."):
            dev = qml.device("lightning.qubit", kernel_for_ops=["I am a list"], wires=2)

    @pytest.mark.skipif(not lq._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_all_exported_gates_are_available(self):
        """Test all exported gates from lightning_qubit_ops are accessible"""
        from pennylane_lightning import lightning_qubit_ops
        from pennylane_lightning.lightning_qubit_ops import StateVectorC128 as SV

        for kernel, gate_op in lightning_qubit_ops.EXPORTED_KERNEL_OPS:
            if gate_op != "Matrix":
                assert getattr(SV, f"{gate_op}_{kernel}", None) is not None
            else:
                assert getattr(SV, f"applyMatrix_{kernel}", None) is not None

    def test_no_backprop(self):
        """Test that lightning.qubit does not support the backprop
        differentiation method."""
        if not CPP_BINARY_AVAILABLE:
            pytest.skip("Skipping test because lightning.qubit is behaving like default.qubit")

        dev = qml.device("lightning.qubit", wires=2)

        def circuit():
            """Simple quantum function."""
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError):
            qml.QNode(circuit, dev, diff_method="backprop")

    def test_best_gets_lightning(self):
        """Test that the best differentiation method returns lightning
        qubit."""
        if not CPP_BINARY_AVAILABLE:
            pytest.skip("Skipping test because lightning.qubit is behaving like default.qubit")

        dev = qml.device("lightning.qubit", wires=2)

        def circuit():
            """Simple quantum function."""
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev, diff_method="best")
        assert isinstance(qnode.device, LightningQubit)

    def test_args(self):
        """Test that the plugin requires correct arguments"""

        with pytest.raises(TypeError, match="missing 1 required positional argument: 'wires'"):
            qml.device("lightning.qubit")

    def test_qubit_circuit(self, qubit_device_1_wire, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""

        p = 0.543

        @qml.qnode(qubit_device_1_wire)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_qubit_identity(self, qubit_device_1_wire, tol):
        """Test that the default qubit plugin provides correct result for the Identity expectation"""

        p = 0.543

        @qml.qnode(qubit_device_1_wire)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        assert np.isclose(circuit(p), 1, atol=tol, rtol=0)

    def test_nonzero_shots(self, tol):
        """Test that the default qubit plugin provides correct result for high shot number"""

        shots = 10**4
        dev = qml.device("lightning.qubit", wires=1, shots=shots)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        assert np.isclose(np.mean(runs), -np.sin(p), atol=1e-2, rtol=0)

    # This test is ran against the state |0> with one Z expval
    @pytest.mark.parametrize(
        "name,expected_output",
        [
            ("PauliX", -1),
            ("PauliY", -1),
            ("PauliZ", 1),
            ("Hadamard", 0),
        ],
    )
    def test_supported_gate_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, name, expected_output
    ):
        """Tests supported gates that act on a single wire that are not parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_operation(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            op(wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state |Phi+> with two Z expvals
    @pytest.mark.parametrize(
        "name,expected_output",
        [
            ("CNOT", [-1 / 2, 1]),
            ("SWAP", [-1 / 2, -1 / 2]),
            ("CZ", [-1 / 2, -1 / 2]),
        ],
    )
    def test_supported_gate_two_wires_no_parameters(
        self, qubit_device_2_wires, tol, name, expected_output
    ):
        """Tests supported gates that act on two wires that are not parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_2_wires.supports_operation(name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            qml.QubitStateVector(np.array([1 / 2, 0, 0, math.sqrt(3) / 2]), wires=[0, 1])
            op(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,expected_output",
        [
            ("CSWAP", [-1, -1, 1]),
        ],
    )
    def test_supported_gate_three_wires_no_parameters(
        self, qubit_device_3_wires, tol, name, expected_output
    ):
        """Tests supported gates that act on three wires that are not parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_3_wires.supports_operation(name)

        @qml.qnode(qubit_device_3_wires)
        def circuit():
            qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
            op(wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran with two Z expvals
    @pytest.mark.parametrize(
        "name,par,expected_output",
        [
            ("BasisState", [0, 0], [1, 1]),
            ("BasisState", [1, 0], [-1, 1]),
            ("BasisState", [0, 1], [1, -1]),
            ("QubitStateVector", [1, 0, 0, 0], [1, 1]),
            ("QubitStateVector", [0, 0, 1, 0], [-1, 1]),
            ("QubitStateVector", [0, 1, 0, 0], [1, -1]),
        ],
    )
    def test_supported_state_preparation(
        self, qubit_device_2_wires, tol, name, par, expected_output
    ):
        """Tests supported state preparations"""

        op = getattr(qml.ops, name)

        assert qubit_device_2_wires.supports_operation(name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            op(np.array(par), wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran with two Z expvals
    @pytest.mark.parametrize(
        "name,par,wires,expected_output",
        [
            ("BasisState", [1, 1], [0, 1], [-1, -1]),
            ("BasisState", [1], [0], [-1, 1]),
            ("BasisState", [1], [1], [1, -1]),
        ],
    )
    def test_basis_state_2_qubit_subset(
        self, qubit_device_2_wires, tol, name, par, wires, expected_output
    ):
        """Tests qubit basis state preparation on subsets of qubits"""

        op = getattr(qml.ops, name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            op(np.array(par), wires=wires)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is run with two expvals
    @pytest.mark.parametrize(
        "name,par,wires,expected_output",
        [
            ("QubitStateVector", [0, 1], [1], [1, -1]),
            ("QubitStateVector", [0, 1], [0], [-1, 1]),
            ("QubitStateVector", [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], [1], [1, 0]),
            ("QubitStateVector", [1j / 2.0, np.sqrt(3) / 2.0], [1], [1, -0.5]),
            ("QubitStateVector", [(2 - 1j) / 3.0, 2j / 3.0], [0], [1 / 9.0, 1]),
        ],
    )
    def test_state_vector_2_qubit_subset(
        self, qubit_device_2_wires, tol, name, par, wires, expected_output
    ):
        """Tests qubit state vector preparation on subsets of 2 qubits"""

        op = getattr(qml.ops, name)

        par = np.array(par)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            op(par, wires=wires)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is run with three expvals
    @pytest.mark.parametrize(
        "name,par,wires,expected_output",
        [
            (
                "QubitStateVector",
                [1j / np.sqrt(10), (1 - 2j) / np.sqrt(10), 0, 0, 0, 2 / np.sqrt(10), 0, 0],
                [0, 1, 2],
                [1 / 5.0, 1.0, -4 / 5.0],
            ),
            ("QubitStateVector", [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], [0, 2], [0.0, 1.0, 0.0]),
            ("QubitStateVector", [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], [0, 1], [0.0, 0.0, 1.0]),
            ("QubitStateVector", [0, 1, 0, 0, 0, 0, 0, 0], [2, 1, 0], [-1.0, 1.0, 1.0]),
            ("QubitStateVector", [0, 1j, 0, 0, 0, 0, 0, 0], [0, 2, 1], [1.0, -1.0, 1.0]),
            ("QubitStateVector", [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [1, 0], [-1.0, 0.0, 1.0]),
            ("QubitStateVector", [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1], [0.0, -1.0, 1.0]),
        ],
    )
    def test_state_vector_3_qubit_subset(
        self, qubit_device_3_wires, tol, name, par, wires, expected_output
    ):
        """Tests qubit state vector preparation on subsets of 3 qubits"""

        op = getattr(qml.ops, name)

        par = np.array(par)

        @qml.qnode(qubit_device_3_wires)
        def circuit():
            op(par, wires=wires)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran on the state |0> with one Z expvals
    @pytest.mark.parametrize(
        "name,par,expected_output",
        [
            ("PhaseShift", [math.pi / 2], 1),
            ("PhaseShift", [-math.pi / 4], 1),
            ("RX", [math.pi / 2], 0),
            ("RX", [-math.pi / 4], 1 / math.sqrt(2)),
            ("RY", [math.pi / 2], 0),
            ("RY", [-math.pi / 4], 1 / math.sqrt(2)),
            ("RZ", [math.pi / 2], 1),
            ("RZ", [-math.pi / 4], 1),
            ("Rot", [math.pi / 2, 0, 0], 1),
            ("Rot", [0, math.pi / 2, 0], 0),
            ("Rot", [0, 0, math.pi / 2], 1),
            ("Rot", [math.pi / 2, -math.pi / 4, -math.pi / 4], 1 / math.sqrt(2)),
            ("Rot", [-math.pi / 4, math.pi / 2, math.pi / 4], 0),
            ("Rot", [-math.pi / 4, math.pi / 4, math.pi / 2], 1 / math.sqrt(2)),
        ],
    )
    def test_supported_gate_single_wire_with_parameters(
        self, qubit_device_1_wire, tol, name, par, expected_output
    ):
        """Tests supported gates that act on a single wire that are parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_operation(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            op(*par, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state 1/2|00>+sqrt(3)/2|11> with two Z expvals
    @pytest.mark.parametrize(
        "name,par,expected_output",
        [
            ("CRX", [0], [-1 / 2, -1 / 2]),
            ("CRX", [-math.pi], [-1 / 2, 1]),
            ("CRX", [math.pi / 2], [-1 / 2, 1 / 4]),
            ("CRY", [0], [-1 / 2, -1 / 2]),
            ("CRY", [-math.pi], [-1 / 2, 1]),
            ("CRY", [math.pi / 2], [-1 / 2, 1 / 4]),
            ("CRZ", [0], [-1 / 2, -1 / 2]),
            ("CRZ", [-math.pi], [-1 / 2, -1 / 2]),
            ("CRZ", [math.pi / 2], [-1 / 2, -1 / 2]),
            ("CRot", [math.pi / 2, 0, 0], [-1 / 2, -1 / 2]),
            ("CRot", [0, math.pi / 2, 0], [-1 / 2, 1 / 4]),
            ("CRot", [0, 0, math.pi / 2], [-1 / 2, -1 / 2]),
            ("CRot", [math.pi / 2, 0, -math.pi], [-1 / 2, -1 / 2]),
            ("CRot", [0, math.pi / 2, -math.pi], [-1 / 2, 1 / 4]),
            ("CRot", [-math.pi, 0, math.pi / 2], [-1 / 2, -1 / 2]),
            ("ControlledPhaseShift", [0], [-1 / 2, -1 / 2]),
            ("ControlledPhaseShift", [-math.pi], [-1 / 2, -1 / 2]),
            ("ControlledPhaseShift", [math.pi / 2], [-1 / 2, -1 / 2]),
            ("ControlledPhaseShift", [math.pi], [-1 / 2, -1 / 2]),
        ],
    )
    def test_supported_gate_two_wires_with_parameters(
        self, qubit_device_2_wires, tol, name, par, expected_output
    ):
        """Tests supported gates that act on two wires that are parameterized"""

        op = getattr(qml.ops, name)

        assert qubit_device_2_wires.supports_operation(name)

        @qml.qnode(qubit_device_2_wires)
        def circuit():
            qml.QubitStateVector(np.array([1 / 2, 0, 0, math.sqrt(3) / 2]), wires=[0, 1])
            op(*par, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,state,expected_output",
        [
            ("PauliX", [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            ("PauliX", [1 / math.sqrt(2), -1 / math.sqrt(2)], -1),
            ("PauliX", [1, 0], 0),
            ("PauliY", [1 / math.sqrt(2), 1j / math.sqrt(2)], 1),
            ("PauliY", [1 / math.sqrt(2), -1j / math.sqrt(2)], -1),
            ("PauliY", [1, 0], 0),
            ("PauliZ", [1, 0], 1),
            ("PauliZ", [0, 1], -1),
            ("PauliZ", [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            ("Hadamard", [1, 0], 1 / math.sqrt(2)),
            ("Hadamard", [0, 1], -1 / math.sqrt(2)),
            ("Hadamard", [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / math.sqrt(2)),
        ],
    )
    def test_supported_observable_single_wire_no_parameters(
        self, qubit_device_1_wire, tol, name, state, expected_output
    ):
        """Tests supported observables on single wires without parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_observable(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "name,state,expected_output,par",
        [
            ("Identity", [1, 0], 1, []),
            ("Identity", [0, 1], 1, []),
            ("Identity", [1 / math.sqrt(2), -1 / math.sqrt(2)], 1, []),
        ],
    )
    def test_supported_observable_single_wire_with_parameters(
        self, qubit_device_1_wire, tol, name, state, expected_output, par
    ):
        """Tests supported observables on single wires with parameters."""

        obs = getattr(qml.ops, name)

        assert qubit_device_1_wire.supports_observable(name)

        @qml.qnode(qubit_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    def test_multi_samples_return_correlated_results(self, qubit_device_2_wires):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        dev = qubit_device_2_wires
        dev.shots = 1000

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])

    @pytest.mark.parametrize("num_wires", [3, 4, 5, 6, 7, 8])
    def test_multi_samples_return_correlated_results_more_wires_than_size_of_observable(
        self, num_wires
    ):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        dev = qml.device("lightning.qubit", wires=num_wires, shots=1000)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, theta, phi, varphi, qubit_device_3_wires, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qubit_device_3_wires
        dev.reset()

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_identity(self, theta, phi, varphi, qubit_device_3_wires, tol):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qubit_device_3_wires
        dev.reset()

        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = np.cos(varphi) * np.cos(phi)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, qubit_device_3_wires, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qubit_device_3_wires
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.expval(obs)

        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Tests for variance of tensor observables"""

    def test_paulix_pauliy(self, theta, phi, varphi, qubit_device_3_wires, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qubit_device_3_wires

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_pauliz_hadamard(self, theta, phi, varphi, qubit_device_3_wires, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qubit_device_3_wires
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.reset()
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        res = dev.var(obs)

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(res, expected, atol=tol, rtol=0)


# Tolerance for non-analytic tests
TOL_STOCHASTIC = 0.05


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
@pytest.mark.parametrize("shots", [None, 100000])
class TestTensorSample:
    """Test sampling tensor the tensor product of observables"""

    def test_paulix_pauliy(self, theta, phi, varphi, shots, monkeypatch, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        tolerance = tol if shots is None else TOL_STOCHASTIC
        dev = qml.device("lightning.qubit", wires=3, shots=shots)

        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples() if shots is not None else None

        s1 = obs.eigvals
        p = dev.probability(wires=obs.wires)

        # s1 should only contain 1 and -1
        assert np.allclose(s1**2, 1, atol=tolerance, rtol=0)

        mean = s1 @ p
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, atol=tolerance, rtol=0)

        var = (s1**2) @ p - (s1 @ p).real ** 2
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, atol=tolerance, rtol=0)

    def test_pauliz_hadamard(
        self, theta, phi, varphi, monkeypatch, shots, qubit_device_3_wires, tol
    ):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        tolerance = tol if shots is None else TOL_STOCHASTIC
        dev = qubit_device_3_wires
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            obs.diagonalizing_gates(),
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples() if dev.shots is not None else None

        s1 = obs.eigvals
        p = dev.marginal_prob(dev.probability(), wires=obs.wires)

        # s1 should only contain 1 and -1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

        mean = s1 @ p
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1**2) @ p - (s1 @ p).real ** 2
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, atol=tolerance, rtol=0)

    def test_qubitunitary_rotation_hadamard(
        self, theta, phi, varphi, monkeypatch, shots, qubit_device_3_wires, tol
    ):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        tolerance = tol if shots is None else TOL_STOCHASTIC
        dev = qubit_device_3_wires
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)
        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
            ],
            [
                qml.QubitUnitary(
                    obs.diagonalizing_gates()[0].get_matrix(),
                    wires=obs.diagonalizing_gates()[0].wires,
                ),
                *obs.diagonalizing_gates()[1:],
            ],
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples() if dev.shots is not None else None

        s1 = obs.eigvals
        p = dev.marginal_prob(dev.probability(), wires=obs.wires)

        # s1 should only contain 1 and -1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

        mean = s1 @ p
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol, rtol=0)

        var = (s1**2) @ p - (s1 @ p).real ** 2
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, atol=tolerance, rtol=0)


def test_warning():
    """Tests if a warning is raised when lightning.qubit binaries are not available"""
    if CPP_BINARY_AVAILABLE:
        pytest.skip("Test only applies when binaries are unavailable")

    with pytest.warns(UserWarning, match="Pre-compiled binaries for lightning.qubit"):
        qml.device("lightning.qubit", wires=1)
