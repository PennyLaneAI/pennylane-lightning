# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for Lightning devices.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from conftest import THETA, PHI, VARPHI, TOL_STOCHASTIC, LightningDevice as ld, device_name

import math
import numpy as np
import pennylane as qml
from pennylane import DeviceError
from pennylane.operation import Operation
import functools


class TestApply:
    """Tests that operations of certain operations are applied correctly or
    that the proper errors are raised.
    """

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
        (qml.Identity, [1, 0], [1, 0]),
        (qml.Identity, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
    ]

    @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
    def test_apply_operation_single_wire_no_parameters(
        self, qubit_device, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""
        # from pennylane.wires import Wires

        dev = qubit_device(wires=1)
        _state = np.array(input).astype(dev.C_DTYPE)
        dev._apply_state_vector(_state, dev.wires)
        dev.apply([operation(wires=[0])])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev.state.dtype == dev.C_DTYPE

    # @pytest.mark.skipif(
    #     device_name == "lightning.kokkos" or device_name == "lightning.gpu",
    #     reason="Only meaningful for lightning_qubit",
    # )
    # @pytest.mark.skipif(not ld._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    # @pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
    # @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    # def test_apply_operation_preserve_pointer_single_wire_no_parameters(
    #     self, qubit_device, operation, input, expected_output, C
    # ):
    #     dev = qubit_device(wires=1)
    #     dev._state = dev._asarray(input, dtype=C)
    #     pointer_before, _ = dev._state.__array_interface__["data"]
    #     dev.apply([operation(wires=[0])])
    #     pointer_after, _ = dev._state.__array_interface__["data"]

    #     assert pointer_before == pointer_after

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
    def test_apply_operation_two_wires_no_parameters(
        self, qubit_device, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have no parameters."""
        dev = qubit_device(wires=2)
        _state = np.array(input).reshape(2 * [2]).astype(dev.C_DTYPE)
        dev._apply_state_vector(_state, dev.wires)
        dev.apply([operation(wires=[0, 1])])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev.state.dtype == dev.C_DTYPE

    # @pytest.mark.skipif(not ld._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    # @pytest.mark.parametrize("operation,input,expected_output", test_data_two_wires_no_parameters)
    # @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    # def test_apply_operation_preserve_pointer_two_wires_no_parameters(
    #     self, qubit_device, operation, input, expected_output, C
    # ):
    #     dev = qubit_device(wires=2)
    #     dev._state = dev._asarray(input, dtype=C).reshape(2 * [2])
    #     pointer_before, _ = dev._state.__array_interface__["data"]
    #     dev.apply([operation(wires=[0, 1])])
    #     pointer_after, _ = dev._state.__array_interface__["data"]

    #     assert pointer_before == pointer_after

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
    def test_apply_operation_three_wires_no_parameters(
        self, qubit_device, tol, operation, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for three wire
        operations that have no parameters."""

        dev = qubit_device(wires=3)
        _state = np.array(input).reshape(3 * [2]).astype(dev.C_DTYPE)
        dev._apply_state_vector(_state, dev.wires)
        dev.apply([operation(wires=[0, 1, 2])])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev.state.dtype == dev.C_DTYPE

    # @pytest.mark.skipif(not ld._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    # @pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_parameters)
    # @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    # def test_apply_operation_preserve_pointer_three_wires_no_parameters(
    #     self, qubit_device, operation, input, expected_output, C
    # ):
    #     dev = qubit_device(wires=3)
    #     dev._state = dev._asarray(input, dtype=C).reshape(3 * [2])
    #     pointer_before, _ = dev._state.__array_interface__["data"]
    #     dev.apply([operation(wires=[0, 1, 2])])
    #     pointer_after, _ = dev._state.__array_interface__["data"]

    #     assert pointer_before == pointer_after

    @pytest.mark.parametrize(
        "operation,expected_output,par",
        [
            (qml.BasisState, [0, 0, 1, 0], [1, 0]),
            (qml.BasisState, [0, 0, 0, 1], [1, 1]),
            (qml.QubitStateVector, [0, 0, 1, 0], [0, 0, 1, 0]),
            (qml.QubitStateVector, [0, 0, 0, 1], [0, 0, 0, 1]),
            (qml.StatePrep, [0, 0, 1, 0], [0, 0, 1, 0]),
            (qml.StatePrep, [0, 0, 0, 1], [0, 0, 0, 1]),
            (
                qml.StatePrep,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
            ),
            (
                qml.StatePrep,
                [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
                [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
            ),
        ],
    )
    def test_apply_operation_state_preparation(
        self, qubit_device, tol, operation, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        par = np.array(par)
        dev = qubit_device(wires=2)
        dev.reset()
        ops = [operation(par, wires=[0, 1])]

        dev.apply(ops)
        assert len(ops) == 1  # input not mutated

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)

    def test_integer_state_preparation(self, qubit_device, tol):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""
        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit0():
            qml.RX(0.2, wires=[0])
            return qml.state()

        @qml.qnode(dev)
        def circuit1():
            qml.StatePrep(np.array([1, 0, 0, 0]), wires=[0, 1])
            qml.RX(0.2, wires=[0])
            return qml.state()

        assert np.allclose(circuit0(), circuit1(), atol=tol, rtol=0)

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
    def test_apply_operation_single_wire_with_parameters(
        self, qubit_device, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        dev = qubit_device(wires=1)
        _state = np.array(input).astype(dev.C_DTYPE)
        dev._apply_state_vector(_state, dev.wires)
        dev.apply([operation(*par, wires=[0])])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev.state.dtype == dev.C_DTYPE

    # @pytest.mark.skipif(not ld._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    # @pytest.mark.parametrize(
    #     "operation,input,expected_output,par", test_data_single_wire_with_parameters
    # )
    # @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    # def test_apply_operation_preserve_pointer_single_wire_with_parameters(
    #     self, qubit_device, operation, input, expected_output, par, C
    # ):
    #     dev = qubit_device(wires=1)
    #     dev._state = dev._asarray(input, dtype=C)
    #     pointer_before, _ = dev._state.__array_interface__["data"]
    #     dev.apply([operation(*par, wires=[0])])
    #     pointer_after, _ = dev._state.__array_interface__["data"]

    #     assert pointer_before == pointer_after

    """ operation,input,expected_output,par """
    test_data_two_wires_with_parameters = [
        (qml.IsingXX, [1, 0, 0, 0], [1 / math.sqrt(2), 0, 0, -1j / math.sqrt(2)], [math.pi / 2]),
        (
            qml.IsingXX,
            [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
            [-0.5j, 0.5, -0.5j, 0.5],
            [math.pi / 2],
        ),
        (qml.IsingXY, [1, 0, 0, 0], [1, 0, 0, 0], [math.pi / 2]),
        (
            qml.IsingXY,
            [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
            [0, 0.5, 0.5j, 1 / math.sqrt(2)],
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
    def test_apply_operation_two_wires_with_parameters(
        self, qubit_device, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        dev = qubit_device(wires=2)
        _state = np.array(input).reshape(2 * [2]).astype(dev.C_DTYPE)
        dev._apply_state_vector(_state, dev.wires)
        dev.apply([operation(*par, wires=[0, 1])])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev.state.dtype == dev.C_DTYPE

    # @pytest.mark.skipif(not ld._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    # @pytest.mark.parametrize(
    #     "operation,input,expected_output,par", test_data_two_wires_with_parameters
    # )
    # @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    # def test_apply_operation_preserve_pointer_two_wires_with_parameters(
    #     self, qubit_device, operation, input, expected_output, par, C
    # ):
    #     dev = qubit_device(wires=2)
    #     dev._state = dev._asarray(input, dtype=C).reshape(2 * [2])
    #     pointer_before, _ = dev._state.__array_interface__["data"]
    #     dev.apply([operation(*par, wires=[0, 1])])
    #     pointer_after, _ = dev._state.__array_interface__["data"]

    #     assert pointer_before == pointer_after

    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_apply_errors_qubit_state_vector(self, stateprep, qubit_device):
        """Test that apply fails for incorrect state preparation, and > 2 qubit gates"""
        dev = qubit_device(wires=2)
        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            dev.apply([stateprep(np.array([1, -1]), wires=[0])])

        with pytest.raises(
            DeviceError,
            match=f"Operation {stateprep(np.array([1, 0]), wires=[0]).name} cannot be used after other Operations have already been applied ",
        ):
            dev.reset()
            dev.apply([qml.RZ(0.5, wires=[0]), stateprep(np.array([0, 1, 0, 0]), wires=[0, 1])])

    def test_apply_errors_basis_state(self, qubit_device):
        dev = qubit_device(wires=2)
        with pytest.raises(
            ValueError, match="BasisState parameter must consist of 0 or 1 integers."
        ):
            dev.apply([qml.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])])

        with pytest.raises(
            ValueError, match="BasisState parameter and wires must be of equal length."
        ):
            dev.apply([qml.BasisState(np.array([0, 1]), wires=[0])])

        with pytest.raises(
            DeviceError,
            match="Operation BasisState cannot be used after other Operations have already been applied ",
        ):
            dev.reset()
            dev.apply([qml.RZ(0.5, wires=[0]), qml.BasisState(np.array([1, 1]), wires=[0, 1])])


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
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_expval_single_wire_no_parameters(
        self, qubit_device, tol, stateprep, operation, input, expected_output
    ):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""
        dev = qubit_device(wires=1)
        obs = operation(wires=[0])

        dev.reset()
        dev.apply([stateprep(np.array(input), wires=[0])], obs.diagonalizing_gates())
        res = dev.expval(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_expval_estimate(self):
        """Test that the expectation value is not analytically calculated"""
        dev = qml.device(device_name, wires=1, shots=3)

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
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_var_single_wire_no_parameters(
        self, qubit_device, tol, stateprep, operation, input, expected_output
    ):
        """Tests that variances are properly calculated for single-wire observables without parameters."""
        dev = qubit_device(wires=1)
        obs = operation(wires=[0])

        dev.reset()
        dev.apply([stateprep(np.array(input), wires=[0])], obs.diagonalizing_gates())
        res = dev.var(obs)

        assert np.isclose(res, expected_output, atol=tol, rtol=0)

    def test_var_estimate(self):
        """Test that the variance is not analytically calculated"""

        dev = qml.device(device_name, wires=1, shots=3)

        @qml.qnode(dev)
        def circuit():
            return qml.var(qml.PauliX(0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance and an analytically calculated one
        assert var != 1.0


class TestSample:
    """Tests that samples are properly calculated."""

    def test_sample_dimensions(self, qubit_device):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qubit_device(wires=2)
        dev.reset()

        dev.apply([qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])])

        dev.shots = 10
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()
        s1 = dev.sample(qml.PauliZ(wires=[0]))
        assert np.array_equal(s1.shape, (10,))

        dev.reset()
        dev.shots = 12
        dev._wires_measured = {1}
        dev._samples = dev.generate_samples()
        s2 = dev.sample(qml.PauliZ(wires=[1]))
        assert np.array_equal(s2.shape, (12,))

        dev.reset()
        dev.shots = 17
        dev._wires_measured = {0, 1}
        dev._samples = dev.generate_samples()
        s3 = dev.sample(qml.PauliX(0) @ qml.PauliZ(1))
        assert np.array_equal(s3.shape, (17,))

    def test_sample_values(self, qubit_device, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        # Explicitly resetting is necessary as the internal
        # state is set to None in __init__ and only properly
        # initialized during reset
        dev = qubit_device(wires=2)
        dev.reset()

        dev.shots = 1000
        dev.apply([qml.RX(1.5708, wires=[0])])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()

        s1 = dev.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)


class TestLightningDeviceIntegration:
    """Integration tests for lightning device. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_load_default_qubit_device(self):
        """Test that the default plugin loads correctly"""

        dev = qml.device(device_name, wires=2)
        assert dev.num_wires == 2
        assert dev.shots is None
        assert dev.short_name == device_name

    @pytest.mark.skipif(not ld._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_no_backprop(self):
        """Test that lightning device does not support the backprop
        differentiation method."""

        dev = qml.device(device_name, wires=2)

        def circuit():
            """Simple quantum function."""
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError):
            qml.QNode(circuit, dev, diff_method="backprop")

    @pytest.mark.skipif(not ld._CPP_BINARY_AVAILABLE, reason="Lightning binary required")
    def test_best_gets_lightning(self):
        """Test that the best differentiation method returns lightning
        qubit."""
        dev = qml.device(device_name, wires=2)

        def circuit():
            """Simple quantum function."""
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev, diff_method="best")
        assert isinstance(qnode.device, ld)

    def test_args(self):
        """Test that the plugin requires correct arguments"""

        with pytest.raises(TypeError, match="missing 1 required positional argument: 'wires'"):
            qml.device(device_name)

    def test_qubit_circuit(self, qubit_device, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""

        p = 0.543
        dev = qubit_device(wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_qubit_identity(self, qubit_device, tol):
        """Test that the default qubit plugin provides correct result for the Identity expectation"""

        p = 0.543
        dev = qubit_device(wires=1)

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        assert np.isclose(circuit(p), 1, atol=tol, rtol=0)

    def test_nonzero_shots(self, tol_stochastic):
        """Test that the default qubit plugin provides correct result for high shot number"""

        shots = 10**4
        dev = qml.device(device_name, wires=1, shots=shots)

        p = 0.543

        @qml.qnode(dev)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        runs = []
        for _ in range(100):
            runs.append(circuit(p))

        assert np.isclose(np.mean(runs), -np.sin(p), atol=tol_stochastic, rtol=0)

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
        self, qubit_device, tol, name, expected_output
    ):
        """Tests supported gates that act on a single wire that are not parameterized"""
        dev = qubit_device(wires=1)
        op = getattr(qml.ops, name)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
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
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_supported_gate_two_wires_no_parameters(
        self, qubit_device, tol, stateprep, name, expected_output
    ):
        """Tests supported gates that act on two wires that are not parameterized"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            stateprep(np.array([1 / 2, 0, 0, math.sqrt(3) / 2]), wires=[0, 1])
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
        self, qubit_device, tol, name, expected_output
    ):
        """Tests supported gates that act on three wires that are not parameterized"""
        dev = qubit_device(wires=3)
        op = getattr(qml.ops, name)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
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
    def test_supported_state_preparation(self, qubit_device, tol, name, par, expected_output):
        """Tests supported state preparations"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
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
    def test_basis_state_2_qubit_subset(self, qubit_device, tol, name, par, wires, expected_output):
        """Tests qubit basis state preparation on subsets of qubits"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        @qml.qnode(dev)
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
        self, qubit_device, tol, name, par, wires, expected_output
    ):
        """Tests qubit state vector preparation on subsets of 2 qubits"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        par = np.array(par)

        @qml.qnode(dev)
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
        self, qubit_device, tol, name, par, wires, expected_output
    ):
        """Tests qubit state vector preparation on subsets of 3 qubits"""
        dev = qubit_device(wires=3)
        op = getattr(qml.ops, name)

        par = np.array(par)

        @qml.qnode(dev)
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
        self, qubit_device, tol, name, par, expected_output
    ):
        """Tests supported gates that act on a single wire that are parameterized"""
        dev = qubit_device(wires=1)
        op = getattr(qml.ops, name)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
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
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_supported_gate_two_wires_with_parameters(
        self, qubit_device, tol, stateprep, name, par, expected_output
    ):
        """Tests supported gates that act on two wires that are parameterized"""
        dev = qubit_device(wires=2)
        op = getattr(qml.ops, name)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            stateprep(np.array([1 / 2, 0, 0, math.sqrt(3) / 2]), wires=[0, 1])
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
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_supported_observable_single_wire_no_parameters(
        self, qubit_device, tol, stateprep, name, state, expected_output
    ):
        """Tests supported observables on single wires without parameters."""
        dev = qubit_device(wires=1)
        obs = getattr(qml.ops, name)

        assert dev.supports_observable(name)

        @qml.qnode(dev)
        def circuit():
            stateprep(np.array(state), wires=[0])
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
    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_supported_observable_single_wire_with_parameters(
        self, qubit_device, tol, stateprep, name, state, expected_output, par
    ):
        """Tests supported observables on single wires with parameters."""
        dev = qubit_device(wires=1)
        obs = getattr(qml.ops, name)

        assert dev.supports_observable(name)

        @qml.qnode(dev)
        def circuit():
            stateprep(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    def test_multi_samples_return_correlated_results(self, qubit_device):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """
        dev = qubit_device(wires=2)
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

        dev = qml.device(device_name, wires=num_wires, shots=1000)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])

    def test_snapshot_is_ignored_without_shot(self):
        """Tests if the Snapshot operator is ignored correctly"""
        dev = qml.device(device_name, wires=4)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot()
            qml.adjoint(qml.Snapshot())
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        outcomes = circuit()

        assert np.allclose(outcomes, [0.0])

    def test_snapshot_is_ignored_with_shots(self):
        """Tests if the Snapshot operator is ignored correctly"""
        dev = qml.device(device_name, wires=4, shots=1000)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Snapshot()
            qml.adjoint(qml.Snapshot())
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

        outcomes = circuit()

        assert np.array_equal(outcomes[0], outcomes[1])

    def test_apply_qpe(self, qubit_device, tol):
        """Test the application of qml.QuantumPhaseEstimation"""
        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.QuantumPhaseEstimation(qml.matrix(qml.Hadamard)(wires=0), [0], [1])
            return qml.probs(wires=[0, 1])

        circuit()

        res_sv = dev.state
        res_probs = dev.probability([0, 1])

        expected_sv = np.array(
            [
                0.85355339 + 0.000000e00j,
                -0.14644661 - 6.123234e-17j,
                0.35355339 + 0.000000e00j,
                0.35355339 + 0.000000e00j,
            ]
        )
        expected_prob = np.array([0.72855339, 0.02144661, 0.125, 0.125])

        assert np.allclose(res_sv, expected_sv, atol=tol, rtol=0)
        assert np.allclose(res_probs, expected_prob, atol=tol, rtol=0)


@pytest.mark.skipif(
    device_name == "lightning.kokkos" or device_name == "lightning.gpu",
    reason="lightning.kokkos/gpu does not support apply with rotations.",
)
@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qubit_device(wires=3)
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

    def test_pauliz_identity(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliZ and Identity works correctly"""
        dev = qubit_device(wires=3)
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

    def test_pauliz_hadamard(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qubit_device(wires=3)
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


@pytest.mark.skipif(
    device_name == "lightning.kokkos" or device_name == "lightning.gpu",
    reason="lightning.kokkos/gpu does not support apply with rotations.",
)
@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
class TestTensorVar:
    """Tests for variance of tensor observables"""

    def test_paulix_pauliy(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = qubit_device(wires=3)
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

    def test_pauliz_hadamard(self, theta, phi, varphi, qubit_device, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = qubit_device(wires=3)
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


@pytest.mark.skipif(
    device_name == "lightning.kokkos" or device_name == "lightning.gpu",
    reason="lightning.kokkos/lightning.gpu does not support apply with rotations.",
)
@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
@pytest.mark.parametrize("shots", [None, 100000])
class TestTensorSample:
    """Test sampling tensor the tensor product of observables"""

    def test_paulix_pauliy(self, theta, phi, varphi, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        tolerance = tol if shots is None else TOL_STOCHASTIC
        dev = qml.device(device_name, wires=3, shots=shots)

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

        s1 = qml.eigvals(obs)
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

    def test_pauliz_hadamard(self, theta, phi, varphi, shots, qubit_device, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        tolerance = tol if shots is None else TOL_STOCHASTIC
        dev = qubit_device(wires=3)
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

        s1 = qml.eigvals(obs)
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

    def test_qubitunitary_rotation_hadamard(self, theta, phi, varphi, shots, qubit_device, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        tolerance = tol if shots is None else TOL_STOCHASTIC
        dev = qubit_device(wires=3)
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
                    qml.matrix(obs.diagonalizing_gates()[0]),
                    wires=obs.diagonalizing_gates()[0].wires,
                ),
                *obs.diagonalizing_gates()[1:],
            ],
        )

        dev._wires_measured = {0, 1, 2}
        dev._samples = dev.generate_samples() if dev.shots is not None else None

        s1 = qml.eigvals(obs)
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


# This block of similar tests coming from pennylane will not fail:
# def ansatz(a, b, c):
#     qml.RX(a, wires=0)
#     qml.RX(b, wires=1)
#     qml.RX(c, wires=2)
#     qml.CNOT(wires=[0, 1])
#     qml.CNOT(wires=[1, 2])
# Z = np.array([[1, 0], [0, -1]])

# from pennylane import expval, var, sample

# @pytest.mark.parametrize("shots", [None, int(1e6)])
# @pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
# class TestTensorExpval:
#     """Test tensor expectation values"""

#     @pytest.fixture
#     def tolerance(self, shots, tol):
#         if shots is not None:
#             return {"atol": 0.01, "rtol": 0.1}

#         return {"atol": tol, "rtol": 0}

#     def test_tensor_product(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product ZxZ gives the same result as simply
#         using an Hermitian matrix"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         @qml.qnode(dev)
#         def circuit1(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.PauliZ(0) @ qml.PauliZ(2))

#         @qml.qnode(dev)
#         def circuit2(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.Hermitian(np.kron(Z, Z), wires=[0, 2]))

#         res1 = circuit1(theta, phi, varphi)
#         res2 = circuit2(theta, phi, varphi)

#         assert np.allclose(res1, res2, **tolerance)

#     def test_combine_tensor_with_non_tensor(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product along with a non-tensor product
#         continues to function correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         @qml.qnode(dev)
#         def circuit1(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.PauliZ(0) @ qml.PauliZ(2)), expval(qml.PauliZ(1))

#         @qml.qnode(dev)
#         def circuit2(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.Hermitian(np.kron(Z, Z), wires=[0, 2]))

#         @qml.qnode(dev)
#         def circuit3(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.PauliZ(1))

#         res1 = circuit1(theta, phi, varphi)
#         res2 = circuit2(theta, phi, varphi), circuit3(theta, phi, varphi)

#         assert np.allclose(res1, res2, **tolerance)

#     def test_paulix_tensor_pauliy(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving PauliX and PauliY works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.PauliX(0) @ qml.PauliY(2))

#         res = circuit(theta, phi, varphi)
#         expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

#         assert np.allclose(res, expected, **tolerance)

#     @pytest.mark.autograd
#     def test_paulix_tensor_pauliy_gradient(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving PauliX and PauliY works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.PauliX(0) @ qml.PauliY(2))

#         dcircuit = qml.grad(circuit, 0)
#         res = dcircuit(theta, phi, varphi)
#         expected = np.cos(theta) * np.sin(phi) * np.sin(varphi)

#         assert np.allclose(res, expected, **tolerance)

#     def test_pauliz_tensor_identity(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving PauliZ and Identity works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2))

#         res = circuit(theta, phi, varphi)
#         expected = np.cos(varphi) * np.cos(phi)

#         assert np.allclose(res, expected, **tolerance)

#     def test_pauliz_tensor_hadamard(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving PauliZ and hadamard works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

#         res = circuit(theta, phi, varphi)
#         expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

#         assert np.allclose(res, expected, **tolerance)

#     def test_hermitian(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving an Hermitian matrix works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         A = np.array(
#             [
#                 [-6, 2 + 1j, -3, -5 + 2j],
#                 [2 - 1j, 0, 2 - 1j, -5 + 4j],
#                 [-3, 2 + 1j, 0, -4 + 3j],
#                 [-5 - 2j, -5 - 4j, -4 - 3j, -6],
#             ]
#         )

#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.PauliZ(0) @ qml.Hermitian(A, [1, 2]))

#         res = circuit(theta, phi, varphi)
#         expected = 0.5 * (
#             -6 * np.cos(theta) * (np.cos(varphi) + 1)
#             - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
#             + 3 * np.cos(varphi) * np.sin(phi)
#             + np.sin(phi)
#         )

#         assert np.allclose(res, expected, **tolerance)

#     def test_hermitian_tensor_hermitian(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving two Hermitian matrices works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         A1 = np.array([[1, 2], [2, 4]])

#         A2 = np.array(
#             [
#                 [-6, 2 + 1j, -3, -5 + 2j],
#                 [2 - 1j, 0, 2 - 1j, -5 + 4j],
#                 [-3, 2 + 1j, 0, -4 + 3j],
#                 [-5 - 2j, -5 - 4j, -4 - 3j, -6],
#             ]
#         )

#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return expval(qml.Hermitian(A1, 0) @ qml.Hermitian(A2, [1, 2]))

#         res = circuit(theta, phi, varphi)
#         expected = 0.25 * (
#             -30
#             + 4 * np.cos(phi) * np.sin(theta)
#             + 3 * np.cos(varphi) * (-10 + 4 * np.cos(phi) * np.sin(theta) - 3 * np.sin(phi))
#             - 3 * np.sin(phi)
#             - 2
#             * (5 + np.cos(phi) * (6 + 4 * np.sin(theta)) + (-3 + 8 * np.sin(theta)) * np.sin(phi))
#             * np.sin(varphi)
#             + np.cos(theta)
#             * (
#                 18
#                 + 5 * np.sin(phi)
#                 + 3 * np.cos(varphi) * (6 + 5 * np.sin(phi))
#                 + 2 * (3 + 10 * np.cos(phi) - 5 * np.sin(phi)) * np.sin(varphi)
#             )
#         )

#         assert np.allclose(res, expected, **tolerance)

#     def test_hermitian_tensor_identity_expectation(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving an Hermitian matrix and the identity works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=2, shots=shots)

#         A = np.array(
#             [[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]]
#         )

#         # pylint: disable=unused-argument
#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             qml.RY(a, wires=0)
#             qml.RY(b, wires=1)
#             qml.CNOT(wires=[0, 1])
#             return expval(qml.Hermitian(A, 0) @ qml.Identity(1))

#         res = circuit(theta, phi, varphi)

#         a = A[0, 0]
#         re_b = A[0, 1].real
#         d = A[1, 1]
#         expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2

#         assert np.allclose(res, expected, **tolerance)


# @pytest.mark.parametrize("shots", [None, int(1e6)])
# @pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
# class TestTensorVar:
#     """Tests for variance of tensor observables"""

#     @pytest.fixture
#     def tolerance(self, shots, tol):
#         if shots is not None:
#             return {"atol": 0.01, "rtol": 0.2}

#         return {"atol": tol, "rtol": 0}

#     def test_paulix_tensor_pauliy(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving PauliX and PauliY works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return var(qml.PauliX(0) @ qml.PauliY(2))

#         res = circuit(theta, phi, varphi)
#         expected = (
#             8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
#             - np.cos(2 * (theta - phi))
#             - np.cos(2 * (theta + phi))
#             + 2 * np.cos(2 * theta)
#             + 2 * np.cos(2 * phi)
#             + 14
#         ) / 16

#         assert np.allclose(res, expected, **tolerance)

#     def test_pauliz_tensor_hadamard(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving PauliZ and hadamard works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return var(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

#         res = circuit(theta, phi, varphi)
#         expected = (
#             3
#             + np.cos(2 * phi) * np.cos(varphi) ** 2
#             - np.cos(2 * theta) * np.sin(varphi) ** 2
#             - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
#         ) / 4

#         assert np.allclose(res, expected, **tolerance)

#     def test_tensor_hermitian(self, shots, theta, phi, varphi, tolerance):
#         """Test that a tensor product involving qml.Hermitian works correctly"""
#         dev = qml.device(device_name, wires=3, shots=shots)
#         # dev = qml.device("default.qubit", wires=3, shots=shots)

#         A = np.array(
#             [
#                 [-6, 2 + 1j, -3, -5 + 2j],
#                 [2 - 1j, 0, 2 - 1j, -5 + 4j],
#                 [-3, 2 + 1j, 0, -4 + 3j],
#                 [-5 - 2j, -5 - 4j, -4 - 3j, -6],
#             ]
#         )

#         @qml.qnode(dev)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return var(qml.PauliZ(0) @ qml.Hermitian(A, [1, 2]))

#         res = circuit(theta, phi, varphi)
#         expected = (
#             1057
#             - np.cos(2 * phi)
#             + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
#             - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
#             + 16 * np.sin(2 * phi)
#             - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
#             - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
#             - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
#             - 8
#             * np.cos(theta)
#             * (
#                 4
#                 * np.cos(phi)
#                 * (
#                     4
#                     + 8 * np.cos(varphi)
#                     + np.cos(2 * varphi)
#                     - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
#                 )
#                 + np.sin(phi)
#                 * (
#                     15
#                     + 8 * np.cos(varphi)
#                     - 11 * np.cos(2 * varphi)
#                     + 42 * np.sin(varphi)
#                     + 3 * np.sin(2 * varphi)
#                 )
#             )
#         ) / 16

#         assert np.allclose(res, expected, **tolerance)


# def tensor_product(observables):
#     return functools.reduce(np.kron, observables)


# @pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
# class TestTensorSample:
#     """Tests for samples of tensor observables"""

#     # pylint: disable=unused-argument
#     def test_paulix_tensor_pauliz(self, theta, phi, varphi, tol_stochastic):
#         """Test that a tensor product involving PauliX and PauliZ works correctly"""
#         # dev = qml.device("default.qubit", wires=2, shots=int(1e6))
#         dev = qml.device(device_name, wires=2, shots=int(1e6))

#         @qml.qnode(dev)
#         def circuit():
#             qml.Hadamard(wires=0)
#             return sample(qml.PauliX(0) @ qml.PauliZ(1))

#         s1 = circuit()

#         # s1 should only contain 1
#         assert np.allclose(s1, 1, atol=tol_stochastic, rtol=0)

#     def test_paulix_tensor_pauliy(self, theta, phi, varphi, tol_stochastic):
#         """Test that a tensor product involving PauliX and PauliY works correctly"""
#         # dev = qml.device("default.qubit", wires=3, shots=int(1e6))
#         dev = qml.device(device_name, wires=3, shots=int(1e6))

#         @qml.qnode(dev, diff_method="parameter-shift")
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return sample(qml.PauliX(0) @ qml.PauliY(2))

#         s1 = circuit(theta, phi, varphi)

#         # s1 should only contain 1 and -1
#         assert np.allclose(s1**2, 1, atol=tol_stochastic, rtol=0)

#     def test_pauliz_tensor_hadamard(self, theta, phi, varphi, tol_stochastic):
#         """Test that a tensor product involving PauliZ and hadamard works correctly"""
#         # dev = qml.device("default.qubit", wires=3, shots=int(1e6))
#         dev = qml.device(device_name, wires=3, shots=int(1e6))

#         @qml.qnode(dev, diff_method="parameter-shift")
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return sample(qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2))

#         s1 = circuit(theta, phi, varphi)

#         # s1 should only contain 1 and -1
#         assert np.allclose(s1**2, 1, atol=tol_stochastic, rtol=0)

#     def test_tensor_hermitian(self, theta, phi, varphi, tol_stochastic):
#         """Test that a tensor product involving qml.Hermitian works correctly"""
#         # dev = qml.device("default.qubit", wires=3, shots=int(1e6))
#         dev = qml.device(device_name, wires=3, shots=int(1e6))

#         A = np.array(
#             [
#                 [-6, 2 + 1j, -3, -5 + 2j],
#                 [2 - 1j, 0, 2 - 1j, -5 + 4j],
#                 [-3, 2 + 1j, 0, -4 + 3j],
#                 [-5 - 2j, -5 - 4j, -4 - 3j, -6],
#             ]
#         )

#         @qml.qnode(dev, diff_method=None)
#         def circuit(a, b, c):
#             ansatz(a, b, c)
#             return sample(qml.PauliZ(0) @ qml.Hermitian(A, [1, 2]))

#         s1 = circuit(theta, phi, varphi)

#         # s1 should only contain the eigenvalues of
#         # the hermitian matrix tensor product Z
#         eigvals = np.linalg.eigvalsh(np.kron(Z, A))
#         assert set(np.round(s1, 8)).issubset(set(np.round(eigvals, 8)))


class TestApplyLightningMethod:
    """Unit tests for the apply_lightning method."""

    def test_apply_identity_skipped(self, mocker, tol):
        """Test identity operation does not perform additional computations."""
        dev = qml.device(device_name, wires=1)
        dev._apply_state_vector(dev._asarray(dev.state).astype(dev.C_DTYPE), dev.wires)

        starting_state = np.array([1, 0], dtype=dev.C_DTYPE)
        op = [qml.Identity(0)]
        dev.apply(op)

        assert np.allclose(dev.state, starting_state, atol=tol, rtol=0)
        assert dev.state.dtype == dev.C_DTYPE

    @pytest.mark.skipif(
        not ld._CPP_BINARY_AVAILABLE or device_name != "lightning.gpu",
        reason="Only meaningful when binary is available.",
    )
    def test_unsupported_operation(self, mocker, tol):
        """Test unsupported operations."""

        class EmptyGate(Operation):
            num_wires = 1

            @staticmethod
            def compute_matrix(*params, **hyperparams):
                return np.eye(0)

        dev = qml.device(device_name, wires=1)
        dev.operations.add("EmptyGate")

        with pytest.raises(ValueError, match="Unsupported operation"):
            dev.apply_lightning([EmptyGate(0)])


@pytest.mark.skipif(
    ld._CPP_BINARY_AVAILABLE, reason="Test only applies when binaries are unavailable"
)
def test_warning():
    """Tests if a warning is raised when lightning device binaries are not available"""
    with pytest.warns(UserWarning, match="Pre-compiled binaries for " + device_name):
        qml.device(device_name, wires=1)


@pytest.mark.parametrize(
    "op",
    [
        qml.BasisState([0, 0], wires=[0, 1]),
        qml.QubitStateVector([0, 1, 0, 0], wires=[0, 1]),
        qml.StatePrep([0, 1, 0, 0], wires=[0, 1]),
    ],
)
@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
def test_circuit_with_stateprep(op, theta, phi, tol):
    """Test mid-circuit StatePrep"""
    n_qubits = 5
    n_wires = 2
    dev_def = qml.device("default.qubit", wires=n_qubits)
    dev = qml.device(device_name, wires=n_qubits)
    m = 2**n_wires
    U = np.random.rand(m, m) + 1j * np.random.rand(m, m)
    U, _ = np.linalg.qr(U)
    init_state = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
    init_state /= np.sqrt(np.dot(np.conj(init_state), init_state))

    def circuit():
        qml.StatePrep(init_state, wires=range(n_qubits))
        qml.RY(theta, wires=[0])
        qml.RY(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        op
        qml.QubitUnitary(U, wires=range(2, 2 + 2 * n_wires, 2))
        return qml.state()

    circ = qml.QNode(circuit, dev)
    circ_def = qml.QNode(circuit, dev_def)
    assert np.allclose(circ(), circ_def(), tol)
