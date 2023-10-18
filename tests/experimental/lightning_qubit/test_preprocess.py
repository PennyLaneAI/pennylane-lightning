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
"""Unit tests for preprocess in devices/qubit."""

import pytest

import numpy as np
import pennylane.numpy as pnp
import pennylane as qml
from pennylane.operation import Operation
from pennylane_lightning.core._preprocess import (
    stopping_condition,
)
from pennylane.devices import ExecutionConfig
from pennylane.measurements import MidMeasureMP, MeasurementValue
from pennylane.tape import QuantumScript
from pennylane import DeviceError

from pennylane_lightning.lightning_qubit import LightningQubit

# pylint: disable=too-few-public-methods


class NoMatOp(Operation):
    """Dummy operation for expanding circuit."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False

    def decomposition(self):
        return [qml.PauliX(self.wires), qml.PauliY(self.wires)]


class NoMatNoDecompOp(Operation):
    """Dummy operation for checking check_validity throws error when
    expected."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False


class TestPrivateHelpers:
    """Test the private helpers for preprocessing."""

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.PauliX(0), True),
            (qml.CRX(0.1, wires=[0, 1]), True),
            (qml.Barrier(), False),
            (qml.QFT(wires=range(5)), True),
            (qml.QFT(wires=range(12)), False),
            (qml.GroverOperator(wires=range(10)), True),
            (qml.GroverOperator(wires=range(14)), False),
        ],
    )
    def test_stopping_condition(self, op, expected):
        """Test that stopping_condition works correctly"""
        res = stopping_condition(op)
        assert res == expected


class TestBatchTransform:
    """Tests for the batch transformations."""

    def test_batch_transform_no_batching(self):
        """Test that batch_transform does nothing when no batching is required."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        device = LightningQubit()

        program, _ = device.preprocess()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        for op, expected in zip(tapes[0].circuit, ops + measurements):
            assert qml.equal(op, expected)

    def test_batch_transform_broadcast_not_adjoint(self):
        """Test that batch_transform does nothing when batching is required but
        internal PennyLane broadcasting can be used (diff method != adjoint)"""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)
        device = LightningQubit()

        program, _ = device.preprocess()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        for op, expected in zip(tapes[0].circuit, ops + measurements):
            assert qml.equal(op, expected)

    def test_batch_transform_broadcast_adjoint(self):
        """Test that batch_transform splits broadcasted tapes correctly when
        the diff method is adjoint"""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        execution_config = ExecutionConfig()
        execution_config.gradient_method = "adjoint"

        device = LightningQubit()

        program, _ = device.preprocess(execution_config=execution_config)
        tapes, _ = program([tape])
        expected_ops = [
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(tapes) == 2
        for i, t in enumerate(tapes):
            for op, expected in zip(t.circuit, expected_ops[i] + measurements):
                assert qml.equal(op, expected)
