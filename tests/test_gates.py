# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the correct application of gates with lightning.qubit.
"""
import itertools

import numpy as np
import pennylane as qml
import pytest

from pennylane_lightning import LightningQubit


@pytest.fixture
def op(op_name):
    return getattr(qml, op_name)


@pytest.mark.parametrize("op_name", LightningQubit.operations)
def test_gate_unitary_correct(op, op_name):
    """Test if lightning.qubit correctly applies gates by reconstructing the unitary matrix and
    comparing to the expected version"""

    if op_name in ("BasisState", "QubitStateVector"):
        pytest.skip("Skipping operation because it is a state preparation")
    if op_name in ("ControlledQubitUnitary", "QubitUnitary", "MultiControlledX", "DiagonalQubitUnitary"):
        pytest.skip("Skipping operation.")

    wires = int(op.num_wires)

    if wires == -1:  # This occurs for operations that do not have a predefined number of wires
        wires = 4

    dev = qml.device("lightning.qubit", wires=wires)
    num_params = op.num_params
    p = [0.1] * num_params

    if op_name == "DiagonalQubitUnitary": p = np.ones(wires)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        op(*p, wires=range(wires))
        return qml.state()
    # print(wires)
    unitary = np.zeros((2 ** wires, 2 ** wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(input)
        unitary[:, i] = out

    unitary_expected = op(*p, wires=range(wires)).matrix

    assert np.allclose(unitary, unitary_expected)


@pytest.mark.parametrize("op_name", LightningQubit.operations)
def test_inverse_unitary_correct(op, op_name):
    """Test if lightning.qubit correctly applies inverse gates by reconstructing the unitary matrix
    and comparing to the expected version"""

    if op_name in ("BasisState", "QubitStateVector"):
        pytest.skip("Skipping operation because it is a state preparation")

    wires = op.num_wires
    dev = qml.device("lightning.qubit", wires=wires)
    num_params = op.num_params
    p = [0.1] * num_params

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        op(*p, wires=range(wires)).inv()
        return qml.state()

    unitary = np.zeros((2 ** wires, 2 ** wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(input)
        unitary[:, i] = out

    unitary_expected = op(*p, wires=range(wires)).inv().matrix

    assert np.allclose(unitary, unitary_expected)


random_unitary = np.array(
    [
        [
            -0.48401572 - 0.11012304j,
            -0.44806504 + 0.46775911j,
            -0.36968281 + 0.19235993j,
            -0.37561358 + 0.13887962j,
        ],
        [
            -0.12838047 + 0.13992187j,
            0.14531831 + 0.45319438j,
            0.28902175 - 0.71158765j,
            -0.24333677 - 0.29721109j,
        ],
        [
            0.26400811 - 0.72519269j,
            0.13965687 + 0.35092711j,
            0.09141515 - 0.14367072j,
            0.14894673 + 0.45886629j,
        ],
        [
            -0.04067799 + 0.34681783j,
            -0.45852968 - 0.03214391j,
            -0.10528164 - 0.4431247j,
            0.50251451 + 0.45476965j,
        ],
    ]
)


@pytest.mark.xfail(strict=True)  # needs support for QubitUnitary
def test_arbitrary_unitary_correct():
    """Test if lightning.qubit correctly applies an arbitrary unitary by reconstructing its
    matrix"""
    wires = 2
    dev = qml.device("lightning.qubit", wires=wires)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        qml.QubitUnitary(random_unitary, wires=range(2))
        return qml.state()

    unitary = np.zeros((2 ** wires, 2 ** wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(input)
        unitary[:, i] = out

    assert np.allclose(unitary, random_unitary)


@pytest.mark.xfail(strict=True)  # needs support for QubitUnitary
def test_arbitrary_inv_unitary_correct():
    """Test if lightning.qubit correctly applies the inverse of an arbitrary unitary by
    reconstructing its matrix"""
    wires = 2
    dev = qml.device("lightning.qubit", wires=wires)

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        qml.QubitUnitary(random_unitary, wires=range(2)).inv()
        return qml.state()

    unitary = np.zeros((2 ** wires, 2 ** wires), dtype=np.complex128)

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(input)
        unitary[:, i] = out

    random_unitary_inv = random_unitary.conj().T
    assert np.allclose(unitary, random_unitary_inv)
