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
Unit tests for the inverse of gates with lightning.qubit
"""
import pytest
import pennylane as qml
from pennylane_lightning import LightningQubit
import numpy as np
import itertools

@pytest.fixture
def op(op_name):
    return getattr(qml, op_name)


@pytest.mark.parametrize("op_name", LightningQubit.kernel_operations)
def test_inverse_correct(op, op_name):
    wires = op.num_wires
    dev = qml.device("lightning.qubit", wires=wires)
    num_params = op.num_params
    p = [0.1] * num_params

    @qml.qnode(dev)
    def output(input):
        qml.BasisState(input, wires=range(wires))
        op(*p, wires=range(wires)).inv()
        return qml.state()

    unitary = np.zeros((2 ** wires, 2 ** wires))

    for i, input in enumerate(itertools.product([0, 1], repeat=wires)):
        out = output(input)
        unitary[:, i] = out

    unitary_expected = op(*p, wires=range(wires)).inv().matrix

    # assert np.allclose(unitary, unitary_expected)

    if not np.allclose(unitary, unitary_expected):
        print(op_name)
        print(unitary)
        print(unitary_expected)
        print()

