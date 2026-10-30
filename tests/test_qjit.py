# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for executing qjit-compiled workflows on the Lightning devices."""

import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice, device_name

if device_name not in ("lightning.qubit", "lightning.kokkos", "lightning.amdgpu", "lightning.gpu"):
    pytest.skip("Device does not support qjit. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

pytest.importorskip("catalyst")


@pytest.fixture(params=[False, True], ids=["legacy", "capture"])
def capture(request):
    """Fixture to toggle program capture in qjit."""
    return request.param


class TestPauliMeasure:
    """Tests for executing qp.pauli_measure natively with qjit."""

    @staticmethod
    def run(capture, prep, pauli_word, wires):
        """Run ``prep`` and measure ``pauli_word`` on ``wires``. The measurement outcome is
        copied onto wire 2 so that the returned state also encodes it."""
        dev = qp.device(device_name, wires=3)

        @qp.qjit(capture=capture)
        @qp.qnode(dev)
        def circuit():
            prep()
            m = qp.pauli_measure(pauli_word, wires=wires)
            qp.cond(m, qp.X)(2)
            return qp.state()

        return circuit()

    @pytest.mark.parametrize("pauli_word, outcome", [("XX", 0), ("YY", 1), ("ZZ", 0)])
    def test_bell_state(self, capture, pauli_word, outcome):
        """Test that measuring a stabilizer of a Bell state returns its eigenvalue and leaves
        the Bell state unchanged."""

        def prep():
            qp.Hadamard(0)
            qp.CNOT([0, 1])

        state = self.run(capture, prep, pauli_word, [0, 1])

        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert np.allclose(state, np.kron(bell_state, np.eye(2)[outcome]))

    @pytest.mark.parametrize(
        "pauli_word, wires, outcome",
        [("ZI", [0, 1], 1), ("IZ", [0, 1], 0), ("ZI", [1, 0], 0), ("II", [0, 1], 0)],
    )
    def test_identity_entries(self, capture, pauli_word, wires, outcome):
        """Test that identity entries of the Pauli word do not contribute to the outcome."""
        state = self.run(capture, lambda: qp.X(0), pauli_word, wires)

        assert np.allclose(state, np.kron([0, 0, 1, 0], np.eye(2)[outcome]))
