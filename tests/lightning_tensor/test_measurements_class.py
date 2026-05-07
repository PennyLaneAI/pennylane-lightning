# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Unit tests for measurements class.
"""

import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice, device_name, get_random_normalized_state  # tested device

if device_name != "lightning.tensor":
    pytest.skip(
        "Skipping tests for the LightningTensorMeasurements class.", allow_module_level=True
    )

from pennylane.exceptions import DeviceError

from pennylane_lightning.lightning_tensor._measurements import LightningTensorMeasurements
from pennylane_lightning.lightning_tensor._tensornet import LightningTensorNet

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)

device_args = []
for method in ["mps", "tn"]:
    for c_dtype in [np.complex64, np.complex128]:
        device_arg = {}
        device_arg["method"] = method
        device_arg["c_dtype"] = c_dtype
        if method == "mps":
            device_arg["max_bond_dim"] = 128
        device_args.append(device_arg)


# General LightningTensorNet fixture, for any number of wires.
@pytest.fixture(
    params=device_args,
)
def lightning_tn(request):
    """Fixture for creating a LightningTensorNet object."""

    def _lightning_tn(n_wires):
        return LightningTensorNet(num_wires=n_wires, **request.param)

    return _lightning_tn


class TestMeasurementFunction:
    """Tests for the measurement method."""

    def test_initialization(self, lightning_tn):
        """Tests for the initialization of the LightningTensorMeasurements class."""
        tensornetwork = lightning_tn(2)
        m = LightningTensorMeasurements(tensornetwork)

        assert m.dtype == tensornetwork.dtype

    def test_not_implemented_state_measurements(self, lightning_tn):
        """Test than a NotImplementedError is raised if the measurement is not a state measurement."""

        tensornetwork = lightning_tn(2)
        m = LightningTensorMeasurements(tensornetwork)

        mp = qp.counts(wires=(0, 1))
        with pytest.raises(NotImplementedError):
            m.get_measurement_function(mp)

    def test_not_supported_sparseH_shot_measurements(self, lightning_tn):
        """Test than a TypeError is raised if the measurement is not supported."""

        tensornetwork = lightning_tn(3)

        m = LightningTensorMeasurements(tensornetwork)

        ops = [qp.PauliX(0), qp.PauliZ(1)]

        obs = qp.SparseHamiltonian(
            qp.Hamiltonian([-1.0, 1.5], [qp.Z(1), qp.X(1)]).sparse_matrix(wire_order=[0, 1, 2]),
            wires=[0, 1, 2],
        )

        for mp in [qp.var(obs), qp.expval(obs)]:
            tape = qp.tape.QuantumScript(ops, [mp], shots=100)

            with pytest.raises(TypeError):
                m.measure_tensor_network(tape)

    def test_not_supported_ham_sum_shot_measurements(self, lightning_tn):
        """Test to see if an exception is raised when the measurement is not supported."""

        tensornetwork = lightning_tn(3)

        m = LightningTensorMeasurements(tensornetwork)

        ops = [qp.PauliX(0), qp.PauliZ(1)]

        obs_ham = qp.Hamiltonian([-1.0, 1.5], [qp.Z(1), qp.X(1)])

        obs_sum = qp.sum(qp.PauliX(0), qp.PauliX(1))

        for mp in [qp.var(obs_ham), qp.var(obs_sum)]:
            tape = qp.tape.QuantumScript(ops, [mp], shots=100)

            with pytest.raises(TypeError):
                m.measure_tensor_network(tape)

    def test_not_supported_shadowmp_shot_measurements(self, lightning_tn):
        """Test than a TypeError is raised if the measurement is not supported."""

        tensornetwork = lightning_tn(3)

        m = LightningTensorMeasurements(tensornetwork)

        ops = [qp.PauliX(0), qp.PauliZ(1)]

        for mp in [qp.classical_shadow(wires=[0, 1]), qp.shadow_expval(qp.PauliX(0))]:
            tape = qp.tape.QuantumScript(ops, [mp], shots=100)

            with pytest.raises(TypeError):
                m.measure_tensor_network(tape)

    @pytest.mark.parametrize("method", [{"method": "mps", "max_bond_dim": 128}, {"method": "tn"}])
    @pytest.mark.parametrize("n_qubits", range(4, 14, 2))
    @pytest.mark.parametrize("n_targets", list(range(1, 4)) + list(range(4, 14, 2)))
    def test_probs_many_wires(self, method, n_qubits, n_targets, tol):
        """Test probs measuring many wires of a random quantum state."""
        if n_targets >= n_qubits:
            pytest.skip("Number of targets cannot exceed the number of wires.")

        dev = qp.device(device_name, wires=n_qubits, **method)
        dq = qp.device("default.qubit", wires=n_qubits)

        init_state = get_random_normalized_state(2**n_qubits)

        ops = [qp.StatePrep(init_state, wires=range(n_qubits))]

        mp = qp.probs(wires=range(n_targets))

        tape = qp.tape.QuantumScript(ops, [mp])
        ref = dq.execute(tape)

        if method["method"] == "tn":
            with pytest.raises(DeviceError):
                res = dev.execute(tape)
        else:
            res = dev.execute(tape)
            assert np.allclose(res, ref, atol=tol, rtol=0)

    @pytest.mark.parametrize("method", [{"method": "mps", "max_bond_dim": 128}, {"method": "tn"}])
    @pytest.mark.parametrize("n_qubits", range(4, 14, 2))
    @pytest.mark.parametrize("n_targets", list(range(1, 4)) + list(range(4, 14, 2)))
    def test_state_many_wires(self, method, n_qubits, n_targets, tol):
        """Test probs measuring many wires of a random quantum state."""
        if n_targets >= n_qubits:
            pytest.skip("Number of targets cannot exceed the number of wires.")

        dev = qp.device(device_name, wires=n_qubits, **method)
        dq = qp.device("default.qubit", wires=n_qubits)

        init_state = get_random_normalized_state(2**n_qubits)

        ops = [qp.StatePrep(init_state, wires=range(n_qubits))]

        mp = qp.state()

        tape = qp.tape.QuantumScript(ops, [mp])
        ref = dq.execute(tape)
        if method["method"] == "tn":
            with pytest.raises(DeviceError):
                res = dev.execute(tape)
        else:
            res = dev.execute(tape)
            assert np.allclose(res, ref, atol=tol, rtol=0)
