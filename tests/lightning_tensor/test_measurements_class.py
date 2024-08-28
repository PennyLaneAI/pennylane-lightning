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
from typing import Sequence

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice, device_name  # tested device
from flaky import flaky
from pennylane.devices import DefaultQubit
from pennylane.measurements import VarianceMP
from scipy.sparse import csr_matrix, random_array

if device_name != "lightning.tensor":
    pytest.skip(
        "Skipping tests for the LightningTensorMeasurements class.", allow_module_level=True
    )

from pennylane_lightning.lightning_tensor._measurements import LightningTensorMeasurements
from pennylane_lightning.lightning_tensor._tensornet import LightningTensorNet

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)


# General LightningTensorNet fixture, for any number of wires.
@pytest.fixture(
    params=[np.complex64, np.complex128],
)
def lightning_tn(request):
    """Fixture for creating a LightningTensorNet object."""

    def _lightning_tn(n_wires):
        return LightningTensorNet(num_wires=n_wires, max_bond_dim=128, c_dtype=request.param)

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

        mp = qml.counts(wires=(0, 1))
        with pytest.raises(NotImplementedError):
            m.get_measurement_function(mp)


def get_hermitian_matrix(n):
    H = np.random.rand(n, n) + 1.0j * np.random.rand(n, n)
    return H + np.conj(H).T


def get_sparse_hermitian_matrix(n):
    H = random_array((n, n), density=0.15)
    H = H + 1.0j * random_array((n, n), density=0.15)
    return csr_matrix(H + H.conj().T)


class TestMeasurements:
    """Tests all measurements"""

    @staticmethod
    def calculate_reference(tape, lightning_tn):
        use_default = True
        new_meas = []
        for m in tape.measurements:
            # NotImplementedError in DefaultQubit
            # We therefore validate against `qml.Hermitian`
            if isinstance(m, VarianceMP) and isinstance(
                m.obs, (qml.Hamiltonian, qml.SparseHamiltonian)
            ):
                use_default = False
                new_meas.append(m.__class__(qml.Hermitian(qml.matrix(m.obs), wires=m.obs.wires)))
                continue
            new_meas.append(m)
        if use_default:
            dev = DefaultQubit(max_workers=1)
            program, _ = dev.preprocess()
            tapes, transf_fn = program([tape])
            results = dev.execute(tapes)
            return transf_fn(results)

        tape = qml.tape.QuantumScript(tape.operations, new_meas)
        tensornet = lightning_tn(tape.num_wires)
        tensornet.set_tensor_network(tape)
        m = LightningTensorMeasurements(tensornet)
        return m.measure_tensor_network(tape)

    # @flaky(max_runs=5)
    @pytest.mark.parametrize("measurement", [qml.expval, qml.probs, qml.var])
    @pytest.mark.parametrize(
        "observable",
        (
            [0],
            [1, 2],
            qml.PauliX(0),
            qml.PauliY(1),
            qml.PauliZ(2),
            qml.sum(qml.PauliX(0), qml.PauliY(0)),
            qml.prod(qml.PauliX(0), qml.PauliY(1)),
            qml.s_prod(2.0, qml.PauliX(0)),
        ),
    )
    def test_single_return_value(self, measurement, observable, lightning_tn, tol):
        if measurement is qml.probs and isinstance(
            observable,
            (qml.ops.Sum, qml.ops.SProd, qml.ops.Prod, qml.Hamiltonian, qml.SparseHamiltonian),
        ):
            pytest.skip(
                f"Observable of type {type(observable).__name__} is not supported for rotating probabilities."
            )

        if measurement is not qml.probs and isinstance(observable, list):
            pytest.skip(
                f"Measurement of type {type(measurement).__name__} does not have a keyword argument 'wires'."
            )

        n_qubits = 5
        np.random.seed(0)
        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        measurements = (
            [measurement(wires=observable)]
            if isinstance(observable, list)
            else [measurement(op=observable)]
        )
        tape = qml.tape.QuantumScript(ops, measurements)

        expected = self.calculate_reference(tape, lightning_tn)
        tensornet = lightning_tn(n_qubits)
        tensornet.set_tensor_network(tape)
        m = LightningTensorMeasurements(tensornet)
        result = m.measure_tensor_network(tape)

        # a few tests may fail in single precision, and hence we increase the tolerance
        assert np.allclose(result, expected, max(tol, 1.0e-4))

    @flaky(max_runs=5)
    @pytest.mark.parametrize("shots", [None, 1000000])
    @pytest.mark.parametrize("measurement", [qml.expval, qml.probs, qml.var])
    @pytest.mark.parametrize(
        "obs0_",
        (
            qml.PauliX(0),
            qml.PauliY(1),
            qml.PauliZ(2),
            qml.sum(qml.PauliX(0), qml.PauliY(0)),
            qml.prod(qml.PauliX(0), qml.PauliY(1)),
            qml.s_prod(2.0, qml.PauliX(0)),
            qml.Hermitian(get_hermitian_matrix(2), wires=[0]),
            # qml.Hamiltonian(
            #    [1.0, 2.0, 3.0], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2) @ qml.PauliZ(3)]
            # ),
        ),
    )
    @pytest.mark.parametrize(
        "obs1_",
        (
            qml.PauliX(0),
            qml.PauliY(1),
            qml.PauliZ(2),
            qml.sum(qml.PauliX(0), qml.PauliY(0)),
            qml.prod(qml.PauliX(0), qml.PauliY(1)),
            qml.s_prod(2.0, qml.PauliX(0)),
            qml.Hermitian(get_hermitian_matrix(2), wires=[0]),
        ),
    )
    def test_double_return_value(self, shots, measurement, obs0_, obs1_, lightning_tn, tol):
        skip_list = (
            qml.ops.Sum,
            qml.ops.SProd,
            qml.ops.Prod,
            qml.Hamiltonian,
            qml.SparseHamiltonian,
        )
        if measurement is qml.probs and (
            isinstance(obs0_, skip_list) or isinstance(obs1_, skip_list)
        ):
            pytest.skip(
                f"Observable of type {type(obs0_).__name__} is not supported for rotating probabilities."
            )

        n_qubits = 4
        np.random.seed(0)
        ops = [qml.Hadamard(i) for i in range(n_qubits)]
        measurements = [measurement(op=obs0_), measurement(op=obs1_)]
        tape = qml.tape.QuantumScript(ops, measurements, shots=shots)

        tensornet = lightning_tn(n_qubits)
        tensornet.set_tensor_network(tape)
        m = LightningTensorMeasurements(tensornet)

        skip_list = (
            qml.ops.Sum,
            qml.Hamiltonian,
            qml.SparseHamiltonian,
        )
        do_skip = measurement is qml.var and (
            isinstance(obs0_, skip_list) or isinstance(obs1_, skip_list)
        )
        do_skip = do_skip or (
            measurement is qml.expval
            and (
                isinstance(obs0_, qml.SparseHamiltonian) or isinstance(obs1_, qml.SparseHamiltonian)
            )
        )
        do_skip = do_skip and shots is not None
        if do_skip:
            with pytest.raises(TypeError):
                _ = m.measure_tensor_network(tape)
            return
        else:
            result = m.measure_tensor_network(tape)

        expected = self.calculate_reference(tape, lightning_tn)
        if len(expected) == 1:
            expected = expected[0]

        assert isinstance(result, Sequence)
        assert len(result) == len(expected)
        # a few tests may fail in single precision, and hence we increase the tolerance
        dtol = tol if shots is None else max(tol, 1.0e-2)
        for r, e in zip(result, expected):
            assert np.allclose(r, e, atol=dtol, rtol=dtol)
