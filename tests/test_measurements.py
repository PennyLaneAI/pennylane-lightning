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
Unit tests for Measurements in Lightning devices.
"""
import itertools
import math
from typing import Sequence

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import device_name, lightning_ops, validate_measurements
from pennylane.exceptions import DeviceError, QuantumFunctionError
from pennylane.measurements import ExpectationMP, Shots, VarianceMP

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_no_measure():
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qml.device(device_name, wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.PauliY(0)

    with pytest.raises(QuantumFunctionError, match="must return either a single measurement"):
        circuit(0.65)


class TestProbs:
    """Test Probs in Lightning devices"""

    @pytest.fixture(params=itertools.product([np.complex64, np.complex128], [None, 2]))
    def dev(self, request):
        return qml.device(device_name, wires=request.param[1], c_dtype=request.param[0])

    @pytest.mark.parametrize(
        "wire, expected", [(0, [0.5, 0.0, 0.5, 0.0]), (1, [0.5, 0.5, 0.0, 0.0])]
    )
    def test_probs_H(self, wire, expected, tol, dev):
        """Test probs with Hadamard"""

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=wire)
            return qml.probs(wires=[0, 1])

        if device_name == "lightning.tensor" and wire == 1 and dev.num_wires is None:
            with pytest.raises(RuntimeError, match="Invalid wire indices order"):
                # With dynamic wires, in this case since wires appear in this order 1, 0
                # The wires will map 1 -> 0 and 0 -> 1. Therefore the wires in the probs
                # measurement will be [1, 0] which is out of order and invalid for LT.
                circuit()
        else:
            assert np.allclose(circuit(), expected, atol=tol, rtol=0)

    def test_probs_tape_none_wires(self, tol, dev):
        """Test probs with a circuit with wires=None"""

        x, y, z = [0.5, 0.3, -0.7]
        expected = [0.903281826, 0.00909338007, 0.0867514634, 0.000873331009]
        wires = None

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.probs(wires=wires)

        assert np.allclose(circuit(), expected, atol=tol, rtol=0)

    def test_probs_tape_empty_wires(self, dev):
        """Test that probs with empty list for wires raises an error"""

        x, y, z = [0.5, 0.3, -0.7]
        wires = []

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=wires)

        with pytest.raises(ValueError, match="Cannot set an empty list of wires"):
            circuit()

    @pytest.mark.parametrize(
        "cases",
        [
            [[0, 1], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [0, [0.9165164490394898, 0.08348355096051052]],
            [[0], [0.9165164490394898, 0.08348355096051052]],
        ],
    )
    def test_probs_tape_wire0(self, cases, tol, dev):
        """Test probs with a circuit on wires=[0]"""

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        if (
            device_name == "lightning.tensor"
            and (isinstance(cases[0], int) or len(cases[0]) < 2)
            and dev.num_wires is None
        ):
            with pytest.raises(ValueError, match="Number of wires must be greater than 1"):
                circuit()
        else:
            assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name in ("lightning.tensor"),
        reason="lightning.tensor does not support out of order prob.",
    )
    @pytest.mark.parametrize(
        "cases",
        [
            [[1, 0], [1, 0], [0.9165164490394898, 0.08348355096051052, 0.0, 0.0]],
            [[2, 0], [2, 0, 1], [0.9165164490394898, 0.08348355096051052, 0.0, 0.0]],
        ],
    )
    def test_probs_matching_device_wire_order(self, cases, tol):
        """Test probs with a circuit on wires=[0] passes if wires are sorted wrt device wires."""

        x, y, z = [0.5, 0.3, -0.7]
        dev = qml.device(device_name, wires=cases[1])

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[2], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [
                [0, 1],
                [
                    0.9178264236525453,
                    0.02096485729264079,
                    0.059841820910257436,
                    0.0013668981445561978,
                ],
            ],
            [0, [0.938791280945186, 0.061208719054813635]],
            [[0], [0.938791280945186, 0.061208719054813635]],
        ],
    )
    def test_probs_tape_wire01(self, cases, tol, dev):
        """Test probs with a circuit on wires=[0,1]"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.5, wires=[0])
            qml.RY(0.3, wires=[1])
            return qml.probs(wires=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliZ(0), [0.91237521, 0.08762479]],
            [qml.PauliZ(1), [0.99003329, 0.00996671]],
            [qml.PauliY(0), [0.22418248, 0.77581752]],
            [qml.PauliY(1), [0.5, 0.5]],
            [qml.PauliX(0), [0.56222044, 0.43777956]],
            [qml.PauliX(1), [0.40066533, 0.59933467]],
            [qml.Hadamard(0), [0.8355898, 0.1644102]],
            [qml.Hadamard(1), [0.77626565, 0.22373435]],
        ],
    )
    def test_probs_named_op(self, cases, tol, dev):
        """Test probs with a named observable"""

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.probs(op=cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    def test_probs_named_op_with_wires(self, dev):
        """Test probs with a circuit on wires=[0]"""

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.probs(op=qml.PauliZ(0), wires=[0])

        with pytest.raises(QuantumFunctionError, match="Cannot specify the wires to probs"):
            circuit()


class TestExpval:
    """Tests for the expval function"""

    @pytest.fixture(params=itertools.product([np.complex64, np.complex128], [None, 2]))
    def dev(self, request):
        return qml.device(device_name, wires=request.param[1], c_dtype=request.param[0])

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), 0.0],
            [qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0), -0.3894183423086505],
            [qml.PauliY(1), 0.0],
            [qml.PauliZ(0), 0.9210609940028852],
            [qml.PauliZ(1), 0.9800665778412417],
        ],
    )
    def test_expval_wire01(self, cases, tol, dev):
        """Test expval with a circuit on wires=[0,1]"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "obs, coeffs, res",
        [
            ([qml.PauliX(0) @ qml.PauliZ(1)], [1.0], 0.0),
            ([qml.PauliZ(0) @ qml.PauliZ(1)], [1.0], math.cos(0.4) * math.cos(-0.2)),
            (
                [
                    qml.PauliX(0) @ qml.PauliZ(1),
                    (
                        qml.Hermitian(
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 3.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0, 1.0],
                                [0.0, 0.0, 1.0, -2.0],
                            ],
                            wires=[0, 1],
                        )
                        if device_name != "lightning.tensor"
                        else qml.Hermitian([[1.0, 0.0], [0.0, 1.0]], wires=[0])
                    ),
                ],
                [0.3, 1.0],
                0.9319728930156066 if device_name != "lightning.tensor" else 1.0,
            ),
        ],
    )
    def test_expval_hamiltonian(self, obs, coeffs, res, tol, dev):
        """Test expval with Hamiltonian"""
        ham = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(ham)

        assert np.allclose(circuit(), res, atol=tol, rtol=0)

    def test_value(self, dev, tol):
        """Test that the expval interface works"""

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RX(x, wires=1)
            return qml.expval(qml.PauliY(0))

        x = 0.54
        res = circuit(x)
        expected = -np.sin(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self, dev):
        """Test that a QuantumFunctionError is raised if the provided
        argument is not an observable"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval(qml.RX(0.742, wires=[0]))

        with pytest.raises(DeviceError, match="Observable RX.*not supported"):
            circuit()

    def test_observable_return_type_is_expectation(self, dev):
        """Test that the return type of the observable is :class:`ExpectationMP`"""

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            res = qml.expval(qml.PauliZ(1))
            assert isinstance(res, ExpectationMP)
            return res

        circuit()


class TestVar:
    """Tests for the var function"""

    @pytest.fixture(params=itertools.product([np.complex64, np.complex128], [None, 2]))
    def dev(self, request):
        return qml.device(device_name, wires=request.param[1], c_dtype=request.param[0])

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), 0.9982450376077382],
            [qml.PauliX(1), 1.0],
            [qml.PauliY(0), 0.6956987716741251],
            [qml.PauliY(1), 1.0],
            [qml.PauliZ(0), 0.3060561907181374],
            [qml.PauliZ(1), -4.440892098500626e-16],
        ],
    )
    def test_var_qml_tape_wire0(self, cases, tol, dev):
        """Test var with a circuit on wires=[0]"""

        x, y, z = [0.5, 0.3, -0.7]

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            return qml.var(cases[0])

        if (
            device_name == "lightning.tensor"
            and cases[0].wires.tolist() == [0]
            and dev.num_wires is None
        ):
            with pytest.raises(ValueError, match="Number of wires must be greater than 1"):
                circuit()
        else:
            assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0), 1.0],
            [qml.PauliX(1), 0.9605304970014426],
            [qml.PauliY(0), 0.8483533546735826],
            [qml.PauliY(1), 1.0],
            [qml.PauliZ(0), 0.15164664532641725],
            [qml.PauliZ(1), 0.03946950299855745],
        ],
    )
    def test_var_qml_tape_wire01(self, cases, tol, dev):
        """Test var with a circuit on wires=[0,1]"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.var(cases[0])

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)

    def test_value(self, dev, tol):
        """Test that the var function works"""

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(x, wires=1)
            return qml.var(qml.PauliZ(0))

        x = 0.54
        res = circuit(x)
        expected = np.sin(x) ** 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self, dev):
        """Test that a QuantumFunctionError is raised if the provided
        argument is not an observable"""

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.var(qml.RX(0.742, wires=[0]))

        with pytest.raises(DeviceError, match="Observable RX.*not supported"):
            circuit()

    def test_observable_return_type_is_variance(self, dev):
        """Test that the return type is :class:`VarianceMP`"""

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(1)
            res = qml.var(qml.PauliZ(0))
            assert isinstance(res, VarianceMP)
            return res

        circuit()


class TestSample:
    """Tests that samples are properly calculated."""

    @pytest.mark.parametrize(
        "shots, wires",
        [
            [10, [0]],
            [12, [1]],
            [17, [0, 1]],
        ],
    )
    def test_sample_dimensions(self, qubit_device, shots, wires):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """
        dev = qubit_device(wires=2)
        ops = [qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])]
        obs = qml.PauliZ(wires=[0])
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=obs)], shots=shots)
        s1 = dev.execute(tape)
        assert np.array_equal(s1.shape, (shots,))

    def test_sample_values(self, qubit_device, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        shots = 1000
        dev = qubit_device(wires=2)
        ops = [qml.RX(1.5708, wires=[0])]
        obs = qml.PauliZ(0)
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=obs)], shots=shots)
        s1 = dev.execute(tape)

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("seed", range(0, 10))
    @pytest.mark.parametrize("nwires", range(1, 11))
    def test_sample_variations(self, qubit_device, nwires, seed):
        """Tests if `sample(wires)` returns correct statistics."""
        shots = 200000
        n_qubits = max(5, nwires + 1)

        rng = np.random.default_rng(seed)
        wires = qml.wires.Wires(rng.permutation(nwires))
        state = rng.random(2**n_qubits) + 1j * rng.random(2**n_qubits)
        state[rng.integers(0, 2**n_qubits, 1)] += state.size / 10
        state /= np.linalg.norm(state)
        ops = [qml.StatePrep(state, wires=range(n_qubits))]
        tape = qml.tape.QuantumScript(ops, [qml.sample(wires=wires)], shots=shots)
        tape_exact = qml.tape.QuantumScript(ops, [qml.probs(wires=wires)])

        dev = qubit_device(wires=n_qubits)
        samples = dev.execute(tape)
        probs = qml.measurements.ProbabilityMP(wires=wires).process_samples(
            np.atleast_2d(samples), wire_order=wires
        )

        dev_ref = qml.device("default.qubit", wires=n_qubits)
        probs_ref = dev_ref.execute(tape_exact)

        assert np.allclose(probs, probs_ref, atol=2.0e-2, rtol=1.0e-4)


@pytest.mark.local_salt(42)
@pytest.mark.parametrize("shots", [None, 100000, [100000, 111111]])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
@pytest.mark.parametrize(
    "obs",
    [
        None,
        [],
        [0],
        [0, 1],
        qml.PauliZ(0),
        qml.PauliY(1),
        qml.PauliZ(0) @ qml.PauliY(1),
        qml.PauliZ(1) @ qml.PauliY(2),
    ],
)
@pytest.mark.parametrize("mcmc", [False, True])
@pytest.mark.parametrize("n_wires", [None, 3])
@pytest.mark.parametrize("kernel_name", ["Local", "NonZeroRandom"])
def test_shots_single_measure_obs(shots, measure_f, obs, n_wires, mcmc, kernel_name, seed):
    """Tests that Lightning handles shots in a circuit where a single measurement of a common observable is performed at the end."""

    if (
        shots is None or device_name in ("lightning.gpu", "lightning.kokkos", "lightning.tensor")
    ) and (mcmc or kernel_name != "Local"):
        pytest.skip(f"Device {device_name} does not have an mcmc option.")

    if measure_f in (qml.expval, qml.var) and isinstance(obs, Sequence):
        pytest.skip("qml.expval, qml.var do not take wire arguments.")

    if measure_f in (qml.counts, qml.sample) and shots is None:
        pytest.skip("qml.counts, qml.sample do not work with shots = None.")

    if measure_f in (qml.expval, qml.var) and obs is None:
        pytest.skip("qml.expval, qml.var requires observable.")

    if device_name in ("lightning.gpu", "lightning.kokkos"):
        dev = qml.device(device_name, wires=n_wires, seed=seed)
    elif device_name == "lightning.qubit":
        dev = qml.device(
            device_name,
            wires=n_wires,
            mcmc=mcmc,
            kernel_name=kernel_name,
            num_burnin=100,
            seed=seed,
        )
    else:
        dev = qml.device(device_name, wires=n_wires)

    dq = qml.device("default.qubit", wires=n_wires, seed=seed)
    params = [np.pi / 4, -np.pi / 4]

    def func(x, y):
        qml.RX(x, 0)
        qml.RX(y, 0)
        qml.RX(y, 1)
        qml.RX(x, 2)
        return measure_f(wires=obs) if isinstance(obs, Sequence) else measure_f(op=obs)

    if obs == []:
        with pytest.raises(ValueError, match="Cannot set an empty list of wires"):
            qn = qml.QNode(func, dev)
            qn = qml.set_shots(qn, shots=shots)
            qn(*params)
    else:
        func1 = qml.QNode(func, dev)
        func1 = qml.set_shots(func1, shots=shots)
        results1 = func1(*params)

        func2 = qml.QNode(func, dq)
        func2 = qml.set_shots(func2, shots=shots)
        results2 = func2(*params)

        validate_measurements(measure_f, shots, results1, results2)


# TODO: Add LT after extending the support for shots_vector
@pytest.mark.local_salt(42)
@pytest.mark.skipif(
    device_name == "lightning.tensor",
    reason="lightning.tensor does not support single-wire devices.",
)
@pytest.mark.parametrize("shots", ((1, 10), (1, 10, 100), (1, 10, 10, 100, 100, 100)))
def test_shots_bins(shots, qubit_device, seed):
    """Tests that Lightning handles multiple shots."""

    dev = qubit_device(wires=1, seed=seed)

    @qml.set_shots(shots)
    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.PauliZ(wires=0))

    assert np.sum(shots) == circuit._shots.total_shots

    assert np.allclose(circuit(), 1.0)
