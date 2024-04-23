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
"""Tests for default qubit preprocessing."""
from functools import reduce
from typing import Sequence

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice, device_name, validate_measurements
from flaky import flaky
from pennylane._device import DeviceError

if device_name not in ("lightning.qubit", "lightning.kokkos"):
    pytest.skip("Native MCM not supported. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_all_invalid_shots_circuit():
    """Test all invalid cases: expval, probs, var measurements."""

    dev = qml.device(device_name, wires=2)
    dq = qml.device("default.qubit", wires=2)

    def circuit_op():
        m = qml.measure(0, postselect=1)
        qml.cond(m, qml.PauliX)(1)
        return (
            qml.expval(op=qml.PauliZ(1)),
            qml.probs(op=qml.PauliY(0) @ qml.PauliZ(1)),
            qml.var(op=qml.PauliZ(1)),
        )

    res1 = qml.QNode(circuit_op, dq)()
    res2 = qml.QNode(circuit_op, dev)(shots=10)
    for r1, r2 in zip(res1, res2):
        if isinstance(r1, Sequence):
            assert len(r1) == len(r2)
        assert np.all(np.isnan(r1))
        assert np.all(np.isnan(r2))

    def circuit_mcm():
        m = qml.measure(0, postselect=1)
        qml.cond(m, qml.PauliX)(1)
        return qml.expval(op=m), qml.probs(op=m), qml.var(op=m)

    res1 = qml.QNode(circuit_mcm, dq)()
    res2 = qml.QNode(circuit_mcm, dev)(shots=10)
    for r1, r2 in zip(res1, res2):
        if isinstance(r1, Sequence):
            assert len(r1) == len(r2)
        assert np.all(np.isnan(r1))
        assert np.all(np.isnan(r2))


def test_unsupported_measurement():
    """Test unsupported ``qml.classical_shadow`` measurement on ``lightning.qubit``."""

    dev = qml.device(device_name, wires=2, shots=1000)
    params = np.pi / 4 * np.ones(2)

    @qml.qnode(dev)
    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return qml.classical_shadow(wires=0)

    if device_name == "lightning.qubit":
        with pytest.raises(
            DeviceError,
            match=f"not accepted with finite shots on lightning.qubit",
        ):
            func(*params)
    else:
        with pytest.raises(
            TypeError,
            match=f"Native mid-circuit measurement mode does not support ClassicalShadowMP measurements.",
        ):
            func(*params)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [5000, [5000, 5001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_simple_mcm(shots, postselect, reset, measure_f):
    """Tests that LightningDevice handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of the mid-circuit measurement value is performed at
    the end."""

    dev = qml.device(device_name, wires=1, shots=shots)
    dq = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)

    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=0)
        return measure_f(op=qml.PauliZ(0))

    func1 = qml.QNode(func, dev)
    func2 = qml.defer_measurements(qml.QNode(func, dq))

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [1000, [1000, 1001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_single_mcm_single_measure_mcm(shots, postselect, reset, measure_f):
    """Tests that LightningDevice handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of the mid-circuit measurement value is performed at
    the end."""

    dev = qml.device(device_name, wires=2, shots=shots)
    dq = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)

    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=m0)

    func1 = qml.QNode(func, dev)
    func2 = qml.defer_measurements(qml.QNode(func, dq))

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


# pylint: disable=unused-argument
def obs_tape(x, y, z, reset=False, postselect=None):
    qml.RX(x, 0)
    qml.RZ(np.pi / 4, 0)
    m0 = qml.measure(0, reset=reset)
    qml.cond(m0 == 0, qml.RX)(np.pi / 4, 0)
    qml.cond(m0 == 0, qml.RZ)(np.pi / 4, 0)
    qml.cond(m0 == 1, qml.RX)(-np.pi / 4, 0)
    qml.cond(m0 == 1, qml.RZ)(-np.pi / 4, 0)
    qml.RX(y, 1)
    qml.RZ(np.pi / 4, 1)
    m1 = qml.measure(1, postselect=postselect)
    qml.cond(m1 == 0, qml.RX)(np.pi / 4, 1)
    qml.cond(m1 == 0, qml.RZ)(np.pi / 4, 1)
    qml.cond(m1 == 1, qml.RX)(-np.pi / 4, 1)
    qml.cond(m1 == 1, qml.RZ)(-np.pi / 4, 1)
    return m0, m1


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [5000, [5000, 5001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
@pytest.mark.parametrize("obs", [qml.PauliZ(0), qml.PauliY(1), qml.PauliZ(0) @ qml.PauliY(1)])
def test_single_mcm_single_measure_obs(shots, postselect, reset, measure_f, obs):
    """Tests that LightningQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of a common observable is performed at the end."""

    dev = qml.device(device_name, wires=2, shots=shots)
    dq = qml.device("default.qubit", shots=shots)
    params = [np.pi / 7, np.pi / 6, -np.pi / 5]

    def func(x, y, z):
        obs_tape(x, y, z, reset=reset, postselect=postselect)
        return measure_f(op=obs)

    func1 = qml.QNode(func, dev)
    func2 = qml.defer_measurements(qml.QNode(func, dq))

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [3000, [3000, 3001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.probs, qml.sample])
@pytest.mark.parametrize("wires", [[0], [0, 1]])
def test_single_mcm_single_measure_wires(shots, postselect, reset, measure_f, wires):
    """Tests that LightningDevice handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of one or several wires is performed at the end."""

    dev = qml.device(device_name, wires=2, shots=shots)
    dq = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)

    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(wires=wires)

    func1 = qml.QNode(func, dev)
    func2 = qml.defer_measurements(qml.QNode(func, dq))

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [5000])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_single_mcm_multiple_measurements(shots, postselect, reset, measure_f):
    """Tests that LightningDevice handles a circuit with a single mid-circuit measurement with reset
    and a conditional gate. Multiple measurements of the mid-circuit measurement value are
    performed."""

    dev = qml.device(device_name, wires=2, shots=shots)
    dq = qml.device("default.qubit", shots=shots)
    params = [np.pi / 7, np.pi / 6, -np.pi / 5]
    obs = qml.PauliY(1)

    def func(x, y, z):
        mcms = obs_tape(x, y, z, reset=reset, postselect=postselect)
        return measure_f(op=obs), measure_f(op=mcms[0])

    func1 = qml.QNode(func, dev)
    func2 = qml.defer_measurements(qml.QNode(func, dq))

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        for r1, r2 in zip(results1, results2):
            validate_measurements(measure_f, shots, r1, r2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [5000, [5000, 5001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.sample, qml.var])
def test_composite_mcm_measure_composite_mcm(shots, postselect, reset, measure_f):
    """Tests that LightningDevice handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    dev = qml.device(device_name, wires=2, shots=shots)
    dq = qml.device("default.qubit", shots=shots)
    param = np.pi / 3

    def func(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        return measure_f(op=(m0 - 2 * m1) * m2 + 7)

    func1 = qml.QNode(func, dev)
    func2 = qml.defer_measurements(qml.QNode(func, dq))

    results1 = func1(param)
    results2 = func2(param)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [10000, [10000, 10001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_composite_mcm_single_measure_obs(shots, postselect, reset, measure_f):
    """Tests that LightningDevice handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a common observable is performed at the end."""

    dev = qml.device(device_name, wires=2, shots=shots)
    dq = qml.device("default.qubit", shots=shots)
    params = [np.pi / 7, np.pi / 6, -np.pi / 5]
    obs = qml.PauliZ(0) @ qml.PauliY(1)

    def func(x, y, z):
        mcms = obs_tape(x, y, z, reset=reset, postselect=postselect)
        qml.cond(mcms[0] != mcms[1], qml.RY)(z, wires=0)
        qml.cond(mcms[0] == mcms[1], qml.RY)(z, wires=1)
        return measure_f(op=obs)

    func1 = qml.QNode(func, dev)
    func2 = qml.defer_measurements(qml.QNode(func, dq))

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [5000, [5000, 5001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.probs, qml.sample])
def test_composite_mcm_measure_value_list(shots, postselect, reset, measure_f):
    """Tests that LightningDevice handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    dev = qml.device(device_name, wires=2, shots=shots)
    dq = qml.device("default.qubit", shots=shots)
    param = np.pi / 3

    def func(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        return measure_f(op=[m0, m1, m2])

    func1 = qml.QNode(func, dev)
    func2 = qml.defer_measurements(qml.QNode(func, dq))

    results1 = func1(param)
    results2 = func2(param)

    validate_measurements(measure_f, shots, results1, results2)
