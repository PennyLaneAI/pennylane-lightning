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

if device_name not in ("lightning.qubit", "lightning.kokkos", "lightning.gpu"):
    pytest.skip("Native MCM not supported. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def get_device(wires, **kwargs):
    kwargs.setdefault("shots", None)
    return qml.device(device_name, wires=wires, **kwargs)


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
    """Test unsupported ``qml.classical_shadow`` measurement on ``lightning.qubit`` or ``lightning.kokkos`` ."""

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
            qml.DeviceError,
            match=f"not accepted with finite shots on lightning.qubit",
        ):
            func(*params)
    if device_name in ("lightning.kokkos", "lightning.gpu"):
        with pytest.raises(
            qml.DeviceError,
            match=r"Measurement shadow\(wires=\[0\]\) not accepted with finite shots on "
            + device_name,
        ):
            func(*params)


@pytest.mark.parametrize("mcm_method", ["deferred", "one-shot"])
def test_qnode_mcm_method(mcm_method, mocker):
    """Test that user specified qnode arg for mid-circuit measurements transform are used correctly"""
    spy = (
        mocker.spy(qml.dynamic_one_shot, "_transform")
        if mcm_method == "one-shot"
        else mocker.spy(qml.defer_measurements, "_transform")
    )
    other_spy = (
        mocker.spy(qml.defer_measurements, "_transform")
        if mcm_method == "one-shot"
        else mocker.spy(qml.dynamic_one_shot, "_transform")
    )

    shots = 10
    device = qml.device(device_name, wires=3, shots=shots)

    @qml.qnode(device, mcm_method=mcm_method)
    def f(x):
        qml.RX(x, 0)
        _ = qml.measure(0)
        qml.CNOT([0, 1])
        return qml.sample(wires=[0, 1])

    _ = f(np.pi / 8)

    spy.assert_called_once()
    other_spy.assert_not_called()


@pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
def test_qnode_postselect_mode(postselect_mode):
    """Test that user specified qnode arg for discarding invalid shots is used correctly"""
    shots = 100
    device = qml.device(device_name, wires=3, shots=shots)
    postselect = 1

    @qml.qnode(device, postselect_mode=postselect_mode)
    def f(x):
        qml.RX(x, 0)
        _ = qml.measure(0, postselect=postselect)
        qml.CNOT([0, 1])
        return qml.sample(wires=[1])

    # Using small-ish rotation angle ensures the number of valid shots will be less than the
    # original number of shots. This helps avoid stochastic failures for the assertion below
    res = f(np.pi / 2)

    if postselect_mode == "hw-like":
        assert len(res) < shots
    else:
        assert len(res) == shots
    assert np.allclose(res, postselect)


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
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
@pytest.mark.parametrize(
    "meas_obj",
    [qml.PauliZ(0), qml.PauliY(1), [0], [0, 1], [1, 0], "mcm", "composite_mcm", "mcm_list"],
)
def test_simple_dynamic_circuit(shots, measure_f, postselect, meas_obj):
    """Tests that LightningQubit handles a simple dynamic circuit with the following measurements:

        * qml.counts with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
        * qml.expval with obs (comp basis or not), MCM, f(MCM), MCM list
        * qml.probs with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
        * qml.sample with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
        * qml.var with obs (comp basis or not), MCM, f(MCM), MCM list

    The above combinations should work for finite shots, shot vectors and post-selecting of either the 0 or 1 branch.
    """

    if measure_f in (qml.expval, qml.var) and (
        isinstance(meas_obj, list) or meas_obj == "mcm_list"
    ):
        pytest.skip("Can't use wires/mcm lists with var or expval")

    dq = qml.device("default.qubit", shots=shots)
    dev = get_device(wires=3, shots=shots)
    params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]

    def func(x, y, z):
        m0, m1 = obs_tape(x, y, z, postselect=postselect)
        mid_measure = (
            m0 if meas_obj == "mcm" else (0.5 * m0 if meas_obj == "composite_mcm" else [m0, m1])
        )
        measurement_key = "wires" if isinstance(meas_obj, list) else "op"
        measurement_value = mid_measure if isinstance(meas_obj, str) else meas_obj
        return measure_f(**{measurement_key: measurement_value})

    results1 = qml.QNode(func, dev, mcm_method="one-shot")(*params)
    results2 = qml.QNode(func, dq, mcm_method="deferred")(*params)

    validate_measurements(measure_f, shots, results1, results2)


@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
def test_multiple_measurements_and_reset(postselect, reset):
    """Tests that LightningQubit handles a circuit with a single mid-circuit measurement with reset
    and a conditional gate. Multiple measurements of the mid-circuit measurement value are
    performed. This function also tests `reset` parametrizing over the parameter."""
    shots = 5000
    dq = qml.device("default.qubit", shots=shots)
    dev = get_device(wires=3, shots=shots)
    params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]
    obs = qml.PauliY(1)

    def func(x, y, z):
        mcms = obs_tape(x, y, z, reset=reset, postselect=postselect)
        return (
            qml.counts(op=obs),
            qml.expval(op=mcms[0]),
            qml.probs(op=obs),
            qml.sample(op=mcms[0]),
            qml.var(op=obs),
        )

    results1 = qml.QNode(func, dev, mcm_method="one-shot")(*params)
    results2 = qml.QNode(func, dq, mcm_method="deferred")(*params)

    for measure_f, r1, r2 in zip(
        [qml.counts, qml.expval, qml.probs, qml.sample, qml.var], results1, results2
    ):
        validate_measurements(measure_f, shots, r1, r2)


@pytest.mark.parametrize(
    "mcm_f",
    [
        lambda x: x * -1,
        lambda x: x * 1,
        lambda x: x * 2,
        lambda x: 1 - x,
        lambda x: x + 1,
        lambda x: x & 3,
        "mix",
        "list",
    ],
)
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_composite_mcms(mcm_f, measure_f):
    """Tests that LightningQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    if measure_f in (qml.expval, qml.var) and (mcm_f in ("list", "mix")):
        pytest.skip(
            "expval/var does not support measuring sequences of measurements or observables."
        )

    if measure_f == qml.probs and mcm_f == "mix":
        pytest.skip(
            "Cannot use qml.probs() when measuring multiple mid-circuit measurements collected using arithmetic operators."
        )

    shots = 3000

    dq = qml.device("default.qubit", shots=shots)
    dev = get_device(wires=3, shots=shots)
    param = np.pi / 3

    @qml.qnode(dev)
    def func(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        obs = (
            (m0 - 2 * m1) * m2 + 7
            if mcm_f == "mix"
            else ([m0, m1, m2] if mcm_f == "list" else mcm_f(m2))
        )
        return measure_f(op=obs)

    results1 = qml.QNode(func, dev, mcm_method="one-shot")(param)
    results2 = qml.QNode(func, dq, mcm_method="deferred")(param)

    validate_measurements(measure_f, shots, results1, results2)


@pytest.mark.parametrize(
    "mcm_f",
    [
        lambda x, y: x + y,
        lambda x, y: x - 7 * y,
        lambda x, y: x & y,
        lambda x, y: x == y,
        lambda x, y: 4.0 * x + 2.0 * y,
    ],
)
def test_counts_return_type(mcm_f):
    """Tests that LightningQubit returns the same keys for ``qml.counts`` measurements with ``dynamic_one_shot`` and ``defer_measurements``."""
    shots = 500

    dq = qml.device("default.qubit", shots=shots)
    dev = get_device(wires=3, shots=shots)
    param = np.pi / 3

    @qml.qnode(dev)
    def func(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        return qml.counts(op=mcm_f(m0, m1))

    results1 = qml.QNode(func, dev, mcm_method="one-shot")(param)
    results2 = qml.QNode(func, dq, mcm_method="deferred")(param)
    for r1, r2 in zip(results1.keys(), results2.keys()):
        assert r1 == r2
