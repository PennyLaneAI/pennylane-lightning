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
from pennylane.exceptions import DeviceError

if device_name not in ("lightning.qubit", "lightning.kokkos", "lightning.gpu"):
    pytest.skip("Native MCM not supported. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def get_device(wires, **kwargs):
    kwargs.setdefault("shots", None)
    return qml.device(device_name, wires=wires, **kwargs)


@pytest.fixture(
    scope="function",
    params=[
        # "deferred",
        # "one-shot",
        "tree-traversal",
    ],
)
def mcm_method(request):
    """Fixture to set the MCM method for the tests."""
    return request.param
    



class TestUnsupportedConfigurationsMCM:
    """Test unsupported configurations for different mid-circuit measurement methods."""

    def generate_circuit(
        self,
        device_kwargs={},
        qnode_kwargs={},
        mcm_kwargs={},
        measurement=qml.expval,
        obs=qml.PauliZ(0),
    ):
        """Generate a circuit with a mid-circuit measurement."""
        dev = qml.device(device_name, **device_kwargs)

        print(f"device_kwargs: {device_kwargs}")
        print(f"qnode_kwargs: {qnode_kwargs}")
        print(f"mcm_kwargs: {mcm_kwargs}")

        @qml.qnode(dev, **qnode_kwargs)
        def func(y):
            qml.RX(y, wires=0)
            m0 = qml.measure(0, **mcm_kwargs)
            qml.cond(m0, qml.RY)(y, wires=1)
            return measurement(obs)

        return func

    def test_unsupported_method(self):

        method = "roller-coaster"
        circuit = self.generate_circuit(
            device_kwargs={"wires": 1, "shots": 100},
            qnode_kwargs={"mcm_method": method},
            mcm_kwargs={"postselect": None, "reset": False},
            measurement=qml.expval,
            obs=qml.PauliZ(0),
        )
        with pytest.raises(
            DeviceError,
            match=f"Unsupported mid-circuit measurement method '{method}' for device {device_name}",
        ):
            circuit(1.33)

    def test_unsupported_measurement(self, mcm_method):
        """Test unsupported ``qml.classical_shadow`` measurement on ``lightning.qubit`` or ``lightning.kokkos`` ."""

        circuit = self.generate_circuit(
            device_kwargs={"wires": 1, "shots": 100},
            qnode_kwargs={"mcm_method": mcm_method},
            mcm_kwargs={"postselect": None, "reset": False},
            measurement=qml.classical_shadow,
            obs=[0],
        )

        if device_name == "lightning.qubit":
            with pytest.raises(
                DeviceError,
                match=f"not accepted with finite shots on lightning.qubit",
            ):
                circuit(1.33)
        if device_name in ("lightning.kokkos", "lightning.gpu"):
            with pytest.raises(
                DeviceError,
                match=r"Measurement shadow\(wires=\[0\]\) not accepted with finite shots on "
                + device_name,
            ):
                circuit(1.33)

    def test_unsupported_configuration_deferred(self):
        """Test unsupported configuration for wires=1, shots=None"""

        circuit = self.generate_circuit(
            device_kwargs={"wires": 1, "shots": None},
            qnode_kwargs={"mcm_method": "deferred"},
            mcm_kwargs={"postselect": None, "reset": False},
            measurement=qml.expval,
            obs=qml.PauliZ(0),
        )

        with pytest.raises(
            qml.wires.WireError,
            match=f"on {device_name} as they contain wires not found on the device: {{1}}",
        ):
            circuit(1.33)

        for postsel in ["hw-like", "fill-shots"]:

            circuit = self.generate_circuit(
                device_kwargs={"wires": 1, "shots": 100},
                qnode_kwargs={"mcm_method": "deferred", "postselect_mode": postsel},
                mcm_kwargs={"postselect": 1, "reset": False},
                measurement=qml.expval,
                obs=qml.PauliZ(0),
            )

            with pytest.raises(
                ValueError,
                match="Postselection is not allowed on the device with deferred measurements. The device must support the Projector gate to apply postselection.",
            ):
                circuit(1.33)

    def test_unsupported_configuration_one_shot(self):
        """Test unsupported configuration for wires=1, shots=None"""

        circuit = self.generate_circuit(
            device_kwargs={"wires": 1, "shots": None},
            qnode_kwargs={"mcm_method": "one-shot"},
            mcm_kwargs={"postselect": None, "reset": False},
            measurement=qml.expval,
            obs=qml.PauliZ(0),
        )

        with pytest.raises(
            ValueError,
            match="Cannot use the 'one-shot' method for mid-circuit measurements with analytic mode.",
        ):
            circuit(1.33)

    def test_unsupported_configuration_tree_traversal(self):
        """Test unsupported configuration for wires=1, shots=None"""

        for measurement in [qml.counts, qml.sample]:
            circuit = self.generate_circuit(
                device_kwargs={"wires": 1, "shots": None},
                qnode_kwargs={"mcm_method": "tree-traversal"},
                mcm_kwargs={"postselect": None, "reset": False},
                measurement=measurement,
                obs=qml.PauliZ(0),
            )

            with pytest.raises(
                DeviceError,
                match="not accepted for analytic simulation on " + device_name,
            ):
                circuit(1.33)


class TestMCMSupportedConfigurationsMCM:

    @pytest.mark.parametrize("mcm_method", ["deferred", "one-shot"])
    def test_qnode_mcm_method(self, mcm_method, mocker):
        """Test that user specified qnode arg for mid-circuit measurements transform are used correctly"""
        mocker.stopall()
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

    @pytest.mark.parametrize("shots", [None, 10])
    def test_qnode_mock_mcm_method_tree_traversal(self, mocker, shots):
        """Test that user specified qnode arg for mid-circuit measurements transform are used correctly"""
        spy_one_shot = (mocker.spy(qml.dynamic_one_shot, "_transform"))
        spy_deferred = (mocker.spy(qml.defer_measurements, "_transform"))

        device = qml.device(device_name, wires=3, shots=shots)

        @qml.qnode(device, mcm_method="tree-traversal")
        def f(x):
            qml.RX(x, 0)
            _ = qml.measure(0)
            qml.CNOT([0, 1])
            return qml.expval(qml.PauliX(0))

        _ = f(np.pi / 8)

        spy_one_shot.assert_not_called()
        spy_deferred.assert_not_called()


    def test_qnode_default_mcm_method_analytical(self, mocker):
        """Test the default mcm method is used for analytical simulation"""
        spy = (mocker.spy(qml.defer_measurements, "_transform"))
        other_spy = (mocker.spy(qml.dynamic_one_shot, "_transform"))

        shots = None
        device = qml.device(device_name, wires=3, shots=shots)

        @qml.qnode(device)
        def f(x):
            qml.RX(x, 0)
            _ = qml.measure(0)
            qml.CNOT([0, 1])
            return qml.expval(qml.PauliX(0))

        _ = f(np.pi / 8)

        spy.assert_called_once()
        other_spy.assert_not_called()

    def test_qnode_default_mcm_method_finite_shots(self, mocker):
        """Test the default mcm method is used for finite shots"""
        other_spy = (mocker.spy(qml.defer_measurements, "_transform"))
        spy = (mocker.spy(qml.dynamic_one_shot, "_transform"))

        shots = 33
        device = qml.device(device_name, wires=3, shots=shots)

        @qml.qnode(device)
        def f(x):
            qml.RX(x, 0)
            _ = qml.measure(0)
            qml.CNOT([0, 1])
            return qml.expval(qml.PauliX(0))

        _ = f(np.pi / 8)

        spy.assert_called_once()
        other_spy.assert_not_called()


    @pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
    def test_qnode_postselect_mode(self, mcm_method, postselect_mode):
        """Test that user specified qnode arg for discarding invalid shots is used correctly"""

        if mcm_method == "deferred":
            pytest.skip(reason="Deferred does not support postselection")

        shots = 100
        device = qml.device(device_name, wires=3, shots=shots)
        postselect = 1

        @qml.qnode(device, postselect_mode=postselect_mode, mcm_method=mcm_method)
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

class TestExecutionMCM:

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

    def test_all_invalid_shots_circuit(self,mcm_method):
        """Test all invalid cases: expval, probs, var measurements."""
        
        if mcm_method == "deferred":
            pytest.skip("Deferred does not support postselection")

        dev = qml.device(device_name, wires=2)
        dq = qml.device("default.qubit", wires=3)

        def circuit_op():
            m = qml.measure(0, postselect=1)
            qml.cond(m, qml.PauliX)(1)
            return (
                qml.expval(op=qml.PauliZ(1)),
                qml.probs(op=qml.PauliY(0) @ qml.PauliZ(1)),
                qml.var(op=qml.PauliZ(1)),
            )

        res1 = qml.QNode(circuit_op, dq)()
        res2 = qml.QNode(circuit_op, dev, mcm_method=mcm_method)(shots=10)
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
        res2 = qml.QNode(circuit_mcm, dev, mcm_method=mcm_method)(shots=10)
        for r1, r2 in zip(res1, res2):
            if isinstance(r1, Sequence):
                assert len(r1) == len(r2)
            assert np.all(np.isnan(r1))
            assert np.all(np.isnan(r2))


    @flaky(max_runs=5)
    @pytest.mark.parametrize("shots", [None, 4000, [3000, 1000]])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
    @pytest.mark.parametrize(
        "measure_obj",
        [qml.PauliZ(0), qml.PauliY(1), [0], [0, 1], [1, 0], "mcm", "composite_mcm", "mcm_list"],
    )
    def test_simple_dynamic_circuit(self, mcm_method, shots, measure_f, postselect, measure_obj):
        """Tests that LightningQubit handles a simple dynamic circuit with the following measurements:

            * qml.counts with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
            * qml.expval with obs (comp basis or not), MCM, f(MCM), MCM list
            * qml.probs  with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
            * qml.sample with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
            * qml.var with obs (comp basis or not), MCM, f(MCM), MCM list

        The above combinations should work for finite shots, shot vectors and post-selecting of either the 0 or 1 branch.
        """

        if measure_f in (qml.expval, qml.var) and (
            isinstance(measure_obj, list) or measure_obj == "mcm_list"
        ):
            pytest.skip("Can't use wires/mcm lists with var or expval")

        if measure_f in (qml.counts, qml.sample) and shots is None:
            pytest.skip("Skip test for None shots with counts/sample")
            
        if mcm_method == "deferred" and postselect is not None:
            pytest.skip("Skip test for postselection with deferred measurements")
            
        if mcm_method == "one-shot" and shots is None:
            # One-shot method does not support None shots
            pytest.skip("Skip test for one-shot with None shots")

        dq = qml.device("default.qubit", shots=shots)
        dev = get_device(wires=4, shots=shots)
        params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]

        def func(x, y, z):
            m0, m1 = TestExecutionMCM.obs_tape(x, y, z, postselect=postselect)
            mid_measure = (
                m0
                if measure_obj == "mcm"
                else (0.5 * m0 if measure_obj == "composite_mcm" else [m0, m1])
            )
            measurement_key = "wires" if isinstance(measure_obj, list) else "op"
            measurement_value = mid_measure if isinstance(measure_obj, str) else measure_obj
            return measure_f(**{measurement_key: measurement_value})

        results1 = qml.QNode(func, dev, mcm_method=mcm_method)(*params)
        results2 = qml.QNode(func, dq, mcm_method="deferred")(*params)

        validate_measurements(measure_f, shots, results1, results2)


    @pytest.mark.parametrize("shots", [None, 4000, [3000, 1000]])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("reset", [False, True])
    def test_multiple_measurements_and_reset(self, mcm_method, shots, postselect, reset):
        """Tests that LightningQubit handles a circuit with a single mid-circuit measurement with reset
        and a conditional gate. Multiple measurements of the mid-circuit measurement value are
        performed. This function also tests `reset` parametrizing over the parameter."""
                    
        if mcm_method == "deferred" and postselect is not None:
            pytest.skip("Skip test for postselection with deferred measurements")
            
        if mcm_method == "one-shot" and shots is None:
            # One-shot method does not support None shots
            pytest.skip("Skip test for one-shot with None shots")

        shots = shots
        dq = qml.device("default.qubit", shots=shots)
        dev = get_device(wires=3, shots=shots)
        params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]
        obs = qml.PauliY(1)

        def func(x, y, z):
            mcms = TestExecutionMCM.obs_tape(x, y, z, reset=reset, postselect=postselect)
            
            if shots is None:
                return (
                    qml.expval(op=mcms[0]),
                    qml.probs(op=obs),
                    qml.var(op=obs),
                )
            else:
                return (
                    qml.counts(op=obs),
                    qml.expval(op=mcms[0]),
                    qml.probs(op=obs),
                    qml.sample(op=mcms[0]),
                    qml.var(op=obs),
                )


        results1 = qml.QNode(func, dev, mcm_method=mcm_method)(*params)
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
    def test_composite_mcms(sefl, mcm_method, mcm_f, measure_f):
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

        results1 = qml.QNode(func, dev, mcm_method=mcm_method)(param)
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
    def test_counts_return_type(self,mcm_method, mcm_f):
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

        results1 = qml.QNode(func, dev, mcm_method=mcm_method)(param)
        results2 = qml.QNode(func, dq, mcm_method="deferred")(param)
        for r1, r2 in zip(results1.keys(), results2.keys()):
            assert r1 == r2
