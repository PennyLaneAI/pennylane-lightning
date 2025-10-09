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

from functools import partial
from typing import Sequence

import numpy as np
import pennylane as qml
import pytest
from conftest import (
    LightningDevice,
    device_name,
    validate_counts,
    validate_measurements,
    validate_others,
    validate_samples,
)
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
        "deferred",
        "one-shot",
        "tree-traversal",
    ],
)
def mcm_method(request):
    """Fixture to set the MCM method for the tests."""
    return request.param


class TestUnsupportedConfigurationsMCM:
    """Test unsupported configurations for different mid-circuit measurement methods."""

    def generate_mcm_circuit(
        self,
        device_kwargs={},
        qnode_kwargs={},
        mcm_kwargs={},
        measurement=qml.expval,
        obs=qml.PauliZ(0),
    ):
        """Generate a circuit with a mid-circuit measurement."""
        dev = qml.device(device_name, **device_kwargs)

        @qml.qnode(dev, **qnode_kwargs)
        def func(y):
            qml.RX(y, wires=0)
            m0 = qml.measure(0, **mcm_kwargs)
            qml.cond(m0, qml.RY)(y, wires=1)
            return measurement(obs)

        return func

    def test_unsupported_method(self):
        method = "roller-coaster"
        circuit = self.generate_mcm_circuit(
            device_kwargs={"wires": 1, "shots": 100},
            qnode_kwargs={"mcm_method": method},
            mcm_kwargs={"postselect": None, "reset": False},
            measurement=qml.expval,
            obs=qml.PauliZ(0),
        )
        with pytest.raises(
            DeviceError, match=f"mcm_method='{method}' is not supported with {device_name}"
        ):
            circuit(1.33)

    def test_unsupported_measurement(self, mcm_method):
        """Test unsupported ``qml.classical_shadow`` measurement on Lightning devices."""

        circuit = self.generate_mcm_circuit(
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
        """Test unsupported configuration for deferred mcm method."""

        circuit = self.generate_mcm_circuit(
            device_kwargs={"wires": 1, "shots": None},
            qnode_kwargs={"mcm_method": "deferred"},
            mcm_kwargs={"postselect": None, "reset": False},
            measurement=qml.expval,
            obs=qml.PauliZ(0),
        )

        with pytest.raises(
            qml.exceptions.WireError,
            match=f"on {device_name} as they contain wires not found on the device: {{1}}",
        ):
            circuit(1.33)

        for postsel in ["hw-like", "fill-shots"]:
            circuit = self.generate_mcm_circuit(
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
        """Test unsupported configuration for one-shot mcm method."""

        circuit = self.generate_mcm_circuit(
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
        """Test unsupported configuration for tree-traversal mcm method."""

        for measurement in [qml.counts, qml.sample]:
            circuit = self.generate_mcm_circuit(
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

    @pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
    def test_unsupported_postselect_mode(self, mcm_method):
        """Test raising an error for unsupported postselection in Lightning"""

        circuit = self.generate_mcm_circuit(
            device_kwargs={"wires": 1, "shots": 100},
            qnode_kwargs={"mcm_method": mcm_method, "postselect_mode": "fill-shots"},
            mcm_kwargs={"postselect": 1, "reset": False},
            measurement=qml.expval,
            obs=qml.Z(0),
        )
        with pytest.raises(
            DeviceError, match="Using postselect_mode='fill-shots' is not supported."
        ):
            circuit(1.23)

    def test_impossible_state_for_TT(self):
        """Test impossible state with mid-circuit measurement for tree-traversal method."""

        def circuit():
            qml.X(0)
            qml.measure(1, postselect=1)
            return qml.expval(qml.PauliZ(0))

        dev_dq = qml.device("default.qubit", wires=2)
        dev_lq = qml.device(device_name, wires=2)

        mcm_method = "tree-traversal"
        qnode_dq = qml.QNode(circuit, dev_dq, mcm_method=mcm_method)
        qnode_lq = qml.QNode(circuit, dev_lq, mcm_method=mcm_method)

        with pytest.raises(ZeroDivisionError, match="division by zero"):
            qnode_dq()

        with pytest.raises(ZeroDivisionError, match="division by zero"):
            qnode_lq()


class TestSupportedConfigurationsMCM:
    def generate_mcm_circuit(
        self,
        device_kwargs={},
        qnode_kwargs={},
        mcm_kwargs={},
        measurement=qml.expval,
        obs=qml.PauliZ(0),
    ):
        """Generate a circuit with a mid-circuit measurement."""
        dev = qml.device(device_name, **device_kwargs)

        @qml.qnode(dev, **qnode_kwargs)
        def func(y):
            qml.RX(y, wires=0)
            m0 = qml.measure(0, **mcm_kwargs)
            qml.cond(m0, qml.RY)(y, wires=1)
            return measurement(obs)

        return func

    @pytest.mark.parametrize("shots", [None, 10])
    def test_qnode_mcm_method(self, mocker, mcm_method, shots):
        """Test that user specified qnode arg for mid-circuit measurements transform are used correctly"""

        if mcm_method == "one-shot" and shots is None:
            pytest.skip("Skip test for one-shot with None shots")

        spy_deffered = mocker.spy(qml.defer_measurements, "_transform")
        spy_one_shot = mocker.spy(qml.dynamic_one_shot, "_transform")
        spy_tree_traversal = mocker.patch(
            "pennylane_lightning.lightning_base.lightning_base.mcm_tree_traversal"
        )

        circuit = self.generate_mcm_circuit(
            device_kwargs={"wires": 3, "shots": shots},
            qnode_kwargs={"mcm_method": mcm_method},
            mcm_kwargs={},
        )

        _ = circuit(np.pi / 8)

        if mcm_method == "deferred":
            spy_deffered.assert_called_once()
            spy_one_shot.assert_not_called()
            spy_tree_traversal.assert_not_called()
        elif mcm_method == "one-shot":
            spy_one_shot.assert_called_once()
            spy_deffered.assert_not_called()
            spy_tree_traversal.assert_not_called()
        elif mcm_method == "tree-traversal":
            spy_tree_traversal.assert_called_once()
            spy_deffered.assert_not_called()
            spy_one_shot.assert_not_called()

    @pytest.mark.parametrize("shots", [None, 10])
    def test_qnode_default_mcm_method_device(self, shots, mocker):
        """Test the default mcm method is used for analytical simulation"""
        spy_deferred = mocker.spy(qml.defer_measurements, "_transform")
        spy_dynamic_one_shot = mocker.spy(qml.dynamic_one_shot, "_transform")
        spy_tree_traversal = mocker.patch(
            "pennylane_lightning.lightning_base.lightning_base.mcm_tree_traversal"
        )

        circuit = self.generate_mcm_circuit(
            device_kwargs={"wires": 3, "shots": shots},
            qnode_kwargs={"mcm_method": "device"},
            mcm_kwargs={},
        )

        _ = circuit(np.pi / 8)

        spy_deferred.assert_not_called()
        spy_dynamic_one_shot.assert_not_called()
        spy_tree_traversal.assert_called_once()

    def test_qnode_default_mcm_method_analytical(self, mocker):
        """Test the default mcm method is used for analytical simulation"""
        spy_deferred = mocker.spy(qml.defer_measurements, "_transform")
        spy_dynamic_one_shot = mocker.spy(qml.dynamic_one_shot, "_transform")
        spy_tree_traversal = mocker.patch(
            "pennylane_lightning.lightning_base.lightning_base.mcm_tree_traversal"
        )

        shots = None

        circuit = self.generate_mcm_circuit(
            device_kwargs={"wires": 3, "shots": shots},
            qnode_kwargs={},
            mcm_kwargs={},
        )

        _ = circuit(np.pi / 8)

        spy_deferred.assert_called_once()
        spy_dynamic_one_shot.assert_not_called()
        spy_tree_traversal.assert_not_called()

    def test_qnode_default_mcm_method_finite_shots(self, mocker):
        """Test the default mcm method is used for finite shots"""

        spy_deferred = mocker.spy(qml.defer_measurements, "_transform")
        spy_dynamic_one_shot = mocker.spy(qml.dynamic_one_shot, "_transform")
        spy_tree_traversal = mocker.patch(
            "pennylane_lightning.lightning_base.lightning_base.mcm_tree_traversal"
        )

        shots = 33

        circuit = self.generate_mcm_circuit(
            device_kwargs={"wires": 3, "shots": shots},
            qnode_kwargs={},
            mcm_kwargs={},
        )

        _ = circuit(np.pi / 8)

        spy_deferred.assert_not_called()
        spy_dynamic_one_shot.assert_called_once()
        spy_tree_traversal.assert_not_called()

    def test_qnode_postselect_mode(self, mcm_method):
        """Test that user specified qnode arg for discarding invalid shots is used correctly"""

        if mcm_method == "deferred":
            pytest.skip(reason="Deferred does not support postselection")

        shots = 100
        device = qml.device(device_name, wires=3)
        postselect = 1

        @partial(qml.set_shots, shots=shots)
        @qml.qnode(device, postselect_mode="hw-like", mcm_method=mcm_method)
        def f(x):
            qml.RX(x, 0)
            _ = qml.measure(0, postselect=postselect)
            qml.CNOT([0, 1])
            return qml.sample(wires=[1])

        # Using small-ish rotation angle ensures the number of valid shots will be less than the
        # original number of shots. This helps avoid stochastic failures for the assertion below
        res = f(np.pi / 2)

        assert len(res) < shots
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

    def test_all_invalid_shots_circuit(self, mcm_method):
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
        res2 = qml.set_shots(qml.QNode(circuit_op, dev, mcm_method=mcm_method), shots=10)()
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
        res2 = qml.set_shots(qml.QNode(circuit_mcm, dev, mcm_method=mcm_method), shots=10)()
        for r1, r2 in zip(res1, res2):
            if isinstance(r1, Sequence):
                assert len(r1) == len(r2)
            assert np.all(np.isnan(r1))
            assert np.all(np.isnan(r2))

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize("shots", [None, 5000, [4000, 4001]])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
    @pytest.mark.parametrize(
        "measure_obj",
        [qml.PauliZ(0), qml.PauliY(1), [0], [0, 1], [1, 0], "mcm", "composite_mcm", "mcm_list"],
    )
    def test_simple_dynamic_circuit(
        self, mcm_method, shots, measure_f, postselect, measure_obj, seed
    ):
        """Tests that LightningDevices handles a simple dynamic circuit with the following measurements:

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
            pytest.skip("Skip test for one-shot with None shots")

        wires = 4 if mcm_method == "deferred" else 2
        dq = qml.device("default.qubit", seed=seed)
        dev = get_device(wires=wires, seed=seed)
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

        results1 = qml.set_shots(qml.QNode(func, dev, mcm_method=mcm_method), shots=shots)(*params)
        results2 = qml.set_shots(qml.QNode(func, dq, mcm_method="deferred"), shots=shots)(*params)

        validate_measurements(measure_f, shots, results1, results2, atol=0.04)

    @pytest.mark.local_salt(42)
    @pytest.mark.parametrize("shots", [None, 4000])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("reset", [False, True])
    def test_multiple_measurements_and_reset(self, mcm_method, shots, postselect, reset, seed):
        """Tests that LightningDevices handles a circuit with a single mid-circuit measurement with reset
        and a conditional gate. Multiple measurements of the mid-circuit measurement value are
        performed. This function also tests `reset` parametrizing over the parameter."""

        if mcm_method == "deferred" and postselect is not None:
            pytest.skip("Skip test for postselection with deferred measurements")

        if mcm_method == "one-shot" and shots is None:
            # One-shot method does not support None shots
            pytest.skip("Skip test for one-shot with None shots")

        shots = shots
        wires = 4 if mcm_method == "deferred" else 2

        dq = qml.device("default.qubit", seed=seed)
        dev = get_device(wires=wires, seed=seed)

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

        results1 = qml.set_shots(qml.QNode(func, dev, mcm_method=mcm_method), shots=shots)(*params)
        results2 = qml.set_shots(qml.QNode(func, dq, mcm_method="deferred"), shots=shots)(*params)

        measurements = (
            [qml.counts, qml.expval, qml.probs, qml.sample, qml.var]
            if shots is not None
            else [qml.expval, qml.probs, qml.var]
        )

        for measure_f, r1, r2 in zip(measurements, results1, results2):
            validate_measurements(measure_f, shots, r1, r2)

    @pytest.mark.local_salt(43)
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
    def test_composite_mcms(sefl, mcm_method, mcm_f, measure_f, seed):
        """Tests that Lightning Devices handles a circuit with a composite mid-circuit measurement and a
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
        wires = 2 if mcm_method != "deferred" else 3

        dq = qml.device("default.qubit", seed=seed)
        dev = get_device(wires=wires, seed=seed)
        param = np.pi / 3

        @partial(qml.set_shots, shots=shots)
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
    def test_counts_return_type(self, mcm_method, mcm_f):
        """Tests that LightningDevices returns the same keys for ``qml.counts`` measurements with ``dynamic_one_shot`` and ``defer_measurements``."""
        shots = 500

        wires = 3 if mcm_method == "deferred" else 2

        dq = qml.device("default.qubit")
        dev = get_device(wires=wires)
        param = np.pi / 3

        @partial(qml.set_shots, shots=shots)
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

    @pytest.mark.parametrize("shots", [40, [40, 40]])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
    @pytest.mark.parametrize(
        "measure_obj",
        [qml.PauliZ(0), qml.PauliY(1), [0], [0, 1], [1, 0], "mcm", "composite_mcm", "mcm_list"],
    )
    def test_seeded_mcm(self, mcm_method, shots, measure_f, postselect, measure_obj):
        """Tests that seeded MCM measurements return the same results for two devices with the same seed."""

        if measure_f in (qml.expval, qml.var) and (
            isinstance(measure_obj, list) or measure_obj == "mcm_list"
        ):
            pytest.skip("Can't use wires/mcm lists with var or expval")

        if mcm_method == "deferred" and postselect is not None:
            pytest.skip("Skip test for postselection with deferred measurements")

        wires = 4 if mcm_method == "deferred" else 2
        dev_1 = get_device(wires=wires, seed=123)
        dev_2 = get_device(wires=wires, seed=123)
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

        results1 = qml.set_shots(qml.QNode(func, dev_1, mcm_method=mcm_method), shots=shots)(
            *params
        )
        results2 = qml.set_shots(qml.QNode(func, dev_2, mcm_method=mcm_method), shots=shots)(
            *params
        )

        if measure_f is qml.counts:
            validate_counts(shots, results1, results2, rtol=0, atol=0)
        elif measure_f is qml.sample:
            validate_samples(shots, results1, results2, rtol=0, atol=0)
        else:
            validate_others(shots, results1, results2, rtol=0, atol=0)
