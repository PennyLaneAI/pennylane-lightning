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
This module tests the eval_jaxpr method.
"""
from functools import partial

import pennylane as qml
import pytest
from conftest import LightningDevice, device_name
from pennylane.exceptions import DeviceError
from pennylane.transforms.defer_measurements import DeferMeasurementsInterpreter

jax = pytest.importorskip("jax")

if device_name == "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


def test_accept_execution_config():
    """Test that eval_jaxpr can accept an ExecutionConfig.

    At this point, it does not do anything, so we do not need to test its effect.
    """

    dev = qml.device(device_name, wires=1)

    jaxpr = jax.make_jaxpr(lambda x: x + 1)(0.1)

    execution_config = qml.devices.ExecutionConfig()

    dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.1, execution_config=execution_config)


def test_no_partitioned_shots():
    """Test that an error is raised if partitioned shots is requested."""

    with pytest.warns(
        qml.exceptions.PennyLaneDeprecationWarning,
        match="shots on device is deprecated",
    ):
        dev = qml.device(device_name, wires=1, shots=(100, 100, 100))
    jaxpr = jax.make_jaxpr(lambda x: x + 1)(0.1)

    with pytest.raises(NotImplementedError, match="does not support partitioned shots"):
        dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.0)


def test_no_wire():
    """Test that an error is raised if the number of wires is not specified."""

    dev = qml.device(device_name, wires=None)
    jaxpr = jax.make_jaxpr(lambda x: x + 1)(0.1)

    with pytest.raises(NotImplementedError, match="Wires must be specified"):
        dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.0)


@pytest.mark.parametrize("use_jit", (True, False))
@pytest.mark.parametrize("x64", (True, False))
def test_simple_execution(use_jit, x64):
    """Test the execution, jitting, and gradient of a simple quantum circuit."""
    original_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", x64)

        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        dev = qml.device(device_name, wires=1)
        jaxpr = jax.make_jaxpr(f)(0.5)

        if use_jit:
            [res] = jax.jit(partial(dev.eval_jaxpr, jaxpr.jaxpr))(jaxpr.consts, 0.5)
        else:
            [res] = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)
        assert qml.math.allclose(res, jax.numpy.cos(0.5))

        if x64:
            assert res.dtype == jax.numpy.float64
        else:
            assert res.dtype == jax.numpy.float32

    finally:
        jax.config.update("jax_enable_x64", original_x64)


def test_capture_remains_enabled_if_measurement_error():
    """Test that capture remains enabled if there is a measurement error."""

    with pytest.warns(
        qml.exceptions.PennyLaneDeprecationWarning,
        match="shots on device is deprecated",
    ):
        dev = qml.device(device_name, wires=1, shots=1)

    def g():
        return qml.state()

    jaxpr = jax.make_jaxpr(g)()

    with pytest.raises(jax.errors.JaxRuntimeError):
        dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    assert qml.capture.enabled()


def test_mcm_reset():
    """Test that mid circuit measurements can reset the state."""

    def f():
        qml.X(0)
        qml.measure(0, reset=True)
        return qml.state()

    dev = qml.device(device_name, wires=1)
    jaxpr = jax.make_jaxpr(f)()

    out = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
    assert qml.math.allclose(out, jax.numpy.array([1.0, 0.0]))  # reset into zero state.


def test_operator_arithmetic():
    """Test that lightning devices can execute operator arithmetic."""

    def f(x):
        qml.RY(1.0, 0)
        qml.adjoint(qml.RY(x, 0))
        _ = qml.SX(1) ** 2
        return qml.expval(qml.Z(0) + 2 * qml.Z(1))

    dev = qml.device(device_name, wires=2)
    jaxpr = jax.make_jaxpr(f)(0.5)
    output = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)
    expected = jax.numpy.cos(1 - 0.5) - 2 * 1
    assert qml.math.allclose(output, expected)


class TestSampling:
    """Test cases for generating samples."""

    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("x64", (True, False))
    def test_known_sampling(self, use_jit, x64):
        """Test sampling output with deterministic sampling output"""

        original_x64 = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", x64)

            def sampler():
                qml.X(0)
                return qml.sample(wires=(0, 1))

            with pytest.warns(
                qml.exceptions.PennyLaneDeprecationWarning,
                match="shots on device is deprecated",
            ):
                dev = qml.device(device_name, wires=2, shots=10)
            jaxpr = jax.make_jaxpr(sampler)()

            if use_jit:
                [results] = jax.jit(partial(dev.eval_jaxpr, jaxpr.jaxpr))(jaxpr.consts)
            else:
                [results] = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

            expected0 = jax.numpy.ones((10,))  # zero wire
            expected1 = jax.numpy.zeros((10,))  # one wire
            expected = jax.numpy.vstack([expected0, expected1]).T

            assert qml.math.allclose(results, expected)
            if x64:
                assert results.dtype == jax.numpy.int64
            else:
                assert results.dtype == jax.numpy.int32

        finally:
            jax.config.update("jax_enable_x64", original_x64)

    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("x64", (True, False))
    def test_seeded_sampling(self, use_jit, x64):
        """Test sampling output with deterministic sampling output"""

        original_x64 = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", x64)

            def sampler():
                qml.Hadamard(0)
                return qml.sample(wires=0)

            with pytest.warns(
                qml.exceptions.PennyLaneDeprecationWarning,
                match="shots on device is deprecated",
            ):

                dev1 = qml.device(device_name, wires=2, shots=10, seed=123)
                dev2 = qml.device(device_name, wires=2, shots=10, seed=123)
                dev3 = qml.device(device_name, wires=2, shots=10, seed=321)
            jaxpr = jax.make_jaxpr(sampler)()

            if use_jit:
                [results1] = jax.jit(partial(dev1.eval_jaxpr, jaxpr.jaxpr))(jaxpr.consts)
                [results2] = jax.jit(partial(dev2.eval_jaxpr, jaxpr.jaxpr))(jaxpr.consts)
                [results3] = jax.jit(partial(dev3.eval_jaxpr, jaxpr.jaxpr))(jaxpr.consts)
            else:
                [results1] = dev1.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
                [results2] = dev2.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
                [results3] = dev3.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

            assert qml.math.allclose(results1, results2)
            assert not qml.math.allclose(results1, results3)

        finally:
            jax.config.update("jax_enable_x64", original_x64)

    @pytest.mark.parametrize("mcm_value", (0, 1))
    def test_return_mcm(self, mcm_value):
        """Test that the interpreter can return the result of mid circuit measurements"""

        def f():
            if mcm_value:
                qml.X(0)
            return qml.measure(0)

        dev = qml.device(device_name, wires=1)
        jaxpr = jax.make_jaxpr(f)()
        output = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert qml.math.allclose(output, mcm_value)

    def test_classical_transformation_mcm_value(self):
        """Test that mid circuit measurements can be used in classical manipulations."""

        def f():
            qml.X(0)
            m0 = qml.measure(0)  # 1
            qml.X(0)  # reset to 0
            qml.RX(2 * m0, wires=0)
            return qml.expval(qml.Z(0))

        dev = qml.device(device_name, wires=1)
        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        expected = jax.numpy.cos(2.0)
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("mp_type", (qml.sample, qml.expval, qml.probs))
    def test_mcm_measurements_not_yet_implemented(self, mp_type):
        """Test that measurements of mcms are not yet implemented"""

        def f():
            m0 = qml.measure(0)
            if mp_type == qml.probs:
                return mp_type(op=m0)
            return mp_type(m0)

        with pytest.warns(
            qml.exceptions.PennyLaneDeprecationWarning,
            match="shots on device is deprecated",
        ):
            dev = qml.device(device_name, wires=1, shots=2)
        jaxpr = jax.make_jaxpr(f)()

        with pytest.raises(jax.errors.JaxRuntimeError):
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)


class TestQuantumHOP:
    """Tests for the quantum higher order primitives: adjoint and ctrl."""

    @pytest.mark.parametrize("lazy", [True, False])
    def test_adjoint_transform(self, lazy):
        """Test that the adjoint_transform is not yet implemented."""

        def circuit(x):

            def adjoint_fn(y):
                phi = y * jax.numpy.pi / 2
                qml.RZ(phi, 0)
                qml.RX(phi - jax.numpy.pi, 0)

            qml.adjoint(adjoint_fn, lazy=lazy)(x)
            return qml.state()

        dev = qml.device(device_name, wires=1)

        rz_phi = -1.5 * jax.numpy.pi / 2
        rx_phi = rz_phi + jax.numpy.pi
        expected_state = jax.numpy.array(
            [
                jax.numpy.cos(rx_phi / 2) * jax.numpy.exp(-rz_phi * 1j / 2),
                -1j * jax.numpy.sin(rx_phi / 2) * jax.numpy.exp(rz_phi * 1j / 2),
            ]
        )
        jaxpr = jax.make_jaxpr(circuit)(1.5)
        result = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.5)
        assert qml.math.allclose(result, expected_state)

    def test_ctrl_transform(self):
        """Test that the ctrl_transform is not yet implemented."""

        def circuit(x):
            qml.X(0)

            def ctrl_fn(y):
                phi = y * jax.numpy.pi / 2
                qml.RZ(phi, 2)
                qml.RX(phi - jax.numpy.pi, 2)

            qml.ctrl(ctrl_fn, control=[0, 1], control_values=[1, 0])(x)
            return qml.state()

        rz_phi = 1.5 * jax.numpy.pi / 2
        rx_phi = rz_phi - jax.numpy.pi
        expected_state = qml.math.zeros(8, dtype=complex)
        expected_state[4] = jax.numpy.cos(rx_phi / 2) * jax.numpy.exp(-rz_phi * 1j / 2)
        expected_state[5] = -1j * jax.numpy.sin(rx_phi / 2) * jax.numpy.exp(-rz_phi * 1j / 2)

        jaxpr = jax.make_jaxpr(circuit)(1.5)
        dev = qml.device(device_name, wires=3)
        result = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.5)
        assert qml.math.allclose(result, expected_state)

    def test_nested_ctrl_and_adjoint(self):
        """Tests nesting ctrl and adjoint."""

        def circuit(x):
            qml.X(0)

            def ctrl_fn(y):
                phi = y * jax.numpy.pi / 2
                qml.RZ(phi, 2)
                qml.RX(phi - jax.numpy.pi, 2)

            qml.adjoint(qml.ctrl(ctrl_fn, control=[0, 1], control_values=[1, 0]))(x)
            return qml.state()

        rz_phi = 1.5 * jax.numpy.pi / 2
        rx_phi = rz_phi - jax.numpy.pi
        expected_state = qml.math.zeros(8, dtype=complex)
        expected_state[4] = jax.numpy.cos(rx_phi / 2) * jax.numpy.exp(rz_phi * 1j / 2)
        expected_state[5] = 1j * jax.numpy.sin(rx_phi / 2) * jax.numpy.exp(-rz_phi * 1j / 2)

        jaxpr = jax.make_jaxpr(circuit)(1.5)
        dev = qml.device(device_name, wires=3)
        result = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.5)
        assert qml.math.allclose(result, expected_state)


class TestClassicalComponents:
    """Test execution of classical components."""

    def test_classical_operations_in_circuit(self):
        """Test that we can have classical operations in the circuit."""

        def f(x, y, w):
            qml.RX(2 * x + y, wires=w - 1)
            return qml.expval(qml.Z(0))

        dev = qml.device(device_name, wires=1)

        x = jax.numpy.array(0.5)
        y = jax.numpy.array(1.2)
        w = jax.numpy.array(1)

        jaxpr = jax.make_jaxpr(f)(x, y, w)
        output = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, y, w)
        expected = jax.numpy.cos(2 * x + y)
        assert qml.math.allclose(output, expected)

    def test_for_loop(self):
        """Test that the for loop can be executed."""

        def f(y):
            @qml.for_loop(4)
            def g(i, x):
                qml.RX(x, i)
                return x + 0.1

            g(y)
            return [qml.expval(qml.Z(i)) for i in range(4)]

        x = 1.0
        jaxpr = jax.make_jaxpr(f)(x)
        dev = qml.device(device_name, wires=4)

        output = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert len(output) == 4
        assert qml.math.allclose(output[0], jax.numpy.cos(1.0))
        assert qml.math.allclose(output[1], jax.numpy.cos(1.1))
        assert qml.math.allclose(output[2], jax.numpy.cos(1.2))
        assert qml.math.allclose(output[3], jax.numpy.cos(1.3))

    def test_for_loop_consts(self):
        """Test that the for_loop can be executed properly when it has closure variables."""

        def g(x):
            @qml.for_loop(2)
            def f(i):
                qml.RX(x, i)  # x is closure variable

            f()
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        dev = qml.device(device_name, wires=2)
        x = jax.numpy.array(-0.654)
        jaxpr = jax.make_jaxpr(g)(x)

        res1, res2 = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        expected = jax.numpy.cos(x)
        assert qml.math.allclose(res1, expected)
        assert qml.math.allclose(res2, expected)

    def test_while_loop(self):
        """Test that the while loop can be executed."""

        def f():
            def cond_fn(i):
                return i < 4

            @qml.while_loop(cond_fn)
            def g(i):
                qml.X(i)
                return i + 1

            g(0)
            return [qml.expval(qml.Z(i)) for i in range(4)]

        dev = qml.device(device_name, wires=4)
        jaxpr = jax.make_jaxpr(f)()
        output = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert qml.math.allclose(output, [-1, -1, -1, -1])

    def test_while_loop_with_consts(self):
        """Test that both the cond_fn and body_fn can contain constants with the while loop."""

        def g(x, target):
            def cond_fn(i):
                return i < target

            @qml.while_loop(cond_fn)
            def f(i):
                qml.RX(x, 0)
                return i + 1

            f(0)
            return qml.expval(qml.Z(0))

        x, y = jax.numpy.array(1.2), jax.numpy.array(2)
        jaxpr = jax.make_jaxpr(g)(x, y)
        dev = qml.device(device_name, wires=2)

        output = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, y)

        assert qml.math.allclose(output, jax.numpy.cos(y * x))

    def test_cond_boolean(self):
        """Test that cond can be used with normal classical values."""

        def true_fn(x):
            qml.RX(x, 0)
            return x + 1

        def false_fn(x):
            return 2 * x

        def f(x, val):
            out = qml.cond(val, true_fn, false_fn)(x)
            return qml.probs(wires=0), out

        x = 0.5
        jaxpr = jax.make_jaxpr(f)(x, True)
        dev = qml.device(device_name, wires=1)
        output_true = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, True)

        expected0 = [jax.numpy.cos(0.5 / 2) ** 2, jax.numpy.sin(0.5 / 2) ** 2]
        assert qml.math.allclose(output_true[0], expected0)
        assert qml.math.allclose(output_true[1], 1.5)  # 0.5 + 1

        output_false = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, False)
        assert qml.math.allclose(output_false[0], [1.0, 0.0])
        assert qml.math.allclose(output_false[1], 1.0)  # 2 * 0.5

    def test_cond_mcm(self):
        """Test that cond can be used with the output of mcms."""

        def true_fn(y):
            qml.RX(y, 0)

        # pylint: disable=unused-argument
        def false_fn(y):
            qml.X(0)

        def g(x):
            qml.X(0)
            m0 = qml.measure(0)
            qml.X(0)
            qml.cond(m0, true_fn, false_fn)(x)
            return qml.probs(wires=0)

        x = 0.5
        jaxpr = jax.make_jaxpr(g)(x)
        dev = qml.device(device_name, wires=1)

        output = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        expected = [jax.numpy.cos(x / 2) ** 2, jax.numpy.sin(x / 2) ** 2]
        assert qml.math.allclose(output, expected)

    def test_cond_false_no_false_fn(self):
        """Test nothing is returned when the false_fn is not provided but the condition is false."""

        def true_fn(w):
            qml.X(w)

        def g(condition):
            qml.cond(condition, true_fn)(0)
            return qml.expval(qml.Z(0))

        dev = qml.device(device_name, wires=1)
        jaxpr = jax.make_jaxpr(g)(True)

        out = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, False)
        assert qml.math.allclose(out, 1.0)

    def test_condition_with_consts(self):
        """Test that each branch in a condition can contain consts."""

        def circuit(x, y, z, condition0, condition1):
            def true_fn():
                qml.RX(x, 0)

            def false_fn():
                qml.RX(y, 0)

            def elif_fn():
                qml.RX(z, 0)

            qml.cond(condition0, true_fn, false_fn=false_fn, elifs=((condition1, elif_fn),))()

            return qml.expval(qml.Z(0))

        x = jax.numpy.array(0.3)
        y = jax.numpy.array(0.6)
        z = jax.numpy.array(1.2)

        jaxpr = jax.make_jaxpr(circuit)(x, y, z, True, True)
        dev = qml.device(device_name, wires=1)

        res0 = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, y, z, True, False)
        assert qml.math.allclose(res0, jax.numpy.cos(x))

        res1 = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, y, z, False, True)
        assert qml.math.allclose(res1, jax.numpy.cos(z))  # elif branch = z

        res2 = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, y, z, False, False)
        assert qml.math.allclose(res2, jax.numpy.cos(y))  # false fn = y

    def test_nested_higher_order_primitives(self):
        """Test a conditional inside a for loop."""

        def true_fn(x):
            qml.RX(x, 0)

        def false_fn(x):
            qml.RX(0.1, 0)

        def f(x, n):
            @qml.for_loop(n)
            def loop(i):
                qml.cond(i % 2 == 0, true_fn, false_fn=false_fn)(i * x)

            loop()
            return qml.expval(qml.Z(0))

        x = jax.numpy.array(1.0)
        n = jax.numpy.array(3)
        jaxpr = jax.make_jaxpr(f)(x, n)
        dev = qml.device(device_name, wires=1)

        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, n)
        expected = jax.numpy.cos(0 + 0.1 + 2 * x)
        assert qml.math.allclose(res, expected)


@pytest.mark.parametrize("use_jit", (True, False))
def test_vmap_integration(use_jit):
    """Test that the lightning devices can execute circuits with vmap applied."""

    @qml.qnode(qml.device(device_name, wires=1))
    def circuit(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    x = jax.numpy.array([1.0, 2.0, 3.0])
    f = jax.jit(jax.vmap(circuit)) if use_jit else jax.vmap(circuit)
    results = f(x)
    assert qml.math.allclose(results, jax.numpy.cos(x))


@pytest.mark.parametrize("in_axis", (0, 1, 2))
@pytest.mark.parametrize("out_axis", (0, 1))
def test_vmap_in_axes(in_axis, out_axis):
    """Test that vmap works with specified in_axes and out_axes."""

    # Get the current x64 setting to determine which dtype to use
    x64_enabled = jax.config.jax_enable_x64
    c_dtype = jax.numpy.complex128 if x64_enabled else jax.numpy.complex64

    @qml.qnode(qml.device(device_name, wires=1, c_dtype=c_dtype))
    def circuit(mat):
        qml.QubitUnitary(mat, 0)
        return qml.expval(qml.Z(0)), qml.state()

    mats = jax.numpy.stack(
        [qml.X.compute_matrix(), qml.Y.compute_matrix(), qml.Z.compute_matrix()],
        axis=in_axis,
        dtype=c_dtype,
    )
    expval, state = jax.vmap(circuit, in_axes=in_axis, out_axes=(0, out_axis))(mats)

    assert expval.shape == (3,)
    assert qml.math.allclose(expval, jax.numpy.array([-1, -1, 1]))  # flip, flip, no flip
    assert state.shape == (3, 2) if out_axis == 0 else (2, 3)


class TestDeferMeasurements:
    """Tests that Lightning devices can execute circuits transformed by defer_measurements."""

    def test_single_mcm(self):
        """Test that applying a single MCM works."""

        dev = qml.device(device_name, wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f():
            qml.Hadamard(0)
            qml.measure(0)
            qml.Hadamard(0)
            return qml.expval(qml.PauliX(0))

        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert qml.math.allclose(res, 0)

    def test_qubit_reset(self):
        """Test that resetting a qubit works as expected."""

        dev = qml.device(device_name, wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f():
            qml.PauliX(0)
            qml.measure(0, reset=True)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert qml.math.allclose(res, 1)

    @pytest.mark.parametrize("postselect", [0, 1])
    def test_postselection_error(self, postselect):
        """Test that a runtime error is raised if postselection is used with defer_measurements."""

        dev = qml.device(device_name, wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f():
            qml.PauliX(0)
            qml.measure(0, postselect=postselect)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        with pytest.raises(jax.errors.JaxRuntimeError):
            with pytest.raises(DeviceError, match="Lightning devices do not support postselection"):
                dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    def test_mcms_as_gate_parameters(self):
        """Test that using MCMs as gate parameters works as expected."""

        dev = qml.device(device_name, wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f():
            qml.Hadamard(0)
            m = qml.measure(0)
            qml.RX(m * jax.numpy.pi, 0)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        # If 0 measured, RX does nothing, so state is |0>. If 1 measured, RX(pi)
        # makes state |1> -> |0>, so <Z> will always be 1
        assert qml.math.allclose(res, 1)

    def test_cond(self):
        """Test that using qml.cond with MCM predicates works as expected."""

        dev = qml.device(device_name, wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f(x):
            qml.Hadamard(0)
            qml.Hadamard(1)
            m0 = qml.measure(0)
            m1 = qml.measure(1)

            @qml.cond(m0 == 0)
            def cond_fn(y):
                qml.RY(y, 0)

            @cond_fn.else_if(m1 == 0)
            def _(y):
                qml.RY(2 * y, 0)

            @cond_fn.otherwise
            def _(y):
                qml.RY(3 * y, 0)

            cond_fn(x)

            return qml.expval(qml.PauliZ(0))

        phi = jax.numpy.pi / 3
        jaxpr = jax.make_jaxpr(f)(phi)
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, phi)
        expected = 0.5 * (jax.numpy.cos(phi) + jax.numpy.sin(phi) ** 2)
        assert qml.math.allclose(res, expected)

    def test_cond_non_mcm(self):
        """Test that using qml.cond with non-MCM predicates works as expected."""

        dev = qml.device(device_name, wires=5)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f(x):
            qml.Hadamard(0)
            m0 = qml.measure(0)

            @qml.cond(x > 2.5)
            def cond_fn():
                qml.RX(m0 * jax.numpy.pi, 0)
                # Final state |0>

            @cond_fn.else_if(x > 1.5)
            def _():
                qml.PauliZ(0)
                # Equal prob of |0> and |1>

            @cond_fn.otherwise
            def _():
                qml.Hadamard(0)
                m1 = qml.measure(0)
                qml.RX(m1 * jax.numpy.pi, 0)
                qml.X(0)
                # Final state |1>

            cond_fn()

            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        arg_true = 3.0
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg_true)
        assert qml.math.allclose(res, 1)  # Final state |0>; <Z> = 1

        arg_elif = 2.0
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg_elif)
        assert qml.math.allclose(res, 0)  # Equal prob of |0>, |1>; <Z> = 1

        arg_true = 1.0
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg_true)
        assert qml.math.allclose(res, -1)  # Final state |1>, <Z> = -1

    @pytest.mark.parametrize(
        "mp_fn",
        [
            qml.expval,
            qml.var,
            qml.probs,
        ],
    )
    def test_mcm_statistics(self, mp_fn):
        """Test that collecting statistics on MCMs is handled correctly."""

        dev = qml.device(device_name, wires=5)

        def processing_fn(m1, m2):
            return 2.5 * m1 - m2

        def f():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.Hadamard(0)
            m1 = qml.measure(0)
            qml.Hadamard(0)
            m2 = qml.measure(0)

            outs = (mp_fn(op=m0),)
            if mp_fn is qml.probs:
                outs += (mp_fn(op=[m0, m1, m2]),)
            else:
                outs += (mp_fn(op=processing_fn(m1, m2)),)

            return outs

        transformed_f = DeferMeasurementsInterpreter(num_wires=5)(f)
        qnode_f = qml.QNode(f, dev, mcm_method="deferred")

        jaxpr = jax.make_jaxpr(transformed_f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        with qml.capture.pause():
            expected = qnode_f()

        for r, e in zip(res, expected, strict=True):
            assert qml.math.allclose(r, e)

    def test_shots(self):
        """Tests that defer measurements executes correctly with shots."""
        with pytest.warns(
            qml.exceptions.PennyLaneDeprecationWarning,
            match="shots on device is deprecated",
        ):
            dev = qml.device(device_name, wires=5, shots=100)

        @DeferMeasurementsInterpreter(num_wires=5)
        def f(x):

            @qml.cond(x)
            def x_cond():
                qml.PauliX(0)

            x_cond()
            m = qml.measure(0)

            @qml.cond(m == 0)
            def cond_fn():
                # State after this cond will be |1>
                qml.PauliX(0)

            @cond_fn.otherwise
            def _():
                # State after this will be (|0>-|1>)/sqrt(2)
                qml.Hadamard(0)

            cond_fn()
            return qml.sample(wires=0)

        jaxpr = jax.make_jaxpr(f)(True)
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, False)
        assert qml.math.allclose(res, 1)
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, True)
        # Assert that the result is a mix of 0s and 1s
        assert not qml.math.allclose(res, 0) and not qml.math.allclose(res, 1)


@pytest.mark.parametrize("shots", [None, 100, 1000])
def test_eval_jaxpr_with_shots_parameter(shots):
    """Test that eval_jaxpr accepts and respects the shots parameter."""

    def f(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0)) if shots is None else qml.sample(qml.Z(0))

    dev = qml.device(device_name, wires=1)
    jaxpr = jax.make_jaxpr(f)(0.5)

    result = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5, shots=shots)

    if shots is None:
        # For analytic mode, result should be a single expectation value
        assert qml.math.allclose(result[0], jax.numpy.cos(0.5))
    else:
        # For finite shots, result should be a sample array of the specified length
        assert len(result[0]) == shots
        assert all(sample in [-1, 1] for sample in result[0])


def test_eval_jaxpr_shots_parameter_overrides_device_shots():
    """Test that shots parameter overrides device-level shots setting."""

    # Create device with device-level shots
    with pytest.warns(
        qml.exceptions.PennyLaneDeprecationWarning,
        match="shots on device is deprecated",
    ):
        dev = qml.device(device_name, wires=1, shots=500)

    def f():
        qml.RX(0.5, 0)
        return qml.sample(qml.Z(0))

    jaxpr = jax.make_jaxpr(f)()

    # Test with different shots parameter - should override device shots
    result_100 = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, shots=100)
    assert len(result_100[0]) == 100

    result_1000 = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, shots=1000)
    assert len(result_1000[0]) == 1000

    # Test with None shots - should work in analytic mode
    def f_expval():
        qml.RX(0.5, 0)
        return qml.expval(qml.Z(0))

    jaxpr_expval = jax.make_jaxpr(f_expval)()
    result_analytic = dev.eval_jaxpr(jaxpr_expval.jaxpr, jaxpr_expval.consts, shots=None)
    assert qml.math.allclose(result_analytic[0], jax.numpy.cos(0.5))


@pytest.mark.parametrize("shots", [50, 200, 1000])
def test_eval_jaxpr_multiple_measurements_with_shots(shots):
    """Test eval_jaxpr with multiple measurements and shots parameter."""

    def f():
        qml.RX(0.5, 0)
        qml.RY(0.3, 1)
        return [qml.sample(qml.Z(0)), qml.sample(qml.Z(1))]

    dev = qml.device(device_name, wires=2)
    jaxpr = jax.make_jaxpr(f)()

    result = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, shots=shots)

    # Should have two sample arrays, each with the correct number of shots
    assert len(result) == 2
    assert len(result[0]) == shots
    assert len(result[1]) == shots

    # All samples should be valid measurement outcomes
    assert all(sample in [-1, 1] for sample in result[0])
    assert all(sample in [-1, 1] for sample in result[1])


def test_eval_jaxpr_shots_with_probs():
    """Test eval_jaxpr with shots parameter for probability measurements."""

    def f():
        qml.RX(0.5, 0)
        return qml.probs(wires=0)

    dev = qml.device(device_name, wires=1)
    jaxpr = jax.make_jaxpr(f)()

    # Test with finite shots
    result_shots = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, shots=1000)
    probs_shots = result_shots[0]

    # Should return probabilities that sum to 1
    assert qml.math.allclose(jax.numpy.sum(probs_shots), 1.0, atol=1e-2)
    assert len(probs_shots) == 2  # Two outcomes for single qubit

    # Test with analytic mode
    result_analytic = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, shots=None)
    probs_analytic = result_analytic[0]

    # Analytic and finite-shot results should be close
    assert qml.math.allclose(probs_shots, probs_analytic, atol=0.1)
