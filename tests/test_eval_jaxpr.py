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

jax = pytest.importorskip("jax")
jaxlib = pytest.importorskip("jaxlib")

if device_name == "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


def test_no_partitioned_shots():
    """Test that an error is raised if partitioned shots is requested."""

    dev = qml.device(device_name, wires=1, shots=(100, 100, 100))
    jaxpr = jax.make_jaxpr(lambda x: x + 1)(0.1)

    with pytest.raises(NotImplementedError, match="does not support partitioned shots"):
        dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.0)


@pytest.mark.parametrize("use_jit", (True, False))
def test_simple_execution(use_jit):
    """Test the execution, jitting, and gradient of a simple quantum circuit."""

    def f(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    dev = qml.device(device_name, wires=1)
    jaxpr = jax.make_jaxpr(f)(0.5)

    if use_jit:
        res = jax.jit(partial(dev.eval_jaxpr, jaxpr.jaxpr))(jaxpr.consts, 0.5)
    else:
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)
    assert qml.math.allclose(res, jax.numpy.cos(0.5))


def test_capture_remains_enabled_if_measurement_error():
    """Test that capture remains enabled if there is a measurement error."""

    dev = qml.device(device_name, wires=1, shots=1)

    def g():
        return qml.state()

    jaxpr = jax.make_jaxpr(g)()

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
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
    def test_known_sampling(self, use_jit):
        """Test sampling output with deterministic sampling output"""

        def sampler():
            qml.X(0)
            return qml.sample(wires=(0, 1))

        dev = qml.device(device_name, wires=2, shots=10)
        jaxpr = jax.make_jaxpr(sampler)()

        if use_jit:
            results = jax.jit(partial(dev.eval_jaxpr, jaxpr.jaxpr))(jaxpr.consts)
        else:
            results = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        expected0 = jax.numpy.ones((10,))  # zero wire
        expected1 = jax.numpy.zeros((10,))  # one wire
        expected = jax.numpy.vstack([expected0, expected1]).T

        assert qml.math.allclose(results, expected)

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

        dev = qml.device(device_name, wires=1, shots=2)
        jaxpr = jax.make_jaxpr(f)()

        with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)


class TestQuantumHOP:
    """Tests for the quantum higher order primitives: adjoint and ctrl."""

    def test_adjoint_transform(self):
        """Test that the adjoint_transform is not yet implemented."""

        def circuit(x):
            qml.adjoint(qml.RX)(x, 0)
            return 1

        dev = qml.device(device_name, wires=1)
        jaxpr = jax.make_jaxpr(circuit)(0.5)

        with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)

    def test_ctrl_transform(self):
        """Test that the ctrl_transform is not yet implemented."""

        def circuit():
            qml.ctrl(qml.X, control=1)(0)
            return 2

        dev = qml.device(device_name, wires=2)
        jaxpr = jax.make_jaxpr(circuit)()

        with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)


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


def test_vmap_integration():
    """Test that the lightning devices can execute circuits with vmap applied."""

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x):
        qml.RX(x, 0)
        return qml.expval(qml.Z(0))

    x = jax.numpy.array([1.0, 2.0, 3.0])
    results = jax.vmap(circuit)(x)
    assert qml.math.allclose(results, jax.numpy.cos(x))
