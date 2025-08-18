# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This module tests the ``jaxpr_jvp`` method.
"""
from functools import partial

import pennylane as qml
import pytest
from conftest import LightningDevice, device_name
from pennylane.devices import ExecutionConfig

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

if device_name == "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    """Fixture to enable and disable the plxpr capture."""
    qml.capture.enable()
    yield
    qml.capture.disable()


class TestErrors:
    """Test explicit errors for various unsupported cases."""

    def test_error_not_adjoint_method(self):
        """Test that an error is raised if the gradient method is not 'adjoint'."""

        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (0.5,)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        execution_config = ExecutionConfig(gradient_method="backprop")

        with pytest.raises(
            NotImplementedError, match="LightningQubit does not support gradient_method"
        ):
            qml.device(device_name, wires=1).jaxpr_jvp(jaxpr.jaxpr, args, (0.5,), execution_config)

    def test_mismatch_args_tangents(self):
        """Test that the jaxpr_jvp method raises an error if the number of arguments and tangents do not match."""

        @qml.qnode(device=qml.device(device_name, wires=1))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (0.5,)
        tangents = (0.5, 0.5)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        with pytest.raises(ValueError, match="The number of arguments and tangents must match"):
            qml.device(device_name, wires=1).jaxpr_jvp(jaxpr.jaxpr, args, tangents)

    def test_only_measurements(self):
        """Test that the jaxpr_jvp method raises an error if the circuit does not return a measurement."""

        def circuit(x):
            qml.RX(x, 0)
            return x

        args = (0.5,)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        with pytest.raises(NotImplementedError, match="The circuit should return a measurement"):
            qml.device(device_name, wires=1).jaxpr_jvp(jaxpr.jaxpr, args, (0.5,))

    def test_wrong_order(self):
        """Test that the jaxpr_jvp method raises an error if the order of the arguments does match the tape parameters."""

        def circuit(x, y):
            qml.RY(y, 0)
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (0.5, 0.6)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
            qml.device(device_name, wires=1).jaxpr_jvp(jaxpr.jaxpr, args, (0.5, 0.6))

        assert (
            "NotImplementedError: The provided arguments do not match the parameters of the jaxpr converted to quantum tape"
            in str(exc_info.value)
        )

    def test_wrong_number(self):
        """Test that the jaxpr_jvp method raises an error if the number of arguments does not match the tape parameters."""

        def f(x):
            @qml.for_loop(3)
            def g(i):
                qml.RX(x, i)

            g()
            return qml.expval(qml.Z(0))

        args = (0.5,)
        jaxpr = jax.make_jaxpr(f)(*args)

        with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
            qml.device(device_name, wires=1).jaxpr_jvp(jaxpr.jaxpr, args, (0.5,))

        assert (
            "NotImplementedError: The provided arguments do not match the parameters of the jaxpr converted to quantum tape"
            in str(exc_info.value)
        )

    def test_wrong_length(self):
        """Test that the jaxpr_jvp method raises an error if the length of the arguments does not match the tape parameters."""

        def circuit(x):
            qml.RX(x, 0)
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (0.5,)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
            qml.device(device_name, wires=1).jaxpr_jvp(jaxpr.jaxpr, args, (0.5,))

        assert (
            "NotImplementedError: The provided arguments do not match the parameters of the jaxpr converted to quantum tape"
            in str(exc_info.value)
        )

    def test_no_classical_preprocessing(self):
        """Test that the jaxpr_jvp method raises an error if the circuit contains classical preprocessing."""

        def f(x):
            y = x**2
            qml.RX(y[0], 0)
            qml.RX(y[1], 1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        x = jnp.array([1.5, 2.5])
        dx = jnp.array([2.0, 3.0])
        jaxpr = jax.make_jaxpr(f)(x)

        with pytest.raises(jax.errors.JaxRuntimeError) as exc_info:
            qml.device(device_name, wires=1).jaxpr_jvp(jaxpr.jaxpr, (x,), (dx,))

        assert (
            "NotImplementedError: The provided arguments do not match the parameters of the jaxpr converted to quantum tape"
            in str(exc_info.value)
        )

    def test_no_int_tangent(self):
        """Test that the jaxpr_jvp method raises an error if the tangents contain integers."""

        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (0.5,)
        tangents = (1,)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        with pytest.raises(ValueError, match="Tangents cannot be of integer type"):
            qml.device(device_name, wires=1).jaxpr_jvp(jaxpr.jaxpr, args, tangents)

    def test_no_shots(self):
        """Test that the jaxpr_jvp method raises an error if the circuit contains shots."""

        def circuit(x):
            qml.RX(x, 0)
            return qml.sample(qml.Z(0))

        args = (0.5,)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        with pytest.warns(
            qml.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
        ):
            with pytest.raises(
                NotImplementedError,
                match="LightningBase does not support finite shots for ``jaxpr_jvp``",
            ):
                qml.device(device_name, wires=1, shots=100).jaxpr_jvp(jaxpr.jaxpr, args, (0.5,))


class TestCorrectResults:
    """Test the correctness of the results and jvp for various circuits."""

    @pytest.mark.parametrize("use_jit", (False, True))
    def test_basic_circuit(self, use_jit):
        """Test the calculation of results and jvp for a basic circuit."""

        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        args = (0.82,)
        tangents = (2.0,)
        dev_jaxpr_jvp = qml.device(device_name, wires=1).jaxpr_jvp

        executor = partial(dev_jaxpr_jvp, jaxpr.jaxpr)
        if use_jit:
            executor = jax.jit(executor)

        results, dresults = executor(args, tangents)

        assert len(results) == 1
        assert qml.math.allclose(results, jnp.cos(args[0]))
        assert len(dresults) == 1
        assert qml.math.allclose(dresults[0], tangents[0] * -jnp.sin(args[0]))

    def test_diffentiable_op_math(self):
        """Test that we can handle differentiable op math in the circuit."""

        def f(x):
            qml.adjoint(qml.RX(x, 0))
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        args = (0.82,)
        tangents = (2.0,)
        dev_jaxpr_jvp = qml.device(device_name, wires=1).jaxpr_jvp

        results, dresults = dev_jaxpr_jvp(jaxpr.jaxpr, args, tangents)

        assert len(results) == 1
        assert qml.math.allclose(results, jnp.cos(args[0]))
        assert len(dresults) == 1
        assert qml.math.allclose(dresults[0], tangents[0] * -jnp.sin(args[0]))

    def test_abstract_zero_tangent(self):
        """Test we get the derivatives will be zero if the tangent is abstract zero."""

        def f(x):
            _ = x + 1
            qml.RX(0.5, 0)
            return qml.expval(qml.Z(0))

        args = (0.5,)
        tangents = (jax.interpreters.ad.Zero(jax.core.ShapedArray((), float)),)

        jaxpr = jax.make_jaxpr(f)(0.5)
        [results], [dresults] = qml.device(device_name, wires=1).jaxpr_jvp(
            jaxpr.jaxpr, args, tangents
        )

        assert qml.math.allclose(results, jnp.cos(0.5))
        assert qml.math.allclose(dresults, 0)

    def test_multiple_in(self):
        """Test that we can differentiate multiple inputs."""

        def f(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            qml.CNOT((0, 1))
            return qml.expval(qml.Y(0))

        x = jnp.array(0.5)
        y = jnp.array(1.2)
        dx = jnp.array(2.0)
        dy = jnp.array(3.0)

        jaxpr = jax.make_jaxpr(f)(x, y)

        dev_jaxpr_jvp = qml.device(device_name, wires=2).jaxpr_jvp

        [res], [dres] = dev_jaxpr_jvp(jaxpr.jaxpr, (x, y), (dx, dy))
        expected = -jnp.sin(x) * jnp.sin(y)
        assert qml.math.allclose(res, expected)

        expected_dres1 = dx * -jnp.cos(x) * jnp.sin(y)
        expected_dres2 = dy * -jnp.sin(x) * jnp.cos(y)
        expected_dres = expected_dres1 + expected_dres2
        assert qml.math.allclose(dres, expected_dres)

    def test_multiple_output(self):
        """Test we can compute the jvp with multiple outputs."""

        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        dev_jaxpr_jvp = qml.device(device_name, wires=2).jaxpr_jvp

        x = -0.5
        res, dres = dev_jaxpr_jvp(jaxpr.jaxpr, (x,), (2.0,))

        assert qml.math.allclose(res[0], 0)
        assert qml.math.allclose(res[1], -jnp.sin(x))
        assert qml.math.allclose(res[2], jnp.cos(x))

        assert qml.math.allclose(dres[0], 0)
        assert qml.math.allclose(dres[1], 2.0 * -jnp.cos(x))
        assert qml.math.allclose(dres[2], 2.0 * -jnp.sin(x))

    def test_multiple_in_and_out(self):
        """Test that we can differentiate multiple inputs and outputs."""

        def f(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            return qml.expval(qml.Y(0)), qml.expval(qml.X(1))

        x = jnp.array(0.5)
        y = jnp.array(1.2)
        dx = jnp.array(2.0)
        dy = jnp.array(3.0)

        jaxpr = jax.make_jaxpr(f)(x, y)

        args = (x, y)
        tangents = (dx, dy)

        dev_jaxpr_jvp = qml.device(device_name, wires=2).jaxpr_jvp

        res, dres = dev_jaxpr_jvp(jaxpr.jaxpr, args, tangents)

        expected = -jnp.sin(x), jnp.sin(y)
        assert qml.math.allclose(res, expected)

        expected_dres = dx * -jnp.cos(x), dy * jnp.cos(y)
        assert qml.math.allclose(dres, expected_dres)

    def test_input_array(self):
        """Test that the input array is handled correctly."""

        def f(x):
            qml.RX(x[0], 0)
            qml.RX(x[1], 1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        x = jnp.array([1.5, 2.5])
        dx = jnp.array([2.0, 3.0])
        jaxpr = jax.make_jaxpr(f)(x)

        dev_jaxpr_jvp = qml.device(device_name, wires=2).jaxpr_jvp

        [res], [dres] = dev_jaxpr_jvp(jaxpr.jaxpr, (x,), (dx,))

        expected = jnp.cos(x[0]) * jnp.cos(x[1])
        assert qml.math.allclose(res, expected)
        dexpected = -jnp.sin(x[0]) * dx[0] * jnp.cos(x[1]) + jnp.cos(x[0]) * -jnp.sin(x[1]) * dx[1]
        assert qml.math.allclose(dres, dexpected)

    def test_jaxpr_consts(self):
        """Test that we can execute jaxpr with consts."""

        def f():
            x = jnp.array([1.0])
            qml.RX(x[0], 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()

        dev_jaxpr_jvp = qml.device(device_name, wires=2).jaxpr_jvp

        const = jnp.array([1.2])
        dconst = jnp.array([0.25])
        [res], [dres] = dev_jaxpr_jvp(jaxpr.jaxpr, (const,), (dconst,))

        assert qml.math.allclose(res, jnp.cos(1.2))
        assert qml.math.allclose(dres, dconst[0] * -jnp.sin(1.2))

    def test_jaxpr_consts_and_traced_args(self):
        """Test that we can execute jaxpr with consts and traced arguments together."""

        const = jnp.array(0.5)

        def f(x):
            qml.RX(const, 0)
            qml.RY(x, 1)
            qml.CNOT((0, 1))
            return qml.expval(qml.Y(0))

        x = jnp.array(0.5)
        y = jnp.array(1.2)
        dx = jnp.array(2.0)
        dy = jnp.array(3.0)

        jaxpr = jax.make_jaxpr(f)(x)

        dev_jaxpr_jvp = qml.device(device_name, wires=2).jaxpr_jvp

        [res], [dres] = dev_jaxpr_jvp(jaxpr.jaxpr, (x, y), (dx, dy))
        expected = -jnp.sin(x) * jnp.sin(y)
        assert qml.math.allclose(res, expected)

        expected_dres = dx * -jnp.cos(x) * jnp.sin(y) + dy * -jnp.sin(x) * jnp.cos(y)
        assert qml.math.allclose(dres, expected_dres)

    def test_multi_param_op(self):
        """Test that we can differentiate multiple trainable parameters in one gate."""

        def f(x, y, z):
            qml.Rot(x, y, z, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5, 0.6, 0.7)

        dev_jaxpr_jvp = qml.device(device_name, wires=2).jaxpr_jvp

        x = (0.5, 0.6, 0.7)
        dx = (0.5, 0.6, 0.7)

        res, dres = dev_jaxpr_jvp(jaxpr.jaxpr, x, dx)

        expected = jnp.cos(x[1])
        assert qml.math.allclose(res, expected)

        expected_dres = x[1] * -jnp.sin(x[1])
        assert qml.math.allclose(dres, expected_dres)

    def test_multi_param_op_array(self):
        """Test that we can differentiate multiple trainable parameters in one gate, provided as an array."""

        def f(x):
            qml.Rot(x[0], x[1], x[2], 0)
            return qml.expval(qml.Z(0))

        dev_jaxpr_jvp = qml.device(device_name, wires=2).jaxpr_jvp

        x = jnp.array([0.5, 0.6, 0.7])
        dx = jnp.array([0.5, 0.6, 0.7])

        jaxpr = jax.make_jaxpr(f)(x)

        res, dres = dev_jaxpr_jvp(jaxpr.jaxpr, (x,), (dx,))

        expected = jnp.cos(x[1])
        assert qml.math.allclose(res, expected)

        expected_dres = x[1] * -jnp.sin(x[1])
        assert qml.math.allclose(dres, expected_dres)
