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
This module tests the jaxpr_jvp method of the LightningQubit device.
"""
from functools import partial

import pennylane as qml
import pytest
from conftest import LightningDevice, device_name
from pennylane.devices import DefaultExecutionConfig

jax = pytest.importorskip("jax")
jaxlib = pytest.importorskip("jaxlib")

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

        execution_config = DefaultExecutionConfig
        execution_config.gradient_method = "backprop"

        with pytest.raises(
            NotImplementedError, match="LightningQubit does not support gradient_method"
        ):
            qml.device("lightning.qubit", wires=1).jaxpr_jvp(
                jaxpr.jaxpr, args, (0.5,), execution_config
            )

    def test_no_allowed_zeros(self):
        """Test that the jaxpr_jvp method raises an error if the tangents contain zeros."""

        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (0.5,)
        tangents = (jax.interpreters.ad.Zero(jax.core.ShapedArray((), float)),)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        with pytest.raises(
            NotImplementedError,
            match="tangents must not contain jax.interpreter.ad.Zero objects",
        ):
            qml.device("lightning.qubit", wires=1).jaxpr_jvp(jaxpr.jaxpr, args, tangents)

    def test_mismatch_args_tangents(self):
        """Test that the jaxpr_jvp method raises an error if the number of arguments and tangents do not match."""

        @qml.qnode(device=qml.device("lightning.qubit", wires=1))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (0.5,)
        tangents = (0.5, 0.5)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        with pytest.raises(
            NotImplementedError, match="The number of arguments and tangents must match"
        ):
            qml.device("lightning.qubit", wires=1).jaxpr_jvp(jaxpr.jaxpr, args, tangents)

    def test_only_measurements(self):
        """Test that the jaxpr_jvp method raises an error if the circuit does not return a measurement."""

        def circuit(x):
            qml.RX(x, 0)
            return x

        args = (0.5,)
        jaxpr = jax.make_jaxpr(circuit)(*args)

        with pytest.raises(
            NotImplementedError, match="The circuit should return measurement"
        ):
            qml.device("lightning.qubit", wires=1).jaxpr_jvp(jaxpr.jaxpr, args, (0.5,))


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
        dev_jaxpr_jvp = qml.device("lightning.qubit", wires=1).jaxpr_jvp

        executor = partial(dev_jaxpr_jvp, jaxpr.jaxpr)
        if use_jit:
            executor = jax.jit(executor)

        results, dresults = executor(args, tangents)

        assert len(results) == 1
        assert qml.math.allclose(results, jax.numpy.cos(args[0]))
        assert len(dresults) == 1
        assert qml.math.allclose(dresults[0], tangents[0] * -jax.numpy.sin(args[0]))

    def test_multiple_in(self):
        """Test that we can differentiate multiple inputs."""

        def f(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            qml.CNOT((0, 1))
            return qml.expval(qml.Y(0))

        x = jax.numpy.array(0.5)
        y = jax.numpy.array(1.2)
        dx = jax.numpy.array(2.0)
        dy = jax.numpy.array(3.0)

        jaxpr = jax.make_jaxpr(f)(x, y)

        dev_jaxpr_jvp = qml.device("lightning.qubit", wires=2).jaxpr_jvp

        [res], [dres] = dev_jaxpr_jvp(jaxpr.jaxpr, (x, y), (dx, dy))
        expected = -jax.numpy.sin(x) * jax.numpy.sin(y)
        assert qml.math.allclose(res, expected)

        expected_dres = dx * -jax.numpy.cos(x) * jax.numpy.sin(y) + dy * -jax.numpy.sin(
            x
        ) * jax.numpy.cos(y)
        assert qml.math.allclose(dres, expected_dres)

    def test_multiple_output(self):
        """Test we can compute the jvp with multiple outputs."""

        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        dev_jaxpr_jvp = qml.device("lightning.qubit", wires=2).jaxpr_jvp

        x = -0.5
        res, dres = dev_jaxpr_jvp(jaxpr.jaxpr, (x,), (2.0,))

        assert qml.math.allclose(res[0], 0)
        assert qml.math.allclose(res[1], -jax.numpy.sin(x))
        assert qml.math.allclose(res[2], jax.numpy.cos(x))

        assert qml.math.allclose(dres[0], 0)
        assert qml.math.allclose(dres[1], 2.0 * -jax.numpy.cos(x))
        assert qml.math.allclose(dres[2], 2.0 * -jax.numpy.sin(x))

    def test_input_array(self):
        """Test that the input array is handled correctly."""

        def f(x):
            qml.RX(x[0], 0)
            qml.RX(x[1], 1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

        x = jax.numpy.array([1.5, 2.5])
        dx = jax.numpy.array([2.0, 3.0])
        jaxpr = jax.make_jaxpr(f)(x)

        dev_jaxpr_jvp = qml.device("lightning.qubit", wires=2).jaxpr_jvp

        [res], [dres] = dev_jaxpr_jvp(jaxpr.jaxpr, (x,), (dx,))

        expected = jax.numpy.cos(x[0]) * jax.numpy.cos(x[1])
        assert qml.math.allclose(res, expected)
        dexpected = (
            -jax.numpy.sin(x[0]) * dx[0] * jax.numpy.cos(x[1])
            + jax.numpy.cos(x[0]) * -jax.numpy.sin(x[1]) * dx[1]
        )
        assert qml.math.allclose(dres, dexpected)

    def test_jaxpr_consts(self):
        """Test that we can execute jaxpr with consts."""

        def f():
            x = jax.numpy.array([1.0])
            qml.RX(x[0], 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)()

        dev_jaxpr_jvp = qml.device("lightning.qubit", wires=2).jaxpr_jvp

        const = jax.numpy.array([1.2])
        dconst = jax.numpy.array([0.25])
        [res], [dres] = dev_jaxpr_jvp(jaxpr.jaxpr, (const,), (dconst,))

        assert qml.math.allclose(res, jax.numpy.cos(1.2))
        assert qml.math.allclose(dres, dconst[0] * -jax.numpy.sin(1.2))

    def test_multiple_train_params(self):
        """Test that we can differentiate multiple trainable parameters."""

        def f(x, y, z):
            qml.Rot(x, y, z, 0)
            return qml.expval(qml.Z(0))
        
        jaxpr = jax.make_jaxpr(f)(0.5, 0.6, 0.7)

        dev_jaxpr_jvp = qml.device("lightning.qubit", wires=2).jaxpr_jvp

        x = (0.5, 0.6, 0.7)
        dx = (0.5, 0.6, 0.7)

        res, dres = dev_jaxpr_jvp(jaxpr.jaxpr, x, dx)

        expected = jax.numpy.array(0.8253356) 
        assert qml.math.allclose(res, expected)

        expected_dres = jax.numpy.array(-0.33878549)
        assert qml.math.allclose(dres, expected_dres)
