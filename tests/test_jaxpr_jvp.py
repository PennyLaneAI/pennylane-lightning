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

from pennylane_lightning.lightning_qubit.lightning_qubit import execute_and_jvp

jax = pytest.importorskip("jax")
jaxlib = pytest.importorskip("jaxlib")

if device_name == "lightning.tensor":
    pytest.skip(
        "Skipping tests for the LightningTensor class.", allow_module_level=True
    )

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


class TestErrors:
    """Test explicit errors for various unsupported cases."""

    def test_no_allowed_zeros(self):

        @qml.qnode(device=qml.device("lightning.qubit", wires=1))
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
            execute_and_jvp(jaxpr.jaxpr, args, tangents, num_wires=1)

    def test_mismatch_args_tangents(self):

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
            execute_and_jvp(jaxpr.jaxpr, args, tangents, num_wires=1)
