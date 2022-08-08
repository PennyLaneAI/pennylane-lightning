# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests :mod:`pennylane_lightning.LightningQubit` device can be creaated.
"""
import pytest
import numpy as np
import pennylane as qml

from pennylane_lightning.lightning_qubit import CPP_BINARY_AVAILABLE

if not CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_create_device():
    dev = qml.device("lightning.qubit", wires=1)


@pytest.mark.parametrize("C", [np.complex64, np.complex128])
def test_create_device_with_dtype(C):
    dev = qml.device("lightning.qubit", wires=1, c_dtype=C)


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="Numpy only defines complex256 in Linux-like system"
)
def test_create_device_with_unsupported_dtype():
    with pytest.raises(TypeError, match="Unsupported complex Type:"):
        dev = qml.device("lightning.qubit", wires=1, c_dtype=np.complex256)


def test_no_op_arithmetic_support():
    """Test that lightning qubit explicitly does not support SProd, Prod, and Sum."""

    dev = qml.device("lightning.qubit", wires=2)
    for name in {"Prod", "SProd", "Sum"}:
        assert name not in dev.operations

    obs = qml.prod(qml.PauliX(0), qml.PauliY(1))

    @qml.qnode(dev)
    def circuit():
        return qml.expval(obs)

    with pytest.raises(qml.DeviceError, match=r"Observable Prod not supported on device"):
        circuit()
