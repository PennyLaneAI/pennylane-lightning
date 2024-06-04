# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the LightningTensor class.
"""


import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice  # tested device
from pennylane.wires import Wires

from pennylane_lightning.lightning_tensor import LightningTensor

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("Device doesn't have C++ support yet.", allow_module_level=True)


@pytest.mark.parametrize("num_wires", [3, 4, 5])
@pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
def test_device_name_and_init(num_wires, c_dtype):
    """Test the class initialization and returned properties."""
    wires = Wires(range(num_wires)) if num_wires else None
    dev = LightningTensor(wires=wires, maxBondDim=10, c_dtype=c_dtype)
    assert dev.name == "lightning.tensor"
    assert dev.c_dtype == c_dtype
    assert dev.wires == wires
    assert dev.num_wires == num_wires


def test_device_available_as_plugin():
    """Test that the device can be instantiated using ``qml.device``."""
    dev = qml.device("lightning.tensor", wires=2)
    assert isinstance(dev, LightningTensor)


@pytest.mark.parametrize("backend", ["fake_backend"])
def test_invalid_backend(backend):
    """Test an invalid backend."""
    with pytest.raises(ValueError, match=f"Unsupported backend: {backend}"):
        LightningTensor(backend=backend)


@pytest.mark.parametrize("method", ["fake_method"])
def test_invalid_method(method):
    """Test an invalid method."""
    with pytest.raises(ValueError, match=f"Unsupported method: {method}"):
        LightningTensor(method=method)


def test_invalid_bonddims():
    """Test that an error is raised if bonddims are less than 1."""
    with pytest.raises(ValueError):
        LightningTensor(wires=5, maxBondDim=0)


def test_invalid_wires_none():
    """Test that an error is raised if wires are none."""
    with pytest.raises(ValueError):
        LightningTensor(wires=None)


def test_invalid_shots():
    """Test that an error is raised if finite number of shots are requestd."""
    with pytest.raises(ValueError):
        LightningTensor(shots=5)


def test_support_derivatives():
    """Test that the device does not support derivatives yet."""
    dev = LightningTensor(wires=2)
    assert not dev.supports_derivatives()


def test_compute_derivatives():
    """Test that an error is raised if the `compute_derivatives` method is called."""
    dev = LightningTensor(wires=2)
    with pytest.raises(
        NotImplementedError,
        match="The computation of derivatives has yet to be implemented for the lightning.tensor device.",
    ):
        dev.compute_derivatives(circuits=None)


def test_execute_and_compute_derivatives():
    """Test that an error is raised if `execute_and_compute_derivative` method is called."""
    dev = LightningTensor(wires=2)
    with pytest.raises(
        NotImplementedError,
        match="The computation of derivatives has yet to be implemented for the lightning.tensor device.",
    ):
        dev.execute_and_compute_derivatives(circuits=None)


def test_supports_vjp():
    """Test that the device does not support VJP yet."""
    dev = LightningTensor(wires=2)
    assert not dev.supports_vjp()


def test_compute_vjp():
    """Test that an error is raised if `compute_vjp` method is called."""
    dev = LightningTensor(wires=2)
    with pytest.raises(
        NotImplementedError,
        match="The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device.",
    ):
        dev.compute_vjp(circuits=None, cotangents=None)


def test_execute_and_compute_vjp():
    """Test that an error is raised if `execute_and_compute_vjp` method is called."""
    dev = LightningTensor(wires=2)
    with pytest.raises(
        NotImplementedError,
        match="The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device.",
    ):
        dev.execute_and_compute_vjp(circuits=None, cotangents=None)
