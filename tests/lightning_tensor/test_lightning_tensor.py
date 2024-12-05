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
from conftest import LightningDevice, device_name  # tested device
from pennylane.wires import Wires

if device_name != "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)
else:
    from pennylane_lightning.lightning_tensor import LightningTensor

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("Device doesn't have C++ support yet.", allow_module_level=True)


@pytest.mark.parametrize("backend", ["fake_backend"])
def test_invalid_backend(backend):
    """Test an invalid backend."""
    with pytest.raises(ValueError, match=f"Unsupported backend: {backend}"):
        LightningTensor(wires=1, backend=backend)


@pytest.mark.parametrize("method", ["fake_method"])
def test_invalid_method(method):
    """Test an invalid method."""
    with pytest.raises(ValueError, match=f"Unsupported method: {method}"):
        LightningTensor(method=method)


@pytest.mark.parametrize("method", ["mps", "tn"])
class TestTensorNet:

    @pytest.mark.parametrize("num_wires", [3, 4, 5])
    @pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
    def test_device_name_and_init(self, num_wires, c_dtype, method):
        """Test the class initialization and returned properties."""
        wires = Wires(range(num_wires)) if num_wires else None
        if method == "mps":
            dev = LightningTensor(wires=wires, max_bond_dim=10, c_dtype=c_dtype, method=method)
        if method == "tn":
            dev = LightningTensor(wires=wires, c_dtype=c_dtype, method=method)

        assert dev.name == "lightning.tensor"
        assert dev.c_dtype == c_dtype
        assert dev.wires == wires
        assert dev.num_wires == num_wires

    def test_device_available_as_plugin(self, method):
        """Test that the device can be instantiated using ``qml.device``."""
        dev = qml.device("lightning.tensor", wires=2, method=method)
        assert isinstance(dev, LightningTensor)
        assert dev.backend == "cutensornet"
        assert dev.method in ["mps", "tn"]

    def test_invalid_arg(self, method):
        """Test that an error is raised if an invalid argument is provided."""
        with pytest.raises(TypeError):
            LightningTensor(wires=2, kwargs="invalid_arg", method=method)

    def test_invalid_bonddims_mps(self, method):
        """Test that an error is raised if bond dimensions are less than 1."""
        if method == "mps":
            with pytest.raises(ValueError):
                LightningTensor(wires=5, max_bond_dim=0, method=method)

    def test_invalid_bonddims_tn(self, method):
        """Test that an error is raised if bond dimensions are less than 1."""
        if method == "tn":
            with pytest.raises(TypeError):
                LightningTensor(wires=5, max_bond_dim=10, method=method)

    def test_invalid_wires_none(self, method):
        """Test that an error is raised if wires are none."""
        with pytest.raises(ValueError):
            LightningTensor(wires=None, method=method)

    def test_invalid_cutoff_mode(self, method):
        """Test that an error is raised if an invalid cutoff mode is provided."""
        with pytest.raises(ValueError):
            LightningTensor(wires=2, cutoff_mode="invalid_mode", method=method)

    def test_support_derivatives(self, method):
        """Test that the device does not support derivatives yet."""
        dev = LightningTensor(wires=2, method=method)
        assert not dev.supports_derivatives()

    def test_compute_derivatives(self, method):
        """Test that an error is raised if the `compute_derivatives` method is called."""
        dev = LightningTensor(wires=2, method=method)
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the lightning.tensor device.",
        ):
            dev.compute_derivatives(circuits=None)

    def test_execute_and_compute_derivatives(self, method):
        """Test that an error is raised if `execute_and_compute_derivative` method is called."""
        dev = LightningTensor(wires=2, method=method)
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the lightning.tensor device.",
        ):
            dev.execute_and_compute_derivatives(circuits=None)

    def test_supports_vjp(self, method):
        """Test that the device does not support VJP yet."""
        dev = LightningTensor(wires=2, method=method)
        assert not dev.supports_vjp()

    def test_compute_vjp(self, method):
        """Test that an error is raised if `compute_vjp` method is called."""
        dev = LightningTensor(wires=2, method=method)
        with pytest.raises(
            NotImplementedError,
            match="The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device.",
        ):
            dev.compute_vjp(circuits=None, cotangents=None)

    def test_execute_and_compute_vjp(self, method):
        """Test that an error is raised if `execute_and_compute_vjp` method is called."""
        dev = LightningTensor(wires=2, method=method)
        with pytest.raises(
            NotImplementedError,
            match="The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device.",
        ):
            dev.execute_and_compute_vjp(circuits=None, cotangents=None)
