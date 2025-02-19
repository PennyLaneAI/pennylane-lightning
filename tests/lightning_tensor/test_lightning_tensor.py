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
from conftest import LightningDevice, LightningException, device_name
from pennylane.tape import QuantumScript
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


@pytest.mark.parametrize("method", [{"method": "mps", "max_bond_dim": 128}, {"method": "tn"}])
class TestTensorNet:

    @pytest.mark.parametrize("num_wires", [3, 4, 5])
    @pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
    def test_device_name_and_init(self, num_wires, c_dtype, method):
        """Test the class initialization and returned properties."""
        wires = Wires(range(num_wires)) if num_wires else None
        dev = LightningTensor(wires=wires, c_dtype=c_dtype, **method)

        assert dev.name == "lightning.tensor"
        assert dev.c_dtype == c_dtype
        assert dev.wires == wires
        assert dev.num_wires == num_wires

    def test_device_available_as_plugin(self, method):
        """Test that the device can be instantiated using ``qml.device``."""
        dev = qml.device("lightning.tensor", wires=2, **method)
        assert isinstance(dev, LightningTensor)
        assert dev.backend == "cutensornet"
        assert dev.method in ["mps", "tn"]

    def test_invalid_arg(self, method):
        """Test that an error is raised if an invalid argument is provided."""
        with pytest.raises(TypeError):
            LightningTensor(wires=2, kwargs="invalid_arg", **method)

    def test_invalid_bonddims_mps(self, method):
        """Test that an error is raised if bond dimensions are less than 1 in mps method."""
        if method["method"] == "mps":
            with pytest.raises(ValueError):
                LightningTensor(wires=5, max_bond_dim=0, method="mps")

    def test_invalid_bonddims_tn(self, method):
        """Test that an error is raised if bond dimensions are passing as arg in tn method."""
        if method["method"] == "tn":
            with pytest.raises(TypeError):
                LightningTensor(wires=5, max_bond_dim=10, method="tn")

    def test_invalid_cutoff_mode(self, method):
        """Test that an error is raised if an invalid cutoff mode is provided."""
        if method["method"] == "mps":
            with pytest.raises(ValueError):
                LightningTensor(wires=2, cutoff_mode="invalid_mode", **method)
            with pytest.raises(ValueError):
                LightningTensor(wires=2, cutoff_mode="abs", cutoff=-1e-1, **method)
        if method["method"] == "tn":
            with pytest.raises(TypeError):
                LightningTensor(wires=2, cutoff_mode="invalid_mode", **method)

    def test_unsupported_operations(self, method):
        """Test that an error is raised if an unsupported operation is applied."""
        if method["method"] == "mps":
            pytest.skip("Skipping test for MPS method.")
        dev = LightningTensor(wires=2, **method)

        tape = QuantumScript([qml.StatePrep(np.array([1, 0, 0, 0]), wires=[0, 1])])
        with pytest.raises(
            qml.DeviceError, match="Exact Tensor Network does not support StatePrep"
        ):
            dev.execute(tape)

    def test_support_derivatives(self, method):
        """Test that the device does not support derivatives yet."""
        dev = LightningTensor(wires=2, **method)
        assert not dev.supports_derivatives()

    def test_compute_derivatives(self, method):
        """Test that an error is raised if the `compute_derivatives` method is called."""
        dev = LightningTensor(wires=2, **method)
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the lightning.tensor device.",
        ):
            dev.compute_derivatives(circuits=None)

    def test_execute_and_compute_derivatives(self, method):
        """Test that an error is raised if `execute_and_compute_derivative` method is called."""
        dev = LightningTensor(wires=2, **method)
        with pytest.raises(
            NotImplementedError,
            match="The computation of derivatives has yet to be implemented for the lightning.tensor device.",
        ):
            dev.execute_and_compute_derivatives(circuits=None)

    def test_supports_vjp(self, method):
        """Test that the device does not support VJP yet."""
        dev = LightningTensor(wires=2, **method)
        assert not dev.supports_vjp()

    def test_compute_vjp(self, method):
        """Test that an error is raised if `compute_vjp` method is called."""
        dev = LightningTensor(wires=2, **method)
        with pytest.raises(
            NotImplementedError,
            match="The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device.",
        ):
            dev.compute_vjp(circuits=None, cotangents=None)

    def test_execute_and_compute_vjp(self, method):
        """Test that an error is raised if `execute_and_compute_vjp` method is called."""
        dev = LightningTensor(wires=2, **method)
        with pytest.raises(
            NotImplementedError,
            match="The computation of vector-Jacobian product has yet to be implemented for the lightning.tensor device.",
        ):
            dev.execute_and_compute_vjp(circuits=None, cotangents=None)


@pytest.mark.parametrize(
    "wires,max_bond,MPS_shape",
    [
        (2, 128, [[2, 2], [2, 2]]),
        (
            8,
            128,
            [[2, 2], [2, 2, 4], [4, 2, 8], [8, 2, 16], [16, 2, 8], [8, 2, 4], [4, 2, 2], [2, 2]],
        ),
        (8, 8, [[2, 2], [2, 2, 4], [4, 2, 8], [8, 2, 8], [8, 2, 8], [8, 2, 4], [4, 2, 2], [2, 2]]),
        (15, 2, [[2, 2]] + [[2, 2, 2] for _ in range(13)] + [[2, 2]]),
    ],
)
def test_MPSPrep_check_pass(wires, max_bond, MPS_shape):
    """Test the correct behavior regarding MPS shape of MPSPrep."""
    MPS = [np.zeros(i) for i in MPS_shape]
    dev = LightningTensor(wires=wires, method="mps", max_bond_dim=max_bond)
    dev_wires = dev.wires.tolist()

    def circuit(MPS):
        qml.MPSPrep(mps=MPS, wires=dev_wires)
        return qml.state()

    qnode_ltensor = qml.QNode(circuit, dev)

    _ = qnode_ltensor(MPS)


@pytest.mark.parametrize(
    "wires,max_bond,MPS_shape",
    [
        (
            8,
            8,
            [[2, 2], [2, 2, 4], [4, 2, 8], [8, 2, 16], [16, 2, 8], [8, 2, 4], [4, 2, 2], [2, 2]],
        ),  # Incorrect max bond dim.
        (15, 2, [[2, 2]] + [[2, 2, 2] for _ in range(14)] + [[2, 2]]),  # Incorrect amount of sites
    ],
)
def test_MPSPrep_check_fail(wires, max_bond, MPS_shape):
    """Test the exceptions regarding MPS shape of MPSPrep."""

    MPS = [np.zeros(i) for i in MPS_shape]
    dev = LightningTensor(wires=wires, method="mps", max_bond_dim=max_bond)
    dev_wires = dev.wires.tolist()

    def circuit(MPS):
        qml.MPSPrep(mps=MPS, wires=dev_wires)
        return qml.state()

    qnode_ltensor = qml.QNode(circuit, dev)

    with pytest.raises(
        LightningException,
        match="The incoming MPS does not have the correct layout for lightning.tensor",
    ):
        _ = qnode_ltensor(MPS)


@pytest.mark.parametrize(
    "wires, MPS_shape",
    [
        (2, [[2, 2], [2, 2]]),
    ],
)
def test_MPSPrep_with_tn(wires, MPS_shape):
    """Test the exception of MPSPrep with the method exact tensor network (tn)."""

    MPS = [np.zeros(i) for i in MPS_shape]
    dev = LightningTensor(wires=wires, method="tn")
    dev_wires = dev.wires.tolist()

    def circuit(MPS):
        qml.MPSPrep(mps=MPS, wires=dev_wires)
        return qml.state()

    qnode_ltensor = qml.QNode(circuit, dev)

    with pytest.raises(qml.DeviceError, match="Exact Tensor Network does not support MPSPrep"):
        _ = qnode_ltensor(MPS)


def test_MPSPrep_expansion():
    """Test the expansion of MPSPrep with the method matrix product state (mps)."""

    wires = 4
    MPS = [
        np.array(
            [[-0.998685 + 0.0j, -0.051259 - 0.0j], [0.047547 - 0.01915j, -0.926375 + 0.373098j]]
        ),
        np.array(
            [
                [
                    [-0.875169 - 0.456139j, 0.024624 - 0.060928j],
                    [0.101259 + 0.029813j, 0.025706 + 0.39661j],
                ],
                [
                    [-0.001059 + 0.027386j, 0.468536 + 0.018207j],
                    [-0.042533 - 0.110968j, -0.653527 + 0.436766j],
                ],
            ]
        ),
        np.array(
            [
                [0.906668 + 0.0j, 0.041729 - 0.270393j],
                [-0.092761 + 0.0j, 0.046885 - 0.303805j],
            ]
        ),
    ]

    dev = LightningTensor(wires=wires, method="mps", max_bond_dim=128)

    def circuit():
        qml.MPSPrep(MPS, wires=range(1, wires))
        [qml.Hadamard(i) for i in range(wires)]
        [qml.RX(0.1 * i, wires=i) for i in range(0, wires, 2)]
        return qml.expval(qml.PauliZ(1))

    qnode_ltensor = qml.QNode(circuit, dev)

    assert np.allclose(qnode_ltensor(), -0.076030545078943, atol=1e-10)

    # The reference value is obtained by running the same circuit on the default.qubit device.
    # random_state = [ 0.797003+0.406192j,  0.175192-0.199816j,
    #                 -0.090436+0.016981j, -0.13741 +0.003751j,
    #                 -0.013131-0.042167j,  0.015507+0.156004j,
    #                  0.036284+0.136789j, -0.143301-0.186339j]
    # def circuit():
    #     qml.StatePrep(random_state, wires=range(1, wires))
    #     [qml.Hadamard(i) for i in range(wires)]
    #     [qml.RX(0.1 * i, wires=i) for i in range(0, wires, 2)]
    #     return qml.expval(qml.PauliZ(1))
