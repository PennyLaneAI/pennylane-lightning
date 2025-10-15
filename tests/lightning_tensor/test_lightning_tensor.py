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
from conftest import LightningDevice, device_name
from pennylane.exceptions import DeviceError
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
        with pytest.raises(DeviceError, match="Exact Tensor Network does not support StatePrep"):
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


@pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
class TestTensorNetMPS:
    """Test the MPS method of the LightningTensor class."""

    @pytest.mark.parametrize(
        "wires,max_bond,MPS_shape",
        [
            (2, 128, [[2, 2], [2, 2]]),
            (
                8,
                128,
                [
                    [2, 2],
                    [2, 2, 4],
                    [4, 2, 8],
                    [8, 2, 16],
                    [16, 2, 8],
                    [8, 2, 4],
                    [4, 2, 2],
                    [2, 2],
                ],
            ),
            (
                8,
                8,
                [[2, 2], [2, 2, 4], [4, 2, 8], [8, 2, 8], [8, 2, 8], [8, 2, 4], [4, 2, 2], [2, 2]],
            ),
            (15, 2, [[2, 2]] + [[2, 2, 2] for _ in range(13)] + [[2, 2]]),
        ],
    )
    def test_MPSPrep_check_pass(self, wires, max_bond, MPS_shape, c_dtype):
        """Test the correct behavior regarding MPS shape of MPSPrep."""
        MPS = [np.zeros(i, dtype=c_dtype) for i in MPS_shape]
        dev = LightningTensor(wires=wires, method="mps", max_bond_dim=max_bond, c_dtype=c_dtype)

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
                [
                    [2, 2],
                    [2, 2, 4],
                    [4, 2, 8],
                    [8, 2, 16],
                    [16, 2, 8],
                    [8, 2, 4],
                    [4, 2, 2],
                    [2, 2],
                ],
            ),  # Incorrect max bond dim.
            (
                15,
                2,
                [[2, 2]] + [[2, 2, 2] for _ in range(14)] + [[2, 2]],
            ),  # Incorrect amount of sites
        ],
    )
    def test_MPSPrep_check_fail(self, wires, max_bond, MPS_shape, c_dtype):
        """Test the exceptions regarding MPS shape of MPSPrep."""

        MPS = [np.zeros(i, dtype=c_dtype) for i in MPS_shape]
        dev = LightningTensor(wires=wires, method="mps", max_bond_dim=max_bond, c_dtype=c_dtype)
        dev_wires = dev.wires.tolist()

        def circuit(MPS):
            qml.MPSPrep(mps=MPS, wires=dev_wires)
            return qml.state()

        qnode_ltensor = qml.QNode(circuit, dev)

        with pytest.raises(
            RuntimeError,
            match="The incoming MPS does not have the correct layout for lightning.tensor",
        ):
            _ = qnode_ltensor(MPS)

    @pytest.mark.parametrize(
        "wires, MPS_shape",
        [
            (2, [[2, 2], [2, 2]]),
        ],
    )
    def test_MPSPrep_with_tn(self, wires, MPS_shape, c_dtype):
        """Test the exception of MPSPrep with the method exact tensor network (tn)."""

        MPS = [np.zeros(i, dtype=c_dtype) for i in MPS_shape]
        dev = LightningTensor(wires=wires, method="tn", c_dtype=c_dtype)
        dev_wires = dev.wires.tolist()

        def circuit(MPS):
            qml.MPSPrep(mps=MPS, wires=dev_wires)
            return qml.state()

        qnode_ltensor = qml.QNode(circuit, dev)

        with pytest.raises(DeviceError, match="Exact Tensor Network does not support MPSPrep"):
            _ = qnode_ltensor(MPS)

    @pytest.mark.parametrize(
        "MPS",
        [
            [
                np.array(
                    [
                        [-0.998685414638 + 0.0j, -0.051258585512 - 0.0j],
                        [0.047547165413 - 0.019149664485j, -0.926374774705 + 0.37309829027j],
                    ]
                ),
                np.array(
                    [
                        [
                            [-0.875169134426 - 0.456139031658j, 0.024624413522 - 0.060927908086j],
                            [0.101258621842 + 0.029812759206j, 0.025706347132 + 0.396610276574j],
                        ],
                        [
                            [-0.001058833849 + 0.027386007154j, 0.468536435427 + 0.018207079448j],
                            [-0.042532972535 - 0.110967979922j, -0.653527431685 + 0.436766422106j],
                        ],
                    ]
                ),
                np.array(
                    [
                        [0.906667523774 + 0.0j, 0.041728590132 - 0.270392753796j],
                        [-0.092760816556 + 0.0j, 0.046885022269 - 0.303805382427j],
                    ]
                ),
            ],  # Left canonical MPS
            [
                np.array(
                    [
                        [-0.947588819432 - 0.0j, -0.016185864786 + 0.0j],
                        [0.045114469162 - 0.018169893838j, -0.292520300633 + 0.117812819407j],
                    ]
                ),
                np.array(
                    [
                        [
                            [0.873519624977 + 0.455279305677j, -0.008333069977 + 0.020618420868j],
                            [-0.101067770672 - 0.029756568436j, -0.008699203711 - 0.134215630567j],
                        ],
                        [
                            [0.003175633836 - 0.082135578716j, -0.476435849301 - 0.018514046516j],
                            [0.12756406197 + 0.332813001859j, 0.664545750156 - 0.444130201043j],
                        ],
                    ]
                ),
                np.array(
                    [
                        [-0.957361955779 + 0.0j, -0.044061757604 + 0.285511203186j],
                        [0.288891131099 + 0.0j, -0.146017118194 + 0.946161146722j],
                    ]
                ),
            ],  # Right canonical MPS
            [
                np.array(
                    [
                        [-0.99868541 + 0.0j, -0.05125858 - 0.0j],
                        [0.04754716 - 0.01914966j, -0.92637477 + 0.3730982j],
                    ]
                ),
                np.array(
                    [
                        [
                            [-0.87516913 - 0.45613903j, 0.02462441 - 0.06092790j],
                            [0.10125862 + 0.02981275j, 0.02570634 + 0.39661027j],
                        ],
                        [
                            [-0.00105883 + 0.02738600j, 0.46853643 + 0.01820707j],
                            [-0.04253297 - 0.11096797j, -0.65352743 + 0.43676642j],
                        ],
                    ]
                ),
                np.array(
                    [
                        [0.90666752 + 0.0j, 0.04172859 - 0.27039275j],
                        [-0.09276081 + 0.0j, 0.04688502 - 0.30380538j],
                    ]
                ),
            ],  # Non-canonical MPS
        ],
    )
    def test_MPSPrep_expansion(self, MPS, c_dtype, tol):
        """Test the expansion of MPSPrep with the method matrix product state (mps)."""

        wires = 4

        dev = LightningTensor(wires=wires, method="mps", max_bond_dim=128, c_dtype=c_dtype)

        MPS = [np.array(i, dtype=c_dtype) for i in MPS]

        def circuit():
            qml.MPSPrep(MPS, wires=range(1, wires))
            [qml.Hadamard(i) for i in range(wires)]
            [qml.RX(0.1 * i, wires=i) for i in range(0, wires, 2)]
            return qml.expval(qml.PauliZ(1))

        qnode_ltensor = qml.QNode(circuit, dev)

        assert np.allclose(qnode_ltensor(), -0.076030545078943, atol=tol)

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

    def test_MPSPrep_bad_expansion(self, c_dtype):
        """Test the exception of MPSPrep with the method matrix product state (mps) trying to append a single wire at the beginning of the MPS."""

        wires = 9

        MPS_shape = [[2, 2], [2, 2, 4], [4, 2, 8], [8, 2, 4], [4, 2, 2], [2, 2]]
        MPS = [np.zeros(i, dtype=c_dtype) for i in MPS_shape]
        MPS_wires = len(MPS_shape)

        dev = LightningTensor(wires=wires, method="mps", max_bond_dim=8, c_dtype=c_dtype)

        def circuit():
            qml.MPSPrep(MPS, wires=range(1, MPS_wires))
            [qml.Hadamard(i) for i in range(wires)]
            return qml.expval(qml.PauliZ(1))

        qnode_ltensor = qml.QNode(circuit, dev)

        with pytest.raises(
            DeviceError,
            match="MPSPrep only support to append a single wire at the beginning of the MPS.",
        ):
            _ = qnode_ltensor()

    def test_MPSPrep_bad_expansion_with_wrong_MPS(self, c_dtype):
        """Test the exception of MPSPrep with the method matrix product state (mps) trying to pass a wrong MPS."""

        MPS_shape = [[2, 2], [2, 2, 4], [4, 2, 8], [8, 2, 4], [4, 2, 2], [2, 2]]
        MPS = [np.zeros(i, dtype=c_dtype) for i in MPS_shape]
        MPS_wires = len(MPS_shape)

        wires = MPS_wires + 1

        dev = LightningTensor(wires=wires, method="mps", max_bond_dim=8, c_dtype=c_dtype)

        def circuit():
            qml.MPSPrep(MPS, wires=range(wires))
            [qml.Hadamard(i) for i in range(wires)]
            return qml.expval(qml.PauliZ(1))

        qnode_ltensor = qml.QNode(circuit, dev)

        with pytest.raises(
            RuntimeError,
            match="Error in PennyLane Lightning: The incoming MPS does not have the correct layout for lightning.tensor.",
        ):
            _ = qnode_ltensor()
