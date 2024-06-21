# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for Lightning devices creation.
"""
import pickle as pkl
import sys

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice as ld
from conftest import device_name

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


def test_create_device():
    dev = qml.device(device_name, wires=1)


@pytest.mark.parametrize("C", [np.complex64, np.complex128])
def test_create_device_with_dtype(C):
    dev = qml.device(device_name, wires=1, c_dtype=C)


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="Numpy only defines complex256 in Linux-like system"
)
def test_create_device_with_unsupported_dtype():
    with pytest.raises(TypeError, match="Unsupported complex type:"):
        dev = qml.device(device_name, wires=1, c_dtype=np.complex256)


@pytest.mark.skipif(
    device_name != "lightning.kokkos",
    reason="Only lightning.kokkos has a kwarg kokkos_args.",
)
def test_create_device_with_unsupported_kokkos_args():
    with pytest.raises(TypeError, match="Argument kokkos_args must be of type"):
        dev = qml.device(device_name, wires=1, kokkos_args=np.complex256)


@pytest.mark.skipif(
    device_name != "lightning.gpu",
    reason="Only lightning.gpu has a kwarg mpi_buf_size.",
)
def test_create_device_with_unsupported_mpi_buf_size():
    try:
        from mpi4py import MPI

        with pytest.raises(ImportError, match="MPI related APIs are not found"):
            dev = qml.device(device_name, wires=1)
            dev._mpi_init_helper(1)
    except:
        pass


@pytest.mark.skipif(
    device_name != "lightning.gpu",
    reason="Check if the method is pickleable throught the cpp layer",
)
def test_devpool_is_pickleable():
    dev = qml.device(device_name, wires=2)
    try:
        pickled_devpool = pkl.dumps(dev._dp)
        un_pickled_devpool = pkl.loads(pickled_devpool)

        from pennylane_lightning.lightning_gpu_ops import DevPool
        d = DevPool()

        assert isinstance(un_pickled_devpool, DevPool)
        assert un_pickled_devpool.getTotalDevices() == d.getTotalDevices()


    except Exception:
        pytest.fail("DevPool should be Pickleable")


@pytest.mark.skipif(
    (device_name == "lightning.kokkos" and sys.platform == "win32"),
    reason="lightning.kokkos doesn't support 0 wires on Windows.",
)
@pytest.mark.skipif(
    device_name in ["lightning.gpu", "lightning.tensor"],
    reason=device_name + " doesn't support 0 wires.",
)
def test_device_init_zero_qubit():
    """Test the device initialization with zero-qubit."""

    dev = qml.device(device_name, wires=0)

    @qml.qnode(dev)
    def circuit():
        return qml.state()

    assert np.allclose(circuit(), np.array([1.0]))
