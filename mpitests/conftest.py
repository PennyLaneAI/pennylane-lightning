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
Pytest configuration file for PennyLane-Lightning-GPU test suite.
"""
# pylint: disable=missing-function-docstring,wrong-import-order,unused-import

import itertools
import os

import pennylane as qml
import pytest
from pennylane import numpy as np

# Tuple passed to distributed device ctor
# np.complex for data type and True or False
# for enabling MPI or not.
fixture_params = itertools.product(
    [np.complex64, np.complex128],
    [True, False],
)

# defaults
TOL = 1e-6
TOL_STOCHASTIC = 0.05

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session", params=[2, 3])
def n_subsystems(request):
    """Number of qubits or qumodes."""
    return request.param


# Looking for the device for testing.
default_device = "lightning.gpu"
supported_devices = {"lightning.gpu"}
supported_devices.update({sb.replace(".", "_") for sb in supported_devices})


def get_device():
    """Return the pennylane lightning device.

    The device is ``lightning.gpu`` by default.
    Allowed values are: "lightning.gpu".
    An underscore can also be used instead of a dot.
    If the environment variable ``PL_DEVICE`` is defined, its value is used.
    Underscores are replaced by dots upon exiting.
    """
    device = None
    if "PL_DEVICE" in os.environ:
        device = os.environ.get("PL_DEVICE", default_device)
        device = device.replace("_", ".")
    if device is None:
        device = default_device
    if device not in supported_devices:
        raise ValueError(f"Invalid backend {device}.")
    return device


device_name = get_device()

if device_name not in qml.plugin_devices:
    raise qml.DeviceError(
        f"Device {device_name} does not exist. Make sure the required plugin is installed."
    )

# Device specification
if device_name == "lightning.gpu":
    from pennylane_lightning.lightning_gpu import LightningGPU as LightningDevice
    from pennylane_lightning.lightning_gpu._measurements import (
        LightningGPUMeasurements as LightningMeasurements,
    )
    from pennylane_lightning.lightning_gpu._state_vector import (
        LightningGPUStateVector as LightningStateVector,
    )

else:
    raise qml.DeviceError(f"The MPI tests do not apply to the {device_name} device.")


# General qubit_device fixture, for any number of wires.
@pytest.fixture(
    scope="function",
    params=fixture_params,
)
def qubit_device(request):
    def _device(wires):
        return qml.device(
            device_name,
            wires=wires,
            mpi=True,
            c_dtype=request.param[0],
            batch_obs=request.param[1],
        )

    return _device
