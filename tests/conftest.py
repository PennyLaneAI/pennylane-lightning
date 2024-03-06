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
Pytest configuration file for PennyLane-Lightning test suite.
"""
import os

import numpy as np
import pennylane as qml
import pytest

import pennylane_lightning

# defaults
TOL = 1e-6
TOL_STOCHASTIC = 0.05

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array(
    [
        [
            -0.07843244 - 3.57825948e-01j,
            0.71447295 - 5.38069384e-02j,
            0.20949966 + 6.59100734e-05j,
            -0.50297381 + 2.35731613e-01j,
        ],
        [
            -0.26626692 + 4.53837083e-01j,
            0.27771991 - 2.40717436e-01j,
            0.41228017 - 1.30198687e-01j,
            0.01384490 - 6.33200028e-01j,
        ],
        [
            -0.69254712 - 2.56963068e-02j,
            -0.15484858 + 6.57298384e-02j,
            -0.53082141 + 7.18073414e-02j,
            -0.41060450 - 1.89462315e-01j,
        ],
        [
            -0.09686189 - 3.15085273e-01j,
            -0.53241387 - 1.99491763e-01j,
            0.56928622 + 3.97704398e-01j,
            -0.28671074 - 6.01574497e-02j,
        ],
    ]
)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session")
def tol_stochastic():
    """Numerical tolerance for equality tests."""
    return TOL_STOCHASTIC


@pytest.fixture(scope="session", params=[2, 3])
def n_subsystems(request):
    """Number of qubits or qumodes."""
    return request.param


# Looking for the device for testing.
default_device = "lightning.qubit"
supported_devices = {"lightning.kokkos", "lightning.qubit", "lightning.qubit2", "lightning.gpu"}
supported_devices.update({sb.replace(".", "_") for sb in supported_devices})


def get_device():
    """Return the pennylane lightning device.

    The device is ``lightning.qubit`` by default.
    Allowed values are: "lightning.kokkos" and "lightning.qubit".
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
import pennylane_lightning.lightning_qubit as lightning_ops  # Any definition of lightning_ops will do

if device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos import LightningKokkos as LightningDevice

    if hasattr(pennylane_lightning, "lightning_kokkos_ops"):
        import pennylane_lightning.lightning_kokkos_ops as lightning_ops
elif device_name == "lightning.gpu":
    from pennylane_lightning.lightning_gpu import LightningGPU as LightningDevice

    if hasattr(pennylane_lightning, "lightning_gpu_ops"):
        import pennylane_lightning.lightning_gpu_ops as lightning_ops
elif device_name == "lightning.qubit2":
    from pennylane_lightning.lightning_qubit2 import LightningQubit2 as LightningDevice

    if hasattr(pennylane_lightning, "lightning_qubit_ops"):
        import pennylane_lightning.lightning_qubit_ops as lightning_ops
else:
    from pennylane_lightning.lightning_qubit import LightningQubit as LightningDevice

    if hasattr(pennylane_lightning, "lightning_qubit_ops"):
        import pennylane_lightning.lightning_qubit_ops as lightning_ops


# General qubit_device fixture, for any number of wires.
@pytest.fixture(
    scope="function",
    params=[np.complex64, np.complex128],
)
def qubit_device(request):
    def _device(wires, shots=None):
        return qml.device(device_name, wires=wires, shots=shots, c_dtype=request.param)

    return _device
