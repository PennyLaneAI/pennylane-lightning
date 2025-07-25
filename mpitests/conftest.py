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
Pytest configuration file for PennyLane-Lightning MPI test suite.
"""
# pylint: disable=missing-function-docstring,wrong-import-order,unused-import

import itertools
import os

import pennylane as qml
import pytest
from pennylane import numpy as np
from pennylane.exceptions import DeviceError

import pennylane_lightning

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
supported_devices = {"lightning.gpu", "lightning.kokkos"}
supported_devices.update({sb.replace(".", "_") for sb in supported_devices})


def get_device():
    """Return the pennylane lightning device.

    The device is ``lightning.gpu`` by default.
    Allowed values are: "lightning.gpu, lightning.kokkos".
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
    raise DeviceError(
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

    if hasattr(pennylane_lightning, "lightning_gpu_ops"):
        from pennylane_lightning.lightning_gpu_ops import LightningException
elif device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos import LightningKokkos as LightningDevice
    from pennylane_lightning.lightning_kokkos._measurements import (
        LightningKokkosMeasurements as LightningMeasurements,
    )
    from pennylane_lightning.lightning_kokkos._state_vector import (
        LightningKokkosStateVector as LightningStateVector,
    )

    if hasattr(pennylane_lightning, "lightning_kokkos_ops"):
        from pennylane_lightning.lightning_kokkos_ops import LightningException
else:
    raise DeviceError(f"The MPI tests do not apply to the {device_name} device.")


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


@pytest.fixture(autouse=True)
def restore_global_seed():
    original_state = np.random.get_state()
    yield
    np.random.set_state(original_state)


@pytest.fixture
def seed(request):
    """An integer random number generator seed

    This fixture overrides the ``seed`` fixture provided by pytest-rng, adding the flexibility
    of locally getting a new seed for a test case by applying the ``local_salt`` marker. This is
    useful when the seed from pytest-rng happens to be a bad seed that causes your test to fail.

    .. code_block:: python

        @pytest.mark.local_salt(42)
        def test_something(seed):
            ...

    The value passed to ``local_salt`` needs to be an integer.

    """

    fixture_manager = request._fixturemanager  # pylint:disable=protected-access
    fixture_defs = fixture_manager.getfixturedefs("seed", request.node)
    original_fixture_def = fixture_defs[0]  # the original seed fixture provided by pytest-rng
    original_seed = original_fixture_def.func(request)
    marker = request.node.get_closest_marker("local_salt")
    if marker and marker.args:
        return original_seed + marker.args[0]
    return original_seed
