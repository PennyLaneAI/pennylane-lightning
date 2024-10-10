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
from functools import reduce
from typing import Sequence

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


#######################################################################


def validate_counts(shots, results1, results2):
    """Compares two counts.

    If the results are ``Sequence``s, loop over entries.

    Fails if a key of ``results1`` is not found in ``results2``.
    Passes if counts are too low, chosen as ``100``.
    Otherwise, fails if counts differ by more than ``20`` plus 20 percent.
    """
    if isinstance(results1, Sequence):
        assert isinstance(results2, Sequence)
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            validate_counts(shots, r1, r2)
        return
    for key1, val1 in results1.items():
        val2 = results2[key1]
        if abs(val1 + val2) > 100:
            assert np.allclose(val1, val2, rtol=20, atol=0.2)


def validate_samples(shots, results1, results2):
    """Compares two samples.

    If the results are ``Sequence``s, loop over entries.

    Fails if the results do not have the same shape, within ``20`` entries plus 20 percent.
    This is to handle cases when post-selection yields variable shapes.
    Otherwise, fails if the sums of samples differ by more than ``20`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, Sequence)
        assert isinstance(results2, Sequence)
        assert len(results1) == len(results2)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_samples(s, r1, r2)
    else:
        sh1, sh2 = results1.shape[0], results2.shape[0]
        assert np.allclose(sh1, sh2, rtol=20, atol=0.2)
        assert results1.ndim == results2.ndim
        if results2.ndim > 1:
            assert results1.shape[1] == results2.shape[1]
        np.allclose(np.sum(results1), np.sum(results2), rtol=20, atol=0.2)


def validate_others(shots, results1, results2):
    """Compares two expval, probs or var.

    If the results are ``Sequence``s, validate the average of items.

    If ``shots is None``, validate using ``np.allclose``'s default parameters.
    Otherwise, fails if the results do not match within ``0.01`` plus 20 percent.
    """
    if isinstance(results1, Sequence):
        assert isinstance(results2, Sequence)
        assert len(results1) == len(results2)
        results1 = reduce(lambda x, y: x + y, results1) / len(results1)
        results2 = reduce(lambda x, y: x + y, results2) / len(results2)
        validate_others(shots, results1, results2)
        return
    if shots is None:
        assert np.allclose(results1, results2)
        return
    assert np.allclose(results1, results2, atol=0.01, rtol=0.2)


def validate_measurements(func, shots, results1, results2):
    """Calls the correct validation function based on measurement type."""
    if func is qml.counts:
        validate_counts(shots, results1, results2)
        return

    if func is qml.sample:
        validate_samples(shots, results1, results2)
        return

    validate_others(shots, results1, results2)
