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
from functools import reduce
from typing import Sequence

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
supported_devices = {"lightning.kokkos", "lightning.qubit", "lightning.gpu", "lightning.tensor"}
supported_devices.update({sb.replace(".", "_") for sb in supported_devices})


def get_device():
    """Return the pennylane lightning device.

    The device is ``lightning.qubit`` by default. Allowed values are:
    "lightning.kokkos", and "lightning.qubit". An
    underscore can also be used instead of a dot. If the environment
    variable ``PL_DEVICE`` is defined, its value is used. Underscores
    are replaced by dots upon exiting.
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

LightningException = None

if device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos import LightningKokkos as LightningDevice

    if hasattr(pennylane_lightning, "lightning_kokkos_ops"):
        import pennylane_lightning.lightning_kokkos_ops as lightning_ops
        from pennylane_lightning.lightning_kokkos_ops import LightningException
elif device_name == "lightning.gpu":
    from pennylane_lightning.lightning_gpu import LightningGPU as LightningDevice

    if hasattr(pennylane_lightning, "lightning_gpu_ops"):
        import pennylane_lightning.lightning_gpu_ops as lightning_ops
        from pennylane_lightning.lightning_gpu_ops import LightningException
elif device_name == "lightning.tensor":
    from pennylane_lightning.lightning_tensor import LightningTensor as LightningDevice

    if hasattr(pennylane_lightning, "lightning_tensor_ops"):
        import pennylane_lightning.lightning_tensor_ops as lightning_ops
        from pennylane_lightning.lightning_tensor_ops import LightningException
else:
    from pennylane_lightning.lightning_qubit import LightningQubit as LightningDevice

    if hasattr(pennylane_lightning, "lightning_qubit_ops"):
        import pennylane_lightning.lightning_qubit_ops as lightning_ops
        from pennylane_lightning.lightning_qubit_ops import LightningException


# General qubit_device fixture, for any number of wires.
@pytest.fixture(
    scope="function",
    params=[np.complex64, np.complex128],
)
def qubit_device(request):
    def _device(wires, shots=None):
        if device_name == "lightning.tensor":
            return qml.device(device_name, wires=wires, c_dtype=request.param)
        return qml.device(device_name, wires=wires, shots=shots, c_dtype=request.param)

    return _device


#######################################################################
# Fixtures for testing under new and old opmath


@pytest.fixture(scope="function")
def use_legacy_opmath():
    with qml.operation.disable_new_opmath_cm() as cm:
        yield cm


@pytest.fixture(scope="function")
def use_new_opmath():
    with qml.operation.enable_new_opmath_cm() as cm:
        yield cm


@pytest.fixture(
    params=[qml.operation.disable_new_opmath_cm, qml.operation.enable_new_opmath_cm],
    scope="function",
)
def use_legacy_and_new_opmath(request):
    with request.param() as cm:
        yield cm


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
