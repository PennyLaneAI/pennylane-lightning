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
import configparser
import hashlib
import os
from functools import reduce
from typing import Sequence

import numpy as np
import pennylane as qml
import pytest
from pennylane.exceptions import DeviceError
from scipy.sparse import csr_matrix, random_array

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
    raise DeviceError(
        f"Device {device_name} does not exist. Make sure the required plugin is installed."
    )

# Device specification
import pennylane_lightning.lightning_qubit as lightning_ops  # Any definition of lightning_ops will do

LightningException = None

if device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos import LightningKokkos as LightningDevice
    from pennylane_lightning.lightning_kokkos._adjoint_jacobian import (
        LightningKokkosAdjointJacobian as LightningAdjointJacobian,
    )
    from pennylane_lightning.lightning_kokkos._measurements import (
        LightningKokkosMeasurements as LightningMeasurements,
    )
    from pennylane_lightning.lightning_kokkos._state_vector import (
        LightningKokkosStateVector as LightningStateVector,
    )

    if hasattr(pennylane_lightning, "lightning_kokkos_ops"):
        import pennylane_lightning.lightning_kokkos_ops as lightning_ops
        from pennylane_lightning.lightning_kokkos_ops import LightningException
elif device_name == "lightning.gpu":
    from pennylane_lightning.lightning_gpu import LightningGPU as LightningDevice
    from pennylane_lightning.lightning_gpu._adjoint_jacobian import (
        LightningGPUAdjointJacobian as LightningAdjointJacobian,
    )
    from pennylane_lightning.lightning_gpu._measurements import (
        LightningGPUMeasurements as LightningMeasurements,
    )
    from pennylane_lightning.lightning_gpu._state_vector import (
        LightningGPUStateVector as LightningStateVector,
    )

    if hasattr(pennylane_lightning, "lightning_gpu_ops"):
        import pennylane_lightning.lightning_gpu_ops as lightning_ops
        from pennylane_lightning.lightning_gpu_ops import LightningException
elif device_name == "lightning.tensor":
    from pennylane_lightning.lightning_tensor import LightningTensor as LightningDevice
    from pennylane_lightning.lightning_tensor._measurements import (
        LightningTensorMeasurements as LightningMeasurements,
    )
    from pennylane_lightning.lightning_tensor._tensornet import (
        LightningTensorNet as LightningStateVector,
    )

    LightningAdjointJacobian = None

    if hasattr(pennylane_lightning, "lightning_tensor_ops"):
        import pennylane_lightning.lightning_tensor_ops as lightning_ops
        from pennylane_lightning.lightning_tensor_ops import LightningException
else:
    from pennylane_lightning.lightning_qubit import LightningQubit as LightningDevice
    from pennylane_lightning.lightning_qubit._adjoint_jacobian import LightningAdjointJacobian
    from pennylane_lightning.lightning_qubit._measurements import LightningMeasurements
    from pennylane_lightning.lightning_qubit._state_vector import LightningStateVector

    if hasattr(pennylane_lightning, "lightning_qubit_ops"):
        import pennylane_lightning.lightning_qubit_ops as lightning_ops
        from pennylane_lightning.lightning_qubit_ops import LightningException


# General qubit_device fixture, for any number of wires.
@pytest.fixture(
    scope="function",
    params=[np.complex64, np.complex128],
)
def qubit_device(request):
    def _device(wires, shots=None, seed=None):
        if device_name == "lightning.tensor":
            return qml.device(device_name, wires=wires, shots=shots, c_dtype=request.param)
        else:
            return qml.device(
                device_name, wires=wires, shots=shots, c_dtype=request.param, seed=seed
            )

    return _device


# General LightningStateVector fixture, for any number of wires.
@pytest.fixture(
    scope="function",
    params=(
        [np.complex64, np.complex128]
        if device_name != "lightning.tensor"
        else [
            [c_dtype, method]
            for c_dtype in [np.complex64, np.complex128]
            for method in ["mps", "tn"]
        ]
    ),
)
def lightning_sv(request):
    def _statevector(num_wires, seed=None):
        if device_name == "lightning.tensor":
            return LightningStateVector(
                num_wires=num_wires, c_dtype=request.param[0], method=request.param[1]
            )
        if seed:
            rng = np.random.default_rng(seed)
            return LightningStateVector(num_wires=num_wires, dtype=request.param, rng=rng)
        return LightningStateVector(num_wires=num_wires, dtype=request.param)

    return _statevector


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


def validate_counts(shots, results1, results2, rtol=0.15, atol=30):
    """Compares two counts.

    If the results are ``Sequence``s, loop over entries.

    Fails if a key of ``results1`` is not found in ``results2``.
    Passes if counts are too low, chosen as ``100``.
    Otherwise, fails if counts differ by more than 15 percent plus ``30``.
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
            assert np.allclose(val1, val2, rtol=rtol, atol=atol)


def validate_samples(shots, results1, results2, rtol=0.15, atol=30):
    """Compares two samples.

    If the results are ``Sequence``s, loop over entries.

    Fails if the results do not have the same shape, within 15 percent plus ``30`` entries.
    This is to handle cases when post-selection yields variable shapes.
    Otherwise, fails if the sums of samples differ by more than 15 percent plus ``30``.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, Sequence)
        assert isinstance(results2, Sequence)
        assert len(results1) == len(results2)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_samples(s, r1, r2)
    else:
        sh1, sh2 = results1.shape[0], results2.shape[0]
        assert np.allclose(sh1, sh2, rtol=rtol, atol=atol)
        assert results1.ndim == results2.ndim
        if results2.ndim > 1:
            assert results1.shape[1] == results2.shape[1]
        np.allclose(np.sum(results1), np.sum(results2), rtol=rtol, atol=atol)


def validate_others(shots, results1, results2, atol=0.01, rtol=0.2):
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
    assert np.allclose(results1, results2, atol=atol, rtol=rtol)


def validate_measurements(func, shots, results1, results2, atol=0.01, rtol=0.2):
    """Calls the correct validation function based on measurement type."""
    if func is qml.counts:
        validate_counts(shots, results1, results2, rtol=rtol)
        return

    if func is qml.sample:
        validate_samples(shots, results1, results2, rtol=rtol)
        return

    validate_others(shots, results1, results2, atol=atol, rtol=rtol)


# This hook is called to add info to test report header
def pytest_report_header():
    return [
        "",
        "PennyLane-Lightning Test Suite",
        f"::: Device: {device_name:<17} :::",
    ]


# Extract _default_rng_seed from `rng_salt` in pytest.ini
# which are used for generating random vectors/matrices in functions below
config = configparser.ConfigParser()
pytest_ini_path = os.path.join(os.path.dirname(__file__), "pytest.ini")
read_files = config.read(pytest_ini_path)
config.read("pytest.ini")
rng_salt = config["pytest"]["rng_salt"]
_default_rng_seed = int(hashlib.sha256(rng_salt.encode()).hexdigest(), 16)


def get_random_matrix(n, seed=None):
    """Generate a random complex matrix of shape (n, n)."""
    seed = seed or _default_rng_seed
    rng = np.random.default_rng(seed=seed)
    U = rng.random((n, n)) + 1.0j * rng.random((n, n))
    return U


def get_random_normalized_state(n, seed=None):
    """Generate a random normalized complex state vector of n elements."""
    seed = seed or _default_rng_seed
    rng = np.random.default_rng(seed=seed)
    random_state = rng.random(n) + 1j * rng.random(n)
    return random_state / np.linalg.norm(random_state)


def get_hermitian_matrix(n, seed=None):
    """Generate a random Hermitian matrix of shape (n, n)."""
    seed = seed or _default_rng_seed
    rng = np.random.default_rng(seed=seed)
    H = rng.random((n, n)) + 1.0j * rng.random((n, n))
    return H + np.conj(H).T


def get_sparse_hermitian_matrix(n):
    """Generate a random sparse Hermitian matrix of shape (n, n)."""
    H = random_array((n, n), density=0.15)
    H = H + 1.0j * random_array((n, n), density=0.15)
    return csr_matrix(H + H.conj().T)
