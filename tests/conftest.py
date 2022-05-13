# Copyright 2020 Xanadu Quantum Technologies Inc.

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

import pytest
import numpy as np

import pennylane as qml
from pennylane_lightning import LightningQubit

# defaults
TOL = 1e-6

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session", params=[2, 3])
def n_subsystems(request):
    """Number of qubits or qumodes."""
    return request.param


@pytest.fixture(scope="function", params=[np.complex64, np.complex128])
def qubit_device_1_wire(request):
    return LightningQubit(wires=1, c_dtype=request.param)


@pytest.fixture(scope="function", params=[np.complex64, np.complex128])
def qubit_device_2_wires(request):
    return LightningQubit(wires=2, c_dtype=request.param)


@pytest.fixture(scope="function", params=[np.complex64, np.complex128])
def qubit_device_3_wires(request):
    return LightningQubit(wires=3, c_dtype=request.param)
