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

# defaults
TOL = 1e-3


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))

@pytest.fixture(scope="session", params=[2, 3])
def n_subsystems(request):
    """Number of qubits or qumodes."""
    return request.param


@pytest.fixture(scope="function")
def qubit_device_1_wire():
    return qml.device('lightning.qubit', wires=1)


@pytest.fixture(scope="function")
def qubit_device_2_wires():
    return qml.device('lightning.qubit', wires=2)


@pytest.fixture(scope="function")
def qubit_device_3_wires():
    return qml.device('lightning.qubit', wires=3)
