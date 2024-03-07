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
This module contains unit tests for the LightningQubit2 class
"""

import pytest

import numpy as np
import pennylane as qml
from pennylane_lightning.lightning_qubit import LightningQubit, LightningQubit2
from pennylane_lightning.lightning_qubit.lightning_qubit2 import (
    simulate,
    jacobian,
    simulate_and_jacobian,
)
from pennylane.devices import DefaultQubit

from conftest import LightningDevice  # tested device

if LightningDevice != LightningQubit:
    pytest.skip("Exclusive tests for lightning.qubit. Skipping.", allow_module_level=True)

if not LightningQubit2._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


class TestHelpers:
    """Unit tests for the simulate function"""

    # Test simulate
    # Test jacobian + xfail tests
    # Test simulate_and_jacobian + xfail tests
    # Test stopping_condition
    # Test accepted_observables


class TestInitialization:
    """Unit tests for LightningQubit2 initialization"""

    # Test __init__ errors: invalid num_burnin, kernel name


class TestExecution:
    """Unit tests for executing quantum tapes on LightningQubit2"""

    # Test preprocess
    # Test execute


class TestDerivatives:
    """Unit tests for calculating derivatives with LightningQubit2"""

    # Test supports derivative + xfail tests
    # Test compute_derivatives + xfail tests
    # Test execute_and_compute_derivatives + xfail tests
