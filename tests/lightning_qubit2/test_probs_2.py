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
Tests for process and execute (probability calculation).
"""
import pytest
from conftest import LightningDevice

import numpy as np
import pennylane as qml

from pennylane_lightning.lightning_qubit import LightningQubit2

if LightningDevice != LightningQubit2:
    pytest.skip("Exclusive tests for lightning.qubit2. Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.fixture(params=[np.complex64, np.complex128])
def dev(request):
    return LightningDevice(wires=3, c_dtype=request.param)


def calculate_reference(tape):
    dev = qml.device("default.qubit", max_workers=1)
    program, _ = dev.preprocess()
    tapes, transf_fn = program([tape])
    results = dev.execute(tapes)
    return transf_fn(results)


def process_and_execute(dev, tape):
    program, _ = dev.preprocess()
    tapes, transf_fn = program([tape])
    results = dev.execute(tapes)
    return transf_fn(results)
