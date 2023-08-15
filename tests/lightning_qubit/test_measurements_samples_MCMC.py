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
Unit tests for MCMC sampling in lightning.qubit.
"""
import pytest
from conftest import LightningDevice  # tested device

import numpy as np
import pennylane as qml

import pytest

from pennylane_lightning.lightning_qubit import LightningQubit


if not LightningQubit._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if LightningDevice != LightningQubit:
    pytest.skip("Exclusive tests for lightning.qubit. Skipping.", allow_module_level=True)


class TestMCMCSample:
    """Tests that samples are properly calculated."""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=2, shots=1000, mcmc=True, c_dtype=request.param)

    test_data_no_parameters = [
        (100, [0], qml.PauliZ(wires=[0]), 100),
        (110, [1], qml.PauliZ(wires=[1]), 110),
        (120, [0, 1], qml.PauliX(0) @ qml.PauliZ(1), 120),
    ]

    @pytest.mark.parametrize("num_shots,measured_wires,operation,shape", test_data_no_parameters)
    def test_mcmc_sample_dimensions(self, dev, num_shots, measured_wires, operation, shape):
        """Tests if the samples returned by sample have
        the correct dimensions
        """
        dev.apply([qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])])

        dev.shots = num_shots
        dev._wires_measured = measured_wires
        dev._samples = dev.generate_samples()
        s1 = dev.sample(operation)

        assert np.array_equal(s1.shape, (shape,))

    @pytest.mark.parametrize("kernel", ["Local", "NonZeroRandom"])
    def test_sample_values(self, tol, kernel):
        """Tests if the samples returned by sample have
        the correct values
        """
        dev = qml.device(
            "lightning.qubit", wires=2, shots=1000, mcmc=True, kernel_name=kernel, num_burnin=100
        )

        dev.apply([qml.RX(1.5708, wires=[0])])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()
        s1 = dev.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("kernel", ["local", "nonZeroRandom", "Global", "global"])
    def test_unsupported_sample_kernels(self, tol, kernel):
        """Tests if the samples returned by sample have
        the correct values
        """
        with pytest.raises(
            NotImplementedError,
            match=f"The {kernel} is not supported and currently only 'Local' and 'NonZeroRandom' kernels are supported.",
        ):
            dev = qml.device(
                "lightning.qubit",
                wires=2,
                shots=1000,
                mcmc=True,
                kernel_name=kernel,
                num_burnin=100,
            )

    def test_wrong_num_burnin(self):
        with pytest.raises(ValueError, match="Shots should be greater than num_burnin."):
            dev = qml.device("lightning.qubit", wires=2, shots=1000, mcmc=True, num_burnin=1000)
