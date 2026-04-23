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

import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice as ld
from conftest import device_name

if device_name != "lightning.qubit":
    pytest.skip(
        f"Device {device_name} does not have an mcmc option. Skipping.",
        allow_module_level=True,
    )

if not ld._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestMCMCSample:
    """Tests that samples are properly calculated."""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qp.device(device_name, wires=2, mcmc=True, c_dtype=request.param)

    test_data_no_parameters = [
        (100, [0], qp.PauliZ(wires=[0]), 100),
        (110, [1], qp.PauliZ(wires=[1]), 110),
        (120, [0, 1], qp.PauliX(0) @ qp.PauliZ(1), 120),
    ]

    @pytest.mark.parametrize("num_shots,measured_wires,operation,shape", test_data_no_parameters)
    def test_mcmc_sample_dimensions(self, dev, num_shots, measured_wires, operation, shape):
        """Tests if the samples returned by sample have
        the correct dimensions
        """
        ops = [qp.RX(1.5708, wires=[0]), qp.RX(1.5708, wires=[1])]
        tape = qp.tape.QuantumScript(ops, [qp.sample(op=operation)], shots=num_shots)
        s1 = dev.execute(tape)

        assert np.array_equal(s1.shape, (shape,))

    @pytest.mark.parametrize("kernel", ["Local", "NonZeroRandom"])
    def test_sample_values(self, tol, kernel):
        """Tests if the samples returned by sample have
        the correct values
        """
        with pytest.warns(
            qp.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
        ):
            dev = qp.device(
                device_name,
                wires=2,
                shots=1000,
                mcmc=True,
                kernel_name=kernel,
                num_burnin=100,
            )
        ops = [qp.RX(1.5708, wires=[0])]
        tape = qp.tape.QuantumScript(ops, [qp.sample(op=qp.PauliZ(0))], shots=1000)
        s1 = dev.execute(tape)

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, atol=tol, rtol=0)

    @pytest.mark.parametrize("kernel", ["local", "nonZeroRandom", "Global", "global"])
    def test_unsupported_sample_kernels(self, tol, kernel):
        """Tests if the samples returned by sample have
        the correct values
        """
        with pytest.warns(
            qp.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
        ):
            # Create device (should not fail at initialization)
            dev = qp.device(
                device_name,
                wires=2,
                shots=1000,
                mcmc=True,
                kernel_name=kernel,
                num_burnin=100,
            )

        # Error should be raised during preprocess when validation runs
        with pytest.raises(
            NotImplementedError,
            match=f"The {kernel} is not supported and currently only 'Local' and 'NonZeroRandom' kernels are supported.",
        ):
            dev.preprocess()

    @pytest.mark.parametrize(["shots", "num_burnin"], [(10, 11), (1000, 1000)])
    def test_wrong_num_burnin(self, shots, num_burnin):
        with pytest.warns(
            qp.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
        ):
            # Create device (should not fail at initialization)
            dev = qp.device(
                device_name,
                wires=2,
                shots=shots,
                mcmc=True,
                kernel_name="Local",
                num_burnin=num_burnin,
            )

        # Error should be raised during preprocess when validation runs
        with pytest.raises(ValueError, match="Shots should be greater than num_burnin."):
            dev.preprocess()

    @pytest.mark.parametrize(["shots", "num_burnin"], [(10, 0), (1000, -1)])
    def test_unacceptable_num_burnin(self, shots, num_burnin):
        with pytest.warns(
            qp.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
        ):
            # Create device (should not fail at initialization)
            dev = qp.device(
                device_name,
                wires=2,
                shots=shots,
                mcmc=True,
                kernel_name="Local",
                num_burnin=num_burnin,
            )

        # Error should be raised during preprocess when validation runs
        with pytest.raises(ValueError, match="num_burnin must be greater than 0"):
            dev.preprocess()

    def test_invalid_kernel_name(self):
        """Test that an error is raised when the kernel_name is not "Local" or "NonZeroRandom"."""

        with pytest.warns(
            qp.exceptions.PennyLaneDeprecationWarning, match="shots on device is deprecated"
        ):
            ld(wires=2, shots=1000, mcmc=True, kernel_name="Local", num_burnin=100).preprocess()
            ld(
                wires=2, shots=1000, mcmc=True, kernel_name="NonZeroRandom", num_burnin=100
            ).preprocess()

            with pytest.raises(
                NotImplementedError,
                match="only 'Local' and 'NonZeroRandom' kernels are supported",
            ):
                ld(wires=2, shots=1000, mcmc=True, kernel_name="bleh").preprocess()
