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

import itertools
import math
from typing import Sequence

import numpy as np
import pennylane as qml
import pytest
from conftest import PHI, THETA, LightningDevice, device_name  # tested device
from pennylane.devices import DefaultExecutionConfig, DefaultQubit

if not LightningDevice._new_API:
    pytest.skip(
        "Exclusive tests for new API devices. Skipping.",
        allow_module_level=True,
    )

if device_name == "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestSimulate:
    """Tests for the simulate method."""

    @staticmethod
    def calculate_reference(tape):
        dev = DefaultQubit(max_workers=1)
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    @staticmethod
    def calculate_result(wires, tape, statevector):
        dev = LightningDevice(wires)
        return dev.simulate(circuit=tape, state=statevector)

    def test_simple_circuit(self, lightning_sv, tol):
        """Tests the simulate method for a simple circuit."""
        tape = qml.tape.QuantumScript(
            [qml.RX(THETA[0], wires=0), qml.RY(PHI[0], wires=1)],
            [qml.expval(qml.PauliX(0))],
            shots=None,
        )
        statevector = lightning_sv(num_wires=2)
        result = self.calculate_result(2, tape, statevector)
        reference = self.calculate_reference(tape)

        assert np.allclose(result, reference, tol)

    test_data_no_parameters = [
        (100, qml.PauliZ(wires=[0]), 100),
        (110, qml.PauliZ(wires=[1]), 110),
        (120, qml.PauliX(0) @ qml.PauliZ(1), 120),
    ]

    @pytest.mark.parametrize("num_shots,operation,shape", test_data_no_parameters)
    def test_sample_dimensions(self, lightning_sv, num_shots, operation, shape):
        """Tests if the samples returned by simulate have the correct dimensions"""
        ops = [qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])]
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=operation)], shots=num_shots)

        statevector = lightning_sv(num_wires=2)
        result = self.calculate_result(2, tape, statevector)

        assert np.array_equal(result.shape, (shape,))

    def test_sample_values(self, lightning_sv, tol):
        """Tests if the samples returned by simulate have the correct values"""
        ops = [qml.RX(1.5708, wires=[0])]
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=qml.PauliZ(0))], shots=1000)

        statevector = lightning_sv(num_wires=1)

        result = self.calculate_result(1, tape, statevector)

        assert np.allclose(result**2, 1, atol=tol, rtol=0)

    @pytest.mark.skipif(
        device_name != "lightning.qubit",
        reason=f"Device {device_name} does not have an mcmc option.",
    )
    @pytest.mark.parametrize("mcmc", [True, False])
    @pytest.mark.parametrize("kernel", ["Local", "NonZeroRandom"])
    def test_sample_values_with_mcmc(self, lightning_sv, tol, mcmc, kernel):
        """Tests if the samples returned by simulate have the correct values using mcmc"""
        ops = [qml.RX(1.5708, wires=[0])]
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=qml.PauliZ(0))], shots=1000)

        statevector = lightning_sv(num_wires=1)

        mcmc_param = {
            "mcmc": mcmc,
            "kernel_name": kernel,
            "num_burnin": 100,
        }

        execution_config = DefaultExecutionConfig

        dev = LightningDevice(wires=1)
        result = dev.simulate(
            circuit=tape, state=statevector, mcmc=mcmc_param, postselect_mode=execution_config
        )

        assert np.allclose(result**2, 1, atol=tol, rtol=0)
