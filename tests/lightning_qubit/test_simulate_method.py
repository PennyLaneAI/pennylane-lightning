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
from conftest import THETA, PHI, LightningDevice, device_name  # tested device
from flaky import flaky
from pennylane.devices import DefaultQubit, DefaultExecutionConfig
from pennylane.measurements import VarianceMP
from scipy.sparse import csr_matrix, random_array

if device_name == "lightning.qubit":
    from pennylane_lightning.lightning_qubit._state_vector import LightningStateVector
    from pennylane_lightning.lightning_qubit.lightning_qubit import simulate


if device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos._state_vector import LightningStateVector
    from pennylane_lightning.lightning_kokkos.lightning_kokkos import simulate


if device_name != "lightning.qubit" and device_name != "lightning.kokkos":
    pytest.skip(
        "Exclusive tests for lightning.qubit and lightning.kokkos. Skipping.",
        allow_module_level=True,
    )

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


# General LightningStateVector fixture, for any number of wires.
@pytest.fixture(
    scope='module',
    params=[np.complex64, np.complex128],
)
def lightning_sv(request):
    def _statevector(num_wires):
        return LightningStateVector(num_wires=num_wires, dtype=request.param)

    return _statevector

class TestSimulate:
    """Tests for the simulate method."""
    
    @staticmethod
    def calculate_reference(tape):
        dev = DefaultQubit(max_workers=1)
        program, _ = dev.preprocess()
        tapes, transf_fn = program([tape])
        results = dev.execute(tapes)
        return transf_fn(results)

    def test_simple_circuit(self, lightning_sv, tol):
        """Tests the simulate method for a simple circuit."""
        tape = qml.tape.QuantumScript(
            [qml.RX(THETA[0], wires=0), 
             qml.RY(PHI[0], wires=1)],
            [qml.expval(qml.PauliX(0))],
            shots=None
        )
        statevector = lightning_sv(num_wires=2)
        result = simulate(circuit=tape, state=statevector)
        reference = self.calculate_reference(tape)
        
        assert np.allclose(result, reference, tol)
        
    test_data_no_parameters = [
        (100, qml.PauliZ(wires=[0]), 100),
        (110, qml.PauliZ(wires=[1]), 110),
        (120, qml.PauliX(0) @ qml.PauliZ(1), 120),
    ]
    
    @pytest.mark.parametrize("num_shots,operation,shape", test_data_no_parameters)
    def test_sample_dimensions(self, lightning_sv, num_shots, operation, shape):
        """Tests if the samples returned by simulate have the correct dimensions
        """
        ops = [qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])]
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=operation)], shots=num_shots)

        statevector = lightning_sv(num_wires=2)
        result = simulate(circuit=tape, state=statevector)
        
        assert np.array_equal(result.shape, (shape,))
        
    def test_sample_values(self, lightning_sv, tol):
        """Tests if the samples returned by simulate have the correct values
        """
        ops = [qml.RX(1.5708, wires=[0])]
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=qml.PauliZ(0))], shots=1000)

        statevector = lightning_sv(num_wires=1)

        result = simulate(circuit=tape, state=statevector)

        assert np.allclose(result**2, 1, atol=tol, rtol=0)


    @pytest.mark.skipif(device_name == "lightning.kokkos", 
                        reason=f"Device {device_name} does not have an mcmc option.")
    @pytest.mark.parametrize("mcmc", [True, False])
    @pytest.mark.parametrize("kernel", ["Local", "NonZeroRandom"])
    def test_sample_values_with_mcmc(self, lightning_sv, tol, mcmc, kernel):
        """Tests if the samples returned by simulate have the correct values
        """
        ops = [qml.RX(1.5708, wires=[0])]
        tape = qml.tape.QuantumScript(ops, [qml.sample(op=qml.PauliZ(0))], shots=1000)

        statevector = lightning_sv(num_wires=1)

        mcmc_param = {
            "mcmc": mcmc,
            "kernel_name": kernel,
            "num_burnin": 100,
        }

        execution_config = DefaultExecutionConfig

        result = simulate(circuit=tape, state=statevector, mcmc=mcmc_param, postselect_mode=execution_config)

        assert np.allclose(result**2, 1, atol=tol, rtol=0)