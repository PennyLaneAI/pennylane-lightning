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
Unit tests for lightning base statevector class
"""

import math

import numpy as np
import pennylane as qp
import pytest
import scipy as sp
from conftest import LightningDevice, LightningStateVector, device_name  # tested device
from pennylane.exceptions import DeviceError
from pennylane.tape import QuantumScript
from pennylane.wires import Wires
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

if device_name == "lightning.kokkos":
    try:
        from pennylane_lightning.lightning_kokkos_ops import InitializationSettings
    except ImportError:
        pass

if device_name == "lightning.gpu":
    from pennylane_lightning.lightning_gpu._mpi_handler import MPIHandler

if device_name == "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)


if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.parametrize("num_wires", range(4))
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_device_name_and_init(num_wires, dtype):
    """Test the class initialization and returned properties."""
    state_vector = LightningStateVector(num_wires, dtype=dtype)
    assert state_vector.dtype == dtype
    if device_name == "lightning.amdgpu":
        assert state_vector.device_name == "lightning.kokkos"
    else:
        assert state_vector.device_name == device_name
    assert state_vector.wires == Wires(range(num_wires))

    if device_name == "lightning.kokkos":
        bad_kokkos_args = np.array([33])
        with pytest.raises(
            TypeError,
            match=f"Argument kokkos_args must be of type {type(InitializationSettings())} but it is of {type(bad_kokkos_args)}.",
        ):
            assert LightningStateVector(num_wires, dtype=dtype, kokkos_args=bad_kokkos_args)

        set_kokkos_args = InitializationSettings().set_num_threads(2)
        state_vector_3 = LightningStateVector(num_wires, dtype=dtype, kokkos_args=set_kokkos_args)

        assert type(state_vector) == type(state_vector_3)


@pytest.mark.parametrize("dtype", [np.double])
def test_wrong_dtype(dtype):
    """Test if the class returns a TypeError for a wrong dtype"""
    with pytest.raises(TypeError, match="Unsupported complex type:"):
        assert LightningStateVector(3, dtype=dtype)


def test_errors_basis_state():
    with pytest.raises(ValueError, match="Basis state must only consist of 0s and 1s;"):
        state_vector = LightningStateVector(2)
        state_vector.apply_operations([qp.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])])
    with pytest.raises(ValueError, match="State must be of length 1;"):
        state_vector = LightningStateVector(1)
        state_vector.apply_operations([qp.BasisState(np.array([0, 1]), wires=[0])])


def test_apply_state_vector_with_lightning_handle(tol):
    state_vector_1 = LightningStateVector(2)
    state_vector_1.apply_operations([qp.BasisState(np.array([0, 1]), wires=[0, 1])])

    if device_name == "lightning.gpu":
        with pytest.raises(
            DeviceError,
            match="LightningGPU does not support allocate external state_vector.",
        ):
            state_vector_2 = LightningStateVector(2)
            state_vector_2._apply_state_vector(state_vector_1.state_vector, Wires([0, 1]))

    else:
        state_vector_2 = LightningStateVector(2)
        state_vector_2._apply_state_vector(state_vector_1.state_vector, Wires([0, 1]))

        assert np.allclose(state_vector_1.state, state_vector_2.state, atol=tol, rtol=0)


@pytest.mark.parametrize(
    "state",
    [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
        [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
    ],
)
@pytest.mark.parametrize(
    "sparse_rep",
    [
        coo_matrix,
        csr_matrix,
        csc_matrix,
    ],
)
def test_apply_operation_sparse_state_preparation(tol, sparse_rep, state):
    """Tests that applying an StatePrep operation works with sparse data representation."""

    wires = 2
    state_vector = LightningStateVector(wires)
    sparse_state = sparse_rep(state)
    state_vector.apply_operations([qp.StatePrep(sparse_state, Wires(range(wires)))])

    assert np.allclose(state_vector.state, np.array(state), atol=tol, rtol=0)


@pytest.mark.parametrize(
    "operation,expected_output,par",
    [
        (qp.BasisState, [0, 0, 1, 0], [1, 0]),
        (qp.BasisState, [0, 0, 0, 1], [1, 1]),
        (qp.StatePrep, [0, 0, 1, 0], [0, 0, 1, 0]),
        (qp.StatePrep, [0, 0, 0, 1], [0, 0, 0, 1]),
        (
            qp.StatePrep,
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
        ),
        (
            qp.StatePrep,
            [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
            [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
        ),
    ],
)
def test_apply_operation_state_preparation(tol, operation, expected_output, par):
    """Tests that applying an operation yields the expected output state for single wire
    operations that have no parameters."""

    wires = 2
    state_vector = LightningStateVector(wires)
    state_vector.apply_operations([operation(np.array(par), Wires(range(wires)))])

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)


@pytest.mark.parametrize(
    "operation,par",
    [
        (qp.BasisState, [1, 0]),
        (
            qp.StatePrep,
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
        ),
    ],
)
def test_reset_state(tol, operation, par):
    """Tests that applying an operation yields the expected output state for single wire
    operations that have no parameters."""

    wires = 2
    state_vector = LightningStateVector(wires)
    state_vector.apply_operations([operation(np.array(par), Wires(range(wires)))])

    state_vector.reset_state()

    expected_output = np.array([1, 0, 0, 0], dtype=state_vector.dtype)
    assert np.allclose(state_vector.state, expected_output, atol=tol, rtol=0)


test_data_no_parameters = [
    (qp.PauliX, [1, 0], [0, 1]),
    (qp.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
    (qp.PauliY, [1, 0], [0, 1j]),
    (qp.PauliY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1j / math.sqrt(2), 1j / math.sqrt(2)]),
    (qp.PauliZ, [1, 0], [1, 0]),
    (qp.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]),
    (qp.S, [1, 0], [1, 0]),
    (qp.S, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1j / math.sqrt(2)]),
    (qp.T, [1, 0], [1, 0]),
    (
        qp.T,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / math.sqrt(2), np.exp(1j * np.pi / 4) / math.sqrt(2)],
    ),
    (qp.Hadamard, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
    (qp.Hadamard, [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1]),
    (qp.Identity, [1, 0], [1, 0]),
    (qp.Identity, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
]


@pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
def test_apply_operation_single_wire_no_parameters(tol, operation, input, expected_output):
    """Tests that applying an operation yields the expected output state for single wire
    operations that have no parameters."""
    wires = 1
    state_vector = LightningStateVector(wires)
    state_vector.apply_operations(
        [qp.StatePrep(np.array(input), Wires(range(wires))), operation(Wires(range(wires)))]
    )

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)
    assert state_vector.state.dtype == state_vector.dtype


test_data_two_wires_no_parameters = [
    (qp.CNOT, [1, 0, 0, 0], [1, 0, 0, 0]),
    (qp.CNOT, [0, 0, 1, 0], [0, 0, 0, 1]),
    (
        qp.CNOT,
        [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
    ),
    (qp.SWAP, [1, 0, 0, 0], [1, 0, 0, 0]),
    (qp.SWAP, [0, 0, 1, 0], [0, 1, 0, 0]),
    (
        qp.SWAP,
        [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
        [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0],
    ),
    (qp.CZ, [1, 0, 0, 0], [1, 0, 0, 0]),
    (qp.CZ, [0, 0, 0, 1], [0, 0, 0, -1]),
    (
        qp.CZ,
        [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)],
        [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
    ),
]


@pytest.mark.parametrize("operation,input,expected_output", test_data_two_wires_no_parameters)
def test_apply_operation_two_wires_no_parameters(tol, operation, input, expected_output):
    """Tests that applying an operation yields the expected output state for two wire
    operations that have no parameters."""
    wires = 2
    state_vector = LightningStateVector(wires)
    state_vector.apply_operations(
        [qp.StatePrep(np.array(input), Wires(range(wires))), operation(Wires(range(wires)))]
    )

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)
    assert state_vector.state.dtype == state_vector.dtype


test_data_three_wires_no_parameters = [
    (qp.CSWAP, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
    (qp.CSWAP, [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]),
    (qp.CSWAP, [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0]),
    (qp.Toffoli, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
    (qp.Toffoli, [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
    (qp.Toffoli, [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]),
    (qp.Toffoli, [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]),
]


@pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_parameters)
def test_apply_operation_three_wires_no_parameters(tol, operation, input, expected_output):
    """Tests that applying an operation yields the expected output state for three wire
    operations that have no parameters."""

    wires = 3
    state_vector = LightningStateVector(wires)
    state_vector.apply_operations(
        [qp.StatePrep(np.array(input), Wires(range(wires))), operation(Wires(range(wires)))]
    )

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)
    assert state_vector.state.dtype == state_vector.dtype


""" operation,input,expected_output,par """
test_data_single_wire_with_parameters = [
    (qp.PhaseShift, [1, 0], [1, 0], [math.pi / 2]),
    (qp.PhaseShift, [0, 1], [0, 1j], [math.pi / 2]),
    (
        qp.PhaseShift,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / math.sqrt(2), 1 / 2 + 1j / 2],
        [math.pi / 4],
    ),
    (qp.RX, [1, 0], [1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [math.pi / 2]),
    (qp.RX, [1, 0], [0, -1j], [math.pi]),
    (
        qp.RX,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / 2 - 1j / 2, 1 / 2 - 1j / 2],
        [math.pi / 2],
    ),
    (qp.RY, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
    (qp.RY, [1, 0], [0, 1], [math.pi]),
    (qp.RY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [math.pi / 2]),
    (qp.RZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
    (qp.RZ, [0, 1], [0, 1j], [math.pi]),
    (
        qp.RZ,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
        [math.pi / 2],
    ),
    (qp.MultiRZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
    (qp.MultiRZ, [0, 1], [0, 1j], [math.pi]),
    (
        qp.MultiRZ,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
        [math.pi / 2],
    ),
    (qp.Rot, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2, 0, 0]),
    (qp.Rot, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
    (
        qp.Rot,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
        [0, 0, math.pi / 2],
    ),
    (
        qp.Rot,
        [1, 0],
        [-1j / math.sqrt(2), -1 / math.sqrt(2)],
        [math.pi / 2, -math.pi / 2, math.pi / 2],
    ),
    (
        qp.Rot,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / 2 + 1j / 2, -1 / 2 + 1j / 2],
        [-math.pi / 2, math.pi, math.pi],
    ),
]


@pytest.mark.parametrize(
    "operation,input,expected_output,par", test_data_single_wire_with_parameters
)
def test_apply_operation_single_wire_with_parameters(tol, operation, input, expected_output, par):
    """Tests that applying an operation yields the expected output state for single wire
    operations that have parameters."""

    wires = 1
    state_vector = LightningStateVector(wires)
    state_vector.apply_operations(
        [qp.StatePrep(np.array(input), Wires(range(wires))), operation(*par, Wires(range(wires)))]
    )

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)
    assert state_vector.state.dtype == state_vector.dtype


""" operation,input,expected_output,par """
test_data_two_wires_with_parameters = [
    (qp.IsingXX, [1, 0, 0, 0], [1 / math.sqrt(2), 0, 0, -1j / math.sqrt(2)], [math.pi / 2]),
    (
        qp.IsingXX,
        [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
        [-0.5j, 0.5, -0.5j, 0.5],
        [math.pi / 2],
    ),
    (qp.IsingXY, [1, 0, 0, 0], [1, 0, 0, 0], [math.pi / 2]),
    (
        qp.IsingXY,
        [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
        [0, 0.5, 0.5j, 1 / math.sqrt(2)],
        [math.pi / 2],
    ),
    (qp.IsingYY, [1, 0, 0, 0], [1 / math.sqrt(2), 0, 0, 1j / math.sqrt(2)], [math.pi / 2]),
    (
        qp.IsingYY,
        [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        [0.5 + 0.5j, 0, 0, 0.5 + 0.5j],
        [math.pi / 2],
    ),
    (qp.IsingZZ, [1, 0, 0, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0, 0, 0], [math.pi / 2]),
    (
        qp.IsingZZ,
        [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        [0.5 - 0.5j, 0, 0, 0.5 - 0.5j],
        [math.pi / 2],
    ),
    (qp.MultiRZ, [1, 0, 0, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0, 0, 0], [math.pi / 2]),
    (
        qp.MultiRZ,
        [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        [0.5 - 0.5j, 0, 0, 0.5 - 0.5j],
        [math.pi / 2],
    ),
    (qp.CRX, [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
    (qp.CRX, [0, 0, 0, 1], [0, 0, -1j, 0], [math.pi]),
    (
        qp.CRX,
        [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
        [0, 1 / math.sqrt(2), 1 / 2, -1j / 2],
        [math.pi / 2],
    ),
    (qp.CRY, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
    (qp.CRY, [0, 0, 0, 1], [0, 0, -1, 0], [math.pi]),
    (
        qp.CRY,
        [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
        [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
        [math.pi / 2],
    ),
    (qp.CRZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2]),
    (qp.CRZ, [0, 0, 0, 1], [0, 0, 0, 1j], [math.pi]),
    (
        qp.CRZ,
        [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
        [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
        [math.pi / 2],
    ),
    (
        qp.CRot,
        [0, 0, 0, 1],
        [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
        [math.pi / 2, 0, 0],
    ),
    (qp.CRot, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
    (
        qp.CRot,
        [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
        [0, 0, 1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
        [0, 0, math.pi / 2],
    ),
    (
        qp.CRot,
        [0, 0, 0, 1],
        [0, 0, 1 / math.sqrt(2), 1j / math.sqrt(2)],
        [math.pi / 2, -math.pi / 2, math.pi / 2],
    ),
    (
        qp.CRot,
        [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
        [0, 1 / math.sqrt(2), 0, -1 / 2 + 1j / 2],
        [-math.pi / 2, math.pi, math.pi],
    ),
    (
        qp.ControlledPhaseShift,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [math.pi / 2],
    ),
    (
        qp.ControlledPhaseShift,
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [math.pi / 2],
    ),
    (
        qp.ControlledPhaseShift,
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [math.pi / 2],
    ),
    (
        qp.ControlledPhaseShift,
        [0, 0, 0, 1],
        [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
        [math.pi / 4],
    ),
    (
        qp.ControlledPhaseShift,
        [1 / math.sqrt(4), 1 / math.sqrt(4), 1 / math.sqrt(4), 1 / math.sqrt(4)],
        [0.5, 0.5, 0.5, 1 / math.sqrt(8) + 1j / math.sqrt(8)],
        [math.pi / 4],
    ),
]


@pytest.mark.parametrize("operation,input,expected_output,par", test_data_two_wires_with_parameters)
def test_apply_operation_two_wires_with_parameters(tol, operation, input, expected_output, par):
    """Tests that applying an operation yields the expected output state for two wire
    operations that have parameters."""
    wires = 2
    state_vector = LightningStateVector(wires)
    state_vector.apply_operations(
        [qp.StatePrep(np.array(input), Wires(range(wires))), operation(*par, Wires(range(wires)))]
    )

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)
    assert state_vector.state.dtype == state_vector.dtype


@pytest.mark.parametrize("operation,input,expected_output,par", test_data_two_wires_with_parameters)
def test_get_final_state(tol, operation, input, expected_output, par):
    """Tests that applying an operation yields the expected output state for two wire
    operations that have parameters."""
    wires = 2
    state_vector = LightningStateVector(wires)
    tape = QuantumScript(
        [qp.StatePrep(np.array(input), Wires(range(wires))), operation(*par, Wires(range(wires)))]
    )
    final_state = state_vector.get_final_state(tape)

    assert np.allclose(final_state.state, np.array(expected_output), atol=tol, rtol=0)
    assert final_state.state.dtype == final_state.dtype
    assert final_state == state_vector


def test_operation_is_sparse_is_false_for_not_supported_devices():
    """_operation_is_sparse returns False if not overridden by the device class."""
    if device_name == "lightning.qubit":
        pytest.skip("Skipping tests for supported devices")

    wires = 2
    state_vector = LightningStateVector(wires)
    assert (
        state_vector._operation_is_sparse(qp.QubitUnitary(sp.sparse.eye(wires), wires=0)) == False
    )


def test_collapse_branch_error():
    """Tests that when collapsing on branch with norm close to 0 gets error."""
    wires = 3
    state_vector = LightningStateVector(wires)

    with pytest.raises(RuntimeError, match="norm close to zero and cannot be normalized"):
        state_vector.state_vector.collapse(0, True)
