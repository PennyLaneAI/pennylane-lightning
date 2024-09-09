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
Unit tests for the serialization helper functions.
"""

import math

import numpy as np
import pennylane as qml
import pytest
from conftest import LightningDevice, LightningStateVector, device_name  # tested device
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

if device_name == "lightning.kokkos":
    try:
        from pennylane_lightning.lightning_kokkos_ops import InitializationSettings
    except ImportError:
        pass
    
if device_name == "lightning.gpu":
    from pennylane_lightning.lightning_gpu._mpi_handler import LightningGPU_MPIHandler

if device_name == "lightning.tensor":
    pytest.skip("Skipping tests for the LightningTensor class.", allow_module_level=True)

if not LightningDevice._new_API:
    pytest.skip(
        "Exclusive tests for new API devices. Skipping.",
        allow_module_level=True,
    )

# if device_name == "lightning.gpu":
#     pytest.skip("LGPU new API in WIP.  Skipping.", allow_module_level=True)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.parametrize("num_wires", range(4))
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_device_name_and_init(num_wires, dtype):
    """Test the class initialization and returned properties."""
    state_vector = LightningStateVector(num_wires, dtype=dtype, device_name=device_name)
    assert state_vector.dtype == dtype
    assert state_vector.device_name == device_name
    assert state_vector.wires == Wires(range(num_wires))

    if device_name == "lightning.kokkos":
        bad_kokkos_args = np.array([33])
        with pytest.raises(
            TypeError,
            match=f"Argument kokkos_args must be of type {type(InitializationSettings())} but it is of {type(bad_kokkos_args)}.",
        ):
            assert LightningStateVector(
                num_wires, dtype=dtype, device_name=device_name, kokkos_args=bad_kokkos_args
            )

        set_kokkos_args = InitializationSettings().set_num_threads(2)
        state_vector_3 = LightningStateVector(
            num_wires, dtype=dtype, device_name=device_name, kokkos_args=set_kokkos_args
        )

        assert type(state_vector) == type(state_vector_3)


def test_wrong_device_name():
    """Test an invalid device name"""
    with pytest.raises(qml.DeviceError, match="The device name"):
        LightningStateVector(3, device_name="thunder.qubit")


@pytest.mark.parametrize("dtype", [np.double])
def test_wrong_dtype(dtype):
    """Test if the class returns a TypeError for a wrong dtype"""
    with pytest.raises(TypeError, match="Unsupported complex type:"):
        assert LightningStateVector(3, dtype=dtype)


def test_errors_basis_state():
    with pytest.raises(ValueError, match="Basis state must only consist of 0s and 1s;"):
        state_vector = LightningStateVector(2)
        state_vector.apply_operations([qml.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])])
    with pytest.raises(ValueError, match="State must be of length 1;"):
        state_vector = LightningStateVector(1)
        state_vector.apply_operations([qml.BasisState(np.array([0, 1]), wires=[0])])


def test_apply_state_vector_with_lightning_handle(tol):
    state_vector_1 = LightningStateVector(2)
    state_vector_1.apply_operations([qml.BasisState(np.array([0, 1]), wires=[0, 1])])

    if device_name == 'lightning.gpu':
        with pytest.raises(qml.DeviceError, match="LightningGPU does not support allocate external state_vector."):
            state_vector_2 = LightningStateVector(2)
            state_vector_2._apply_state_vector(state_vector_1.state_vector, Wires([0, 1]))

    else:
        state_vector_2 = LightningStateVector(2)
        state_vector_2._apply_state_vector(state_vector_1.state_vector, Wires([0, 1]))

        assert np.allclose(state_vector_1.state, state_vector_2.state, atol=tol, rtol=0)


@pytest.mark.parametrize(
    "operation,expected_output,par",
    [
        (qml.BasisState, [0, 0, 1, 0], [1, 0]),
        (qml.BasisState, [0, 0, 0, 1], [1, 1]),
        (qml.QubitStateVector, [0, 0, 1, 0], [0, 0, 1, 0]),
        (qml.QubitStateVector, [0, 0, 0, 1], [0, 0, 0, 1]),
        (qml.StatePrep, [0, 0, 1, 0], [0, 0, 1, 0]),
        (qml.StatePrep, [0, 0, 0, 1], [0, 0, 0, 1]),
        (
            qml.StatePrep,
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
            [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
        ),
        (
            qml.StatePrep,
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
        (qml.BasisState, [1, 0]),
        (qml.QubitStateVector, [0, 0, 1, 0]),
        (
            qml.StatePrep,
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
    (qml.PauliX, [1, 0], [0, 1]),
    (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
    (qml.PauliY, [1, 0], [0, 1j]),
    (qml.PauliY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1j / math.sqrt(2), 1j / math.sqrt(2)]),
    (qml.PauliZ, [1, 0], [1, 0]),
    (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]),
    (qml.S, [1, 0], [1, 0]),
    (qml.S, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1j / math.sqrt(2)]),
    (qml.T, [1, 0], [1, 0]),
    (
        qml.T,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / math.sqrt(2), np.exp(1j * np.pi / 4) / math.sqrt(2)],
    ),
    (qml.Hadamard, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
    (qml.Hadamard, [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1]),
    (qml.Identity, [1, 0], [1, 0]),
    (qml.Identity, [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
]


@pytest.mark.parametrize("operation,input,expected_output", test_data_no_parameters)
def test_apply_operation_single_wire_no_parameters(tol, operation, input, expected_output):
    """Tests that applying an operation yields the expected output state for single wire
    operations that have no parameters."""
    wires = 1
    state_vector = LightningStateVector(wires)
    state_vector.apply_operations(
        [qml.StatePrep(np.array(input), Wires(range(wires))), operation(Wires(range(wires)))]
    )

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)
    assert state_vector.state.dtype == state_vector.dtype


test_data_two_wires_no_parameters = [
    (qml.CNOT, [1, 0, 0, 0], [1, 0, 0, 0]),
    (qml.CNOT, [0, 0, 1, 0], [0, 0, 0, 1]),
    (
        qml.CNOT,
        [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
    ),
    (qml.SWAP, [1, 0, 0, 0], [1, 0, 0, 0]),
    (qml.SWAP, [0, 0, 1, 0], [0, 1, 0, 0]),
    (
        qml.SWAP,
        [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
        [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0],
    ),
    (qml.CZ, [1, 0, 0, 0], [1, 0, 0, 0]),
    (qml.CZ, [0, 0, 0, 1], [0, 0, 0, -1]),
    (
        qml.CZ,
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
        [qml.StatePrep(np.array(input), Wires(range(wires))), operation(Wires(range(wires)))]
    )

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)
    assert state_vector.state.dtype == state_vector.dtype


test_data_three_wires_no_parameters = [
    (qml.CSWAP, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
    (qml.CSWAP, [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]),
    (qml.CSWAP, [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0]),
    (qml.Toffoli, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
    (qml.Toffoli, [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
    (qml.Toffoli, [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]),
    (qml.Toffoli, [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]),
]


@pytest.mark.parametrize("operation,input,expected_output", test_data_three_wires_no_parameters)
def test_apply_operation_three_wires_no_parameters(tol, operation, input, expected_output):
    """Tests that applying an operation yields the expected output state for three wire
    operations that have no parameters."""

    wires = 3
    state_vector = LightningStateVector(wires)
    state_vector.apply_operations(
        [qml.StatePrep(np.array(input), Wires(range(wires))), operation(Wires(range(wires)))]
    )

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)
    assert state_vector.state.dtype == state_vector.dtype


""" operation,input,expected_output,par """
test_data_single_wire_with_parameters = [
    (qml.PhaseShift, [1, 0], [1, 0], [math.pi / 2]),
    (qml.PhaseShift, [0, 1], [0, 1j], [math.pi / 2]),
    (
        qml.PhaseShift,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / math.sqrt(2), 1 / 2 + 1j / 2],
        [math.pi / 4],
    ),
    (qml.RX, [1, 0], [1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [math.pi / 2]),
    (qml.RX, [1, 0], [0, -1j], [math.pi]),
    (
        qml.RX,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / 2 - 1j / 2, 1 / 2 - 1j / 2],
        [math.pi / 2],
    ),
    (qml.RY, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
    (qml.RY, [1, 0], [0, 1], [math.pi]),
    (qml.RY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [math.pi / 2]),
    (qml.RZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
    (qml.RZ, [0, 1], [0, 1j], [math.pi]),
    (
        qml.RZ,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
        [math.pi / 2],
    ),
    (qml.MultiRZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
    (qml.MultiRZ, [0, 1], [0, 1j], [math.pi]),
    (
        qml.MultiRZ,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
        [math.pi / 2],
    ),
    (qml.Rot, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2, 0, 0]),
    (qml.Rot, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
    (
        qml.Rot,
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
        [0, 0, math.pi / 2],
    ),
    (
        qml.Rot,
        [1, 0],
        [-1j / math.sqrt(2), -1 / math.sqrt(2)],
        [math.pi / 2, -math.pi / 2, math.pi / 2],
    ),
    (
        qml.Rot,
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
        [qml.StatePrep(np.array(input), Wires(range(wires))), operation(*par, Wires(range(wires)))]
    )

    assert np.allclose(state_vector.state, np.array(expected_output), atol=tol, rtol=0)
    assert state_vector.state.dtype == state_vector.dtype


""" operation,input,expected_output,par """
test_data_two_wires_with_parameters = [
    (qml.IsingXX, [1, 0, 0, 0], [1 / math.sqrt(2), 0, 0, -1j / math.sqrt(2)], [math.pi / 2]),
    (
        qml.IsingXX,
        [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
        [-0.5j, 0.5, -0.5j, 0.5],
        [math.pi / 2],
    ),
    (qml.IsingXY, [1, 0, 0, 0], [1, 0, 0, 0], [math.pi / 2]),
    (
        qml.IsingXY,
        [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
        [0, 0.5, 0.5j, 1 / math.sqrt(2)],
        [math.pi / 2],
    ),
    (qml.IsingYY, [1, 0, 0, 0], [1 / math.sqrt(2), 0, 0, 1j / math.sqrt(2)], [math.pi / 2]),
    (
        qml.IsingYY,
        [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        [0.5 + 0.5j, 0, 0, 0.5 + 0.5j],
        [math.pi / 2],
    ),
    (qml.IsingZZ, [1, 0, 0, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0, 0, 0], [math.pi / 2]),
    (
        qml.IsingZZ,
        [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        [0.5 - 0.5j, 0, 0, 0.5 - 0.5j],
        [math.pi / 2],
    ),
    (qml.MultiRZ, [1, 0, 0, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0, 0, 0], [math.pi / 2]),
    (
        qml.MultiRZ,
        [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
        [0.5 - 0.5j, 0, 0, 0.5 - 0.5j],
        [math.pi / 2],
    ),
    (qml.CRX, [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
    (qml.CRX, [0, 0, 0, 1], [0, 0, -1j, 0], [math.pi]),
    (
        qml.CRX,
        [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
        [0, 1 / math.sqrt(2), 1 / 2, -1j / 2],
        [math.pi / 2],
    ),
    (qml.CRY, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
    (qml.CRY, [0, 0, 0, 1], [0, 0, -1, 0], [math.pi]),
    (
        qml.CRY,
        [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
        [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
        [math.pi / 2],
    ),
    (qml.CRZ, [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2]),
    (qml.CRZ, [0, 0, 0, 1], [0, 0, 0, 1j], [math.pi]),
    (
        qml.CRZ,
        [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
        [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
        [math.pi / 2],
    ),
    (
        qml.CRot,
        [0, 0, 0, 1],
        [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
        [math.pi / 2, 0, 0],
    ),
    (qml.CRot, [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
    (
        qml.CRot,
        [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
        [0, 0, 1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
        [0, 0, math.pi / 2],
    ),
    (
        qml.CRot,
        [0, 0, 0, 1],
        [0, 0, 1 / math.sqrt(2), 1j / math.sqrt(2)],
        [math.pi / 2, -math.pi / 2, math.pi / 2],
    ),
    (
        qml.CRot,
        [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
        [0, 1 / math.sqrt(2), 0, -1 / 2 + 1j / 2],
        [-math.pi / 2, math.pi, math.pi],
    ),
    (
        qml.ControlledPhaseShift,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [math.pi / 2],
    ),
    (
        qml.ControlledPhaseShift,
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [math.pi / 2],
    ),
    (
        qml.ControlledPhaseShift,
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [math.pi / 2],
    ),
    (
        qml.ControlledPhaseShift,
        [0, 0, 0, 1],
        [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
        [math.pi / 4],
    ),
    (
        qml.ControlledPhaseShift,
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
        [qml.StatePrep(np.array(input), Wires(range(wires))), operation(*par, Wires(range(wires)))]
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
        [qml.StatePrep(np.array(input), Wires(range(wires))), operation(*par, Wires(range(wires)))]
    )
    final_state = state_vector.get_final_state(tape)

    assert np.allclose(final_state.state, np.array(expected_output), atol=tol, rtol=0)
    assert final_state.state.dtype == final_state.dtype
    assert final_state == state_vector
