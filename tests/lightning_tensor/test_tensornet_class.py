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
Unit tests for the tensornet functions.
"""
import numpy as np
import pennylane as qml
import pytest
import scipy
from conftest import (  # tested device
    LightningDevice,
    device_name,
    get_hermitian_matrix,
    get_random_normalized_state,
)
from pennylane.exceptions import DeviceError

if device_name != "lightning.tensor":
    pytest.skip("Skipping tests for the tensornet class.", allow_module_level=True)
else:
    from pennylane_lightning.lightning_tensor._tensornet import (
        LightningTensorNet,
        check_canonical_form,
        decompose_dense,
        expand_mps_first_site,
        gate_matrix_decompose,
        restore_left_canonical_form,
        restore_right_canonical_form,
    )

if not LightningDevice._CPP_BINARY_AVAILABLE:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


@pytest.mark.parametrize("tn_backend", ["mps", "tn"])
@pytest.mark.parametrize("num_wires", range(1, 4))
@pytest.mark.parametrize("bondDims", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("device_name", ["lightning.tensor"])
def test_device_name_and_init(num_wires, bondDims, dtype, device_name, tn_backend):
    """Test the class initialization and returned properties."""
    if num_wires < 2:
        with pytest.raises(ValueError, match="Number of wires must be greater than 1."):
            LightningTensorNet(
                num_wires,
                max_bond_dim=bondDims,
                c_dtype=dtype,
                device_name=device_name,
                method=tn_backend,
            )
        return
    else:
        tensornet = LightningTensorNet(
            num_wires,
            max_bond_dim=bondDims,
            c_dtype=dtype,
            device_name=device_name,
            method=tn_backend,
        )
        assert tensornet.dtype == dtype
        assert tensornet.device_name == device_name
        assert tensornet.num_wires == num_wires
        assert tensornet._method == tn_backend


def test_wrong_device_name():
    """Test an invalid device name"""
    with pytest.raises(DeviceError, match="The device name"):
        LightningTensorNet(3, max_bond_dim=5, device_name="thunder.tensor")


def test_wrong_method_name():
    """Test an invalid method name"""
    with pytest.raises(DeviceError, match="The method "):
        LightningTensorNet(3, max_bond_dim=5, device_name="lightning.tensor", method="spider_web")


@pytest.mark.parametrize("tn_backend", ["mps", "tn"])
def test_errors_basis_state(tn_backend):
    """Test that errors are raised when applying a BasisState operation."""
    with pytest.raises(ValueError, match="Basis state must only consist of 0s and 1s;"):
        tensornet = LightningTensorNet(3, max_bond_dim=5, method=tn_backend)
        tensornet.apply_operations([qml.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])])
    with pytest.raises(ValueError, match="State must be of length 1;"):
        tensornet = LightningTensorNet(3, max_bond_dim=5, method=tn_backend)
        tensornet.apply_operations([qml.BasisState(np.array([0, 1]), wires=[0])])


def test_dense_decompose():
    """Test the dense decomposition function."""
    n_wires = 3
    site_shape = [2, 2]
    max_mpo_bond_dim = 128

    hermitian = get_hermitian_matrix(2**n_wires)

    gate = scipy.linalg.expm(1j * hermitian)
    original_gate = gate.copy()  # for later to double check

    # decompose the gate into MPOs with left canonical form

    mpos = decompose_dense(gate, n_wires, site_shape, max_mpo_bond_dim, canonical_right=False)

    # recreate unitary
    unitary = np.tensordot(mpos[0], mpos[1], axes=([2], [0]))
    unitary = np.tensordot(unitary, mpos[2], axes=([-1], [0]))
    unitary = np.reshape(unitary, (2**n_wires, 2**n_wires))

    assert np.allclose(unitary, original_gate, atol=1e-6)

    # decompose the gate into MPOs with right canonical form

    mpos = decompose_dense(gate, n_wires, site_shape, max_mpo_bond_dim, canonical_right=True)

    # recreate unitary
    unitary = np.tensordot(mpos[0], mpos[1], axes=([2], [0]))
    unitary = np.tensordot(unitary, mpos[2], axes=([-1], [0]))
    unitary = np.reshape(unitary, (2**n_wires, 2**n_wires))

    assert np.allclose(unitary, original_gate, atol=1e-6)


def test_gate_matrix_decompose():
    """Test the gate matrix decomposition function."""
    wires = [0, 1, 2]
    hermitian = get_hermitian_matrix(2 ** len(wires))
    hermitian = hermitian @ hermitian.conj().T

    gate = scipy.linalg.expm(1j * hermitian)
    original_gate = gate.copy()  # for later to double check

    max_mpo_bond_dim = 2 ** len(wires)

    mpos, sorted_wired = gate_matrix_decompose(gate, wires, max_mpo_bond_dim, np.complex128)

    # restore the C-ordering of the matrices
    mpo0 = np.transpose(mpos[0], axes=(2, 1, 0))
    mpo1 = np.transpose(mpos[1], axes=(3, 2, 1, 0))
    mpo2 = np.transpose(mpos[2], axes=(2, 1, 0))

    # recreate unitary
    unitary = np.tensordot(mpo0, mpo1, axes=([1], [0]))
    unitary = np.tensordot(unitary, mpo2, axes=([3], [0]))
    unitary_f = np.transpose(unitary, axes=(5, 3, 1, 4, 2, 0))
    unitary_f = np.reshape(unitary_f, (2 ** len(wires), 2 ** len(wires)))

    assert np.allclose(unitary_f, original_gate, atol=1e-6)


def test_gate_matrix_decompose_out_of_order():
    """Test the gate matrix decomposition function when the wires are not sorted."""
    wires = [1, 2, 0]
    hermitian = get_hermitian_matrix(2 ** len(wires))
    hermitian = hermitian @ hermitian.conj().T

    gate = scipy.linalg.expm(1j * hermitian)
    original_gate = gate.copy()  # for later to double check

    max_mpo_bond_dim = 2 ** len(wires)

    mpos, sorted_wired = gate_matrix_decompose(gate, wires, max_mpo_bond_dim, np.complex128)

    # restore the C-ordering of the matrices
    mpo0 = np.transpose(mpos[0], axes=(2, 1, 0))
    mpo1 = np.transpose(mpos[1], axes=(3, 2, 1, 0))
    mpo2 = np.transpose(mpos[2], axes=(2, 1, 0))

    # check if the wires are the same
    assert sorted_wired == (0, 1, 2)

    # recreate unitary
    unitary = np.tensordot(mpo0, mpo1, axes=([1], [0]))
    unitary = np.tensordot(unitary, mpo2, axes=([3], [0]))
    unitary_f = np.transpose(unitary, axes=(3, 1, 5, 2, 0, 4))
    unitary_f = np.reshape(unitary_f, (2 ** len(wires), 2 ** len(wires)))

    assert np.allclose(unitary_f, original_gate, atol=1e-6)


def test_mps_canonical_form():
    """Test the canonical form functions."""
    n_wires = 3
    site_shape = [2]
    max_mpo_bond_dim = 128

    random_state = get_random_normalized_state(2**n_wires)

    # decompose the gate into MPOs with left canonical form
    mpos = decompose_dense(
        random_state, n_wires, site_shape, max_mpo_bond_dim, canonical_right=False
    )

    # Add virtual bond dimension at the beginning and end
    mpos[0] = np.reshape(mpos[0], [1] + list(mpos[0].shape))
    mpos[-1] = np.reshape(mpos[-1], list(mpos[-1].shape) + [1])

    # check left canonical form
    assert check_canonical_form(mpos, is_right=False)
    # check right canonical form
    assert not check_canonical_form(mpos, is_right=True)

    # decompose the gate into MPOs with left canonical form
    mpos = decompose_dense(
        random_state, n_wires, site_shape, max_mpo_bond_dim, canonical_right=True
    )
    # Add virtual bond dimension at the beginning and end
    mpos[0] = np.reshape(mpos[0], [1] + list(mpos[0].shape))
    mpos[-1] = np.reshape(mpos[-1], list(mpos[-1].shape) + [1])

    # check left canonical form
    assert not check_canonical_form(mpos, is_right=False)
    # check right canonical form
    assert check_canonical_form(mpos, is_right=True)


def test_expand_mps_first_site():
    """Test the expand_mps_first_site function."""
    n_wires = 3
    site_shape = [2]
    max_bond_dim = 128

    random_state = get_random_normalized_state(2**n_wires)

    # decompose the gate into MPOs with right canonical form
    mps = decompose_dense(random_state, n_wires, site_shape, max_bond_dim, canonical_right=True)

    # Add virtual bond dimension at the beginning and end
    mps[0] = np.reshape(mps[0], [1] + list(mps[0].shape))
    mps[-1] = np.reshape(mps[-1], list(mps[-1].shape) + [1])

    # expand the MPS
    new_mps = expand_mps_first_site(mps, max_bond_dim)

    # check length of the MPS
    assert len(new_mps) == n_wires + 1

    # check bond dimensions
    assert new_mps[0].shape == (1, 2, 2)
    assert new_mps[1].shape == (2, 2, 4)
    assert new_mps[2].shape == (4, 2, 2)
    assert new_mps[3].shape == (2, 2, 1)


def test_expand_mps_top_max_bond_dim():
    """Test the expand_mps_top function."""
    n_wires = 6
    site_shape = [2]
    max_bond_dim = 4

    random_state = get_random_normalized_state(2**n_wires)

    # decompse the gate into mps with left canonical form
    mps = decompose_dense(random_state, n_wires, site_shape, max_bond_dim)

    # Add virtual bond dimension at the beginning and end
    mps[0] = np.reshape(mps[0], [1] + list(mps[0].shape))
    mps[-1] = np.reshape(mps[-1], list(mps[-1].shape) + [1])

    # expand the MPS
    mps = expand_mps_first_site(mps, max_bond_dim)

    # check length of the MPS
    assert len(mps) == n_wires + 1

    # check bond dimensions
    assert mps[2].shape == (4, 2, 4)
    assert mps[3].shape == (4, 2, 4)
    assert mps[4].shape == (4, 2, 4)


def test_restore_left_canonical_form():
    """Test the restore_left_canonical_form function."""
    n_wires = 4
    site_shape = [2]
    max_bond_dim = 128

    random_state = get_random_normalized_state(2**n_wires)

    # decompose the gate into mps with right canonical form
    mps = decompose_dense(random_state, n_wires, site_shape, max_bond_dim, canonical_right=True)

    # Add virtual bond dimension at the beginning and end
    mps[0] = np.reshape(mps[0], [1] + list(mps[0].shape))
    mps[-1] = np.reshape(mps[-1], list(mps[-1].shape) + [1])

    # check left canonical form
    assert not check_canonical_form(mps, is_right=False)

    # restore left canonical form
    mps = restore_left_canonical_form(mps, site_shape)

    # check left canonical form
    assert check_canonical_form(mps, is_right=False)
    # check right canonical form
    assert not check_canonical_form(mps, is_right=True)
    # check bond dimensions
    assert mps[0].shape == (1, 2, 2)
    assert mps[1].shape == (2, 2, 4)
    assert mps[2].shape == (4, 2, 2)
    assert mps[3].shape == (2, 2, 1)


def test_restore_right_canonical_form():
    """Test the restore_right_canonical_form function."""
    n_wires = 4
    site_shape = [2]
    max_bond_dim = 128

    random_state = get_random_normalized_state(2**n_wires)

    # decompose the gate into MPS with false canonical form
    mps = decompose_dense(random_state, n_wires, site_shape, max_bond_dim, canonical_right=False)

    # Add virtual bond dimension at the beginning and end
    mps[0] = np.reshape(mps[0], [1] + list(mps[0].shape))
    mps[-1] = np.reshape(mps[-1], list(mps[-1].shape) + [1])

    # check right canonical form
    assert not check_canonical_form(mps, is_right=True)

    # restore right canonical form
    mps = restore_right_canonical_form(mps, site_shape)

    # check right canonical form
    assert check_canonical_form(mps, is_right=True)
    # check left canonical form
    assert not check_canonical_form(mps, is_right=False)

    # check bond dimensions
    assert mps[0].shape == (1, 2, 2)
    assert mps[1].shape == (2, 2, 4)
    assert mps[2].shape == (4, 2, 2)
    assert mps[3].shape == (2, 2, 1)
