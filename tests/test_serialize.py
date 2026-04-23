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

import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice, compare_serialized_ops, device_name
from pennylane.exceptions import DeviceError

from pennylane_lightning.lightning_base._serialize import (
    QuantumScriptSerializer,
    global_phase_diagonal,
)

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if device_name in ("lightning.kokkos", "lightning.amdgpu"):
    from pennylane_lightning.lightning_kokkos_ops.observables import (
        HamiltonianC64,
        HamiltonianC128,
        HermitianObsC64,
        HermitianObsC128,
        NamedObsC64,
        NamedObsC128,
        SparseHamiltonianC64,
        SparseHamiltonianC128,
        TensorProdObsC64,
        TensorProdObsC128,
    )
elif device_name == "lightning.gpu":
    from pennylane_lightning.lightning_gpu_ops.observables import (
        HamiltonianC64,
        HamiltonianC128,
        HermitianObsC64,
        HermitianObsC128,
        NamedObsC64,
        NamedObsC128,
        SparseHamiltonianC64,
        SparseHamiltonianC128,
        TensorProdObsC64,
        TensorProdObsC128,
    )
elif device_name == "lightning.tensor":
    pytest.skip(
        "Lightning Tensor serialization is tested separately in tests/lightning_tensor/test_serialize_tensor.py",
        allow_module_level=True,
    )
else:
    from pennylane_lightning.lightning_qubit_ops.observables import (
        HamiltonianC64,
        HamiltonianC128,
        HermitianObsC64,
        HermitianObsC128,
        NamedObsC64,
        NamedObsC128,
        SparseHamiltonianC64,
        SparseHamiltonianC128,
        TensorProdObsC64,
        TensorProdObsC128,
    )


def test_wrong_device_name():
    """Test the device name is not a valid option"""

    with pytest.raises(DeviceError, match="The device name"):
        QuantumScriptSerializer("thunder.qubit")


@pytest.mark.parametrize("dtype", ["64", "128"])
@pytest.mark.parametrize(
    "obs,obs_type",
    [
        (qp.PauliZ(0), "NamedObsC"),
        (qp.PauliZ(0) @ qp.PauliZ(1), "TensorProdObsC"),
        (qp.Hadamard(0), "NamedObsC"),
        (qp.Hermitian(np.eye(2), wires=0), "HermitianObsC"),
        (
            (qp.PauliZ(0) @ qp.Hadamard(1) @ (0.1 * (qp.PauliZ(2) + qp.PauliX(3)))),
            "TensorProdObsC",
        ),
        (
            qp.PauliZ(0) @ qp.PauliY(1) @ qp.PauliX(2),
            "TensorProdObsC",
        ),
        (
            qp.PauliZ(0) @ qp.PauliY(1) @ (0.1 * (qp.PauliZ(2) + qp.PauliX(3))),
            "HamiltonianC",
        ),
        (
            (
                qp.Hermitian(np.eye(2), wires=0)
                @ qp.Hermitian(np.eye(2), wires=1)
                @ qp.Projector([0], wires=2)
            ),
            "TensorProdObsC",
        ),
        (
            (
                qp.Hermitian(np.eye(2), wires=0)
                @ qp.Hermitian(np.eye(2), wires=1)
                @ qp.Projector([0], wires=1)
            ),
            "HermitianObsC",
        ),
        (
            qp.PauliZ(0) @ qp.Hermitian(np.eye(2), wires=1) @ qp.Projector([0], wires=2),
            "TensorProdObsC",
        ),
        (qp.Projector([0], wires=0), "HermitianObsC"),
        (qp.Hamiltonian([1], [qp.PauliZ(0)]), "NamedObsC"),
        (qp.sum(qp.Hadamard(0), qp.PauliX(1)), "HamiltonianC"),
        (
            (qp.SparseHamiltonian(qp.Hamiltonian([1], [qp.PauliZ(0)]).sparse_matrix(), wires=[0])),
            "SparseHamiltonianC",
        ),
        (2.5 * qp.PauliZ(0), "HamiltonianC"),
    ],
)
def test_obs_returns_expected_type(dtype, obs, obs_type):
    """Tests that observables get serialized to the expected type, with and without wires map"""
    obs_type_mod = globals().get(obs_type + dtype)

    serializer = QuantumScriptSerializer(device_name, True if dtype == "64" else False)
    assert isinstance(serializer._ob(obs, dict(enumerate(obs.wires))), obs_type_mod)
    assert isinstance(serializer._ob(obs), obs_type_mod)


class TestSerializeObs:
    """Tests for the _observables function"""

    wires_dict = {i: i for i in range(10)}

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_tensor_non_tensor_return(self, use_csingle, wires_map):
        """Test expected serialization for a mixture of tensor product and non-tensor product
        return"""
        with qp.tape.QuantumTape() as tape:
            qp.expval(qp.PauliZ(0) @ qp.PauliX(1))
            qp.expval(qp.Hadamard(1))

        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128

        s_expected = [
            tensor_prod_obs([named_obs("PauliZ", [0]), named_obs("PauliX", [1])]),
            named_obs("Hadamard", [1]),
        ]

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_prod_return_with_overlapping_wires(self, use_csingle, wires_map):
        """Test the expected serialization for a Prod return with operands with overlapping wires."""
        obs = qp.prod(
            qp.sum(qp.X(0), qp.s_prod(2, qp.Hadamard(0))),
            qp.sum(qp.s_prod(3, qp.Z(1)), qp.Z(2), qp.Hermitian(np.eye(2), wires=0)),
        )
        tape = qp.tape.QuantumScript([], [qp.expval(obs)])

        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        c_dtype = np.complex64 if use_csingle else np.complex128
        mat = obs.matrix().ravel().astype(c_dtype)

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        s_expected = hermitian_obs(mat, [0, 1, 2])
        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_hermitian_return(self, use_csingle, wires_map):
        """Test expected serialization for a Hermitian return"""
        with qp.tape.QuantumTape() as tape:
            qp.expval(qp.Hermitian(np.eye(4), wires=[0, 1]))

        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        c_dtype = np.complex64 if use_csingle else np.complex128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        s_expected = hermitian_obs(
            np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=c_dtype,
            ),
            [0, 1],
        )
        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_hermitian_tensor_return(self, use_csingle, wires_map):
        """Test expected serialization for a Hermitian return"""
        with qp.tape.QuantumTape() as tape:
            qp.expval(qp.Hermitian(np.eye(4), wires=[0, 1]) @ qp.Hermitian(np.eye(2), wires=[2]))

        c_dtype = np.complex64 if use_csingle else np.complex128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )

        s_expected = tensor_prod_obs(
            [
                hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1]),
                hermitian_obs(np.eye(2, dtype=c_dtype).ravel(), [2]),
            ]
        )

        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_mixed_tensor_return(self, use_csingle, wires_map):
        """Test expected serialization for a mixture of Hermitian and Pauli return"""
        with qp.tape.QuantumTape() as tape:
            qp.expval(qp.Hermitian(np.eye(4), wires=[0, 1]) @ qp.PauliY(2))

        c_dtype = np.complex64 if use_csingle else np.complex128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )

        s_expected = tensor_prod_obs(
            [
                hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1]),
                named_obs("PauliY", [2]),
            ]
        )

        assert s[0] == s_expected

    @pytest.mark.parametrize(
        "test_hermobs0",
        [qp.Hermitian(np.eye(4), wires=[0, 1])],
    )
    @pytest.mark.parametrize(
        "test_hermobs1",
        [qp.Hermitian(np.ones((8, 8)), wires=range(3))],
    )
    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_hamiltonian_return(self, test_hermobs0, test_hermobs1, use_csingle, wires_map):
        """Test expected serialization for a Hamiltonian return"""

        ham = qp.Hamiltonian(
            [0.3, 0.5, 0.4],
            [
                (test_hermobs0 @ qp.PauliY(2)),
                qp.PauliX(0) @ qp.PauliY(2),
                (test_hermobs1),
            ],
        )

        with qp.tape.QuantumTape() as tape:
            qp.expval(ham)

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64
        c_dtype = np.complex64 if use_csingle else np.complex128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )

        s_expected = hamiltonian_obs(
            np.array([0.3, 0.5, 0.4], dtype=r_dtype),
            [
                tensor_prod_obs(
                    [
                        (hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1])),
                        named_obs("PauliY", [2]),
                    ]
                ),
                tensor_prod_obs([named_obs("PauliX", [0]), named_obs("PauliY", [2])]),
                (hermitian_obs(np.ones(64, dtype=c_dtype), [0, 1, 2])),
            ],
        )

        assert s[0] == s_expected

    @pytest.mark.parametrize(
        "test_hermobs0",
        [qp.Hermitian(np.eye(4), wires=[0, 1])],
    )
    @pytest.mark.parametrize(
        "test_hermobs1",
        [qp.Hermitian(np.ones((8, 8)), wires=range(3))],
    )
    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_hamiltonian_tensor_return(self, test_hermobs0, test_hermobs1, use_csingle, wires_map):
        """Test expected serialization for a tensor Hamiltonian return"""

        with qp.tape.QuantumTape() as tape:
            ham = qp.Hamiltonian(
                [0.3, 0.5, 0.4],
                [
                    (test_hermobs0 @ qp.PauliY(2)),
                    qp.PauliX(0) @ qp.PauliY(2),
                    (test_hermobs1),
                ],
            )
            qp.expval(ham @ qp.PauliZ(3))

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64
        c_dtype = np.complex64 if use_csingle else np.complex128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )

        # Expression (ham @ obs) is converted internally by Pennylane
        # where obs is appended to each term of the ham

        s_expected = hamiltonian_obs(
            np.array([0.3, 0.5, 0.4], dtype=r_dtype),
            [
                tensor_prod_obs(
                    [
                        (hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1])),
                        named_obs("PauliY", [2]),
                        named_obs("PauliZ", [3]),
                    ]
                ),
                tensor_prod_obs(
                    [
                        named_obs("PauliX", [0]),
                        named_obs("PauliY", [2]),
                        named_obs("PauliZ", [3]),
                    ]
                ),
                tensor_prod_obs(
                    [
                        (hermitian_obs(np.ones(64, dtype=c_dtype), [0, 1, 2])),
                        named_obs("PauliZ", [3]),
                    ]
                ),
            ],
        )

        assert s[0] == s_expected

    @pytest.mark.parametrize(
        "test_hermobs0",
        [qp.Hermitian(np.eye(4), wires=[0, 1])],
    )
    @pytest.mark.parametrize(
        "test_hermobs1",
        [qp.Hermitian(np.ones((8, 8)), wires=range(3))],
    )
    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_hamiltonian_mix_return(self, test_hermobs0, test_hermobs1, use_csingle, wires_map):
        """Test expected serialization for a Hamiltonian return"""

        ham1 = qp.Hamiltonian(
            [0.3, 0.5, 0.4],
            [
                (test_hermobs0 @ qp.PauliY(2)),
                qp.PauliX(0) @ qp.PauliY(2),
                (test_hermobs1),
            ],
        )
        ham2 = qp.Hamiltonian(
            [0.7, 0.3],
            [
                (qp.PauliX(0) @ qp.Hermitian(np.eye(4), wires=[1, 2])),
                qp.PauliY(0) @ qp.PauliX(2),
            ],
        )

        with qp.tape.QuantumTape() as tape:
            qp.expval(ham1)
            qp.expval(ham2)

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64
        c_dtype = np.complex64 if use_csingle else np.complex128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        s_expected1 = hamiltonian_obs(
            np.array([0.3, 0.5, 0.4], dtype=r_dtype),
            [
                tensor_prod_obs(
                    [
                        (hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1])),
                        named_obs("PauliY", [2]),
                    ]
                ),
                tensor_prod_obs([named_obs("PauliX", [0]), named_obs("PauliY", [2])]),
                (hermitian_obs(np.ones(64, dtype=c_dtype), [0, 1, 2])),
            ],
        )
        s_expected2 = hamiltonian_obs(
            np.array([0.7, 0.3], dtype=r_dtype),
            [
                tensor_prod_obs(
                    [
                        named_obs("PauliX", [0]),
                        (hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [1, 2])),
                    ]
                ),
                tensor_prod_obs([named_obs("PauliY", [0]), named_obs("PauliX", [2])]),
            ],
        )

        assert s[0] == s_expected1
        assert s[1] == s_expected2

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_pauli_rep_return(self, use_csingle, wires_map):
        """Test that an observable with a valid pauli rep is serialized correctly."""
        with qp.tape.QuantumTape() as tape:
            qp.expval(qp.PauliX(0) + qp.PauliZ(0))

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        r_dtype = np.float32 if use_csingle else np.float64

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        s_expected = hamiltonian_obs(
            np.array([1, 1], dtype=r_dtype), [named_obs("PauliX", [0]), named_obs("PauliZ", [0])]
        )
        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_pauli_rep_single_term(self, use_csingle, wires_map):
        """Test that an observable with a single term in the pauli rep is serialized correctly"""
        with qp.tape.QuantumTape() as tape:
            qp.expval(qp.PauliX(0) @ qp.PauliZ(1))

        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        s_expected = tensor_prod_obs([named_obs("PauliX", [0]), named_obs("PauliZ", [1])])
        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_sprod(self, use_csingle, wires_map):
        """Test that SProds are serialized correctly"""
        tape = qp.tape.QuantumScript([], [qp.expval(qp.s_prod(0.1, qp.Hadamard(0)))])

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        rtype = np.float32 if use_csingle else np.float64

        res, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        assert len(res) == 1
        assert isinstance(res[0], hamiltonian_obs)

        coeffs = np.array([0.1]).astype(rtype)
        s_expected = hamiltonian_obs(coeffs, [named_obs("Hadamard", [0])])
        assert res[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_prod(self, use_csingle, wires_map):
        """Test that Prods are serialized correctly"""
        tape = qp.tape.QuantumScript(
            [], [qp.expval(qp.prod(qp.PauliZ(0), qp.PauliX(1)) @ qp.Hadamard(2))]
        )

        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        tensor_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128

        res, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        assert len(res) == 1
        assert isinstance(res[0], tensor_obs)

        s_expected = tensor_obs(
            [named_obs("PauliZ", [0]), named_obs("PauliX", [1]), named_obs("Hadamard", [2])]
        )
        assert res[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_sum(self, use_csingle, wires_map):
        """Test that Sums are serialized correctly"""
        tape = qp.tape.QuantumScript(
            [],
            [
                qp.expval(
                    qp.sum(
                        0.5 * qp.prod(qp.PauliX(0), qp.PauliZ(1), qp.PauliX(2)),
                        0.1 * qp.prod(qp.PauliZ(0), qp.Hadamard(2), qp.PauliY(1)),
                    )
                )
            ],
        )

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        tensor_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        rtype = np.float32 if use_csingle else np.float64

        res, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        assert len(res) == 1
        assert isinstance(res[0], hamiltonian_obs)

        coeffs = np.array([0.5, 0.1]).astype(rtype)
        s_expected = hamiltonian_obs(
            coeffs,
            [
                tensor_obs(
                    [named_obs("PauliX", [0]), named_obs("PauliZ", [1]), named_obs("PauliX", [2])]
                ),
                tensor_obs(
                    [named_obs("PauliZ", [0]), named_obs("PauliY", [1]), named_obs("Hadamard", [2])]
                ),
            ],
        )
        assert res[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_multi_wire_identity(self, use_csingle, wires_map):
        """Tests that multi-wire Identity does not fail serialization."""
        tape = qp.tape.QuantumTape(measurements=[qp.expval(qp.Identity(wires=[1, 2]))])
        res, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, wires_map
        )
        assert len(res) == 1

        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        assert res[0] == named_obs("Identity", [1])


class TestSerializeOps:
    """Tests for the _ops function"""

    wires_dict = {i: i for i in range(10)}

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_basic_circuit(self, wires_map):
        """Test expected serialization for a simple circuit"""
        with qp.tape.QuantumTape() as tape:
            qp.RX(0.4, wires=0)
            qp.RY(0.6, wires=1)
            qp.CNOT(wires=[0, 1])

        mat = np.array([])

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)
        s_expected = (
            (
                ["RX", "RY", "CNOT"],
                [np.array([0.4]), np.array([0.6]), []],
                [[0], [1], [0, 1]],
                [False, False, False],
                [mat, mat, mat],
                [[], [], []],
                [[], [], []],
            ),
            False,
        )
        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_Rot_in_circuit(self, wires_map):
        """Test expected serialization for a circuit with Rot which should be decomposed"""

        with qp.queuing.AnnotatedQueue() as q:
            qp.Rot(0.1, 0.2, 0.3, wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        mat = np.array([])
        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)
        s_expected = (
            (
                ["RZ", "RY", "RZ"],
                [np.array([0.1]), np.array([0.2]), np.array([0.3])],
                [[0], [0], [0]],
                [False, False, False],
                [mat, mat, mat],
                [[], [], []],
                [[], [], []],
            ),
            False,
        )
        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_basic_circuit_not_implemented_ctrl_ops(self, wires_map):
        """Test expected serialization for circuit with a controlled operation that is not implemented"""
        ops = qp.OrbitalRotation(0.1234, wires=range(4))
        with qp.tape.QuantumTape() as tape:
            qp.RX(0.4, wires=0)
            qp.RY(0.6, wires=1)
            qp.ctrl(ops, [4, 5])

        mat = np.array([])
        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)
        s_expected = (
            (
                ["RX", "RY", "QubitUnitary"],
                [np.array([0.4]), np.array([0.6]), [0.0]],
                [[0], [1], list(ops.wires)],
                [False, False, False],
                [mat, mat, np.array([qp.matrix(ops)])],
                [[], [], [4, 5]],
            ),
            False,
        )
        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_multicontrolledx(self, wires_map):
        """Test expected serialization for a circuit with MultiControlledX"""
        with qp.tape.QuantumTape() as tape:
            qp.RX(0.4, wires=0)
            qp.RY(0.6, wires=1)
            qp.ctrl(qp.PauliX(wires=0), [1, 2, 3], control_values=[True, False, False])

        mat = np.array([])
        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)
        s_expected = (
            (
                ["RX", "RY", "PauliX"],
                [np.array([0.4]), np.array([0.6]), []],
                [[0], [1], [0]],
                [False, False, False],
                [mat, mat, mat],
                [[], [], [1, 2, 3]],
                [[], [], [True, False, False]],
            ),
            False,
        )
        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_skips_prep_circuit(self, wires_map):
        """Test expected serialization for a simple circuit with state preparation, such that
        the state preparation is skipped"""
        with qp.tape.QuantumTape() as tape:
            qp.StatePrep([1, 0], wires=0)
            qp.BasisState([1], wires=1)
            qp.RX(0.4, wires=0)
            qp.RY(0.6, wires=1)
            qp.CNOT(wires=[0, 1])

        mat = np.array([])
        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)
        s_expected = (
            (
                ["RX", "RY", "CNOT"],
                [[0.4], [0.6], []],
                [[0], [1], [0, 1]],
                [False, False, False],
                [mat, mat, mat],
                [[], [], []],
                [[], [], []],
            ),
            True,
        )
        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_unsupported_kernel_circuit(self, wires_map):
        """Test expected serialization for a circuit including gates that do not have a dedicated
        kernel"""
        with qp.tape.QuantumTape() as tape:
            qp.CNOT(wires=[0, 1])
            qp.RZ(0.2, wires=2)

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)
        s_expected = (
            (
                ["CNOT", "RZ"],
                [[], [0.2]],
                [[0, 1], [2]],
                [False, False],
            ),
            False,
        )
        assert compare_serialized_ops(s, s_expected)

    def test_custom_wires_circuit(self):
        """Test expected serialization for a simple circuit with custom wire labels"""
        wires_dict = {"a": 0, 3.2: 1}
        with qp.tape.QuantumTape() as tape:
            qp.RX(0.4, wires="a")
            qp.RY(0.6, wires=3.2)
            qp.CNOT(wires=["a", 3.2])
            qp.SingleExcitation(0.5, wires=["a", 3.2])
            qp.SingleExcitationPlus(0.4, wires=["a", 3.2])
            qp.adjoint(qp.SingleExcitationMinus(0.5, wires=["a", 3.2]), lazy=False)
            qp.adjoint(qp.SingleExcitationMinus(0.5, wires=["a", 3.2]), lazy=True)

        mat = np.array([])
        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_dict)
        s_expected = (
            (
                [
                    "RX",
                    "RY",
                    "CNOT",
                    "SingleExcitation",
                    "SingleExcitationPlus",
                    "SingleExcitationMinus",
                    "SingleExcitationMinus",
                ],
                [[0.4], [0.6], [], [0.5], [0.4], [-0.5], [0.5]],
                [[0], [1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                [False, False, False, False, False, False, True],
                [mat, mat, mat, mat, mat, mat, mat],
                [[], [], [], [], [], [], []],
                [[], [], [], [], [], [], []],
            ),
            False,
        )
        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_ctrl_inverse(self, wires_map):
        """Test expected serialization for nested control adjoint operations"""
        ops = qp.OrbitalRotation(0.1234, wires=range(4))
        with qp.tape.QuantumTape() as tape:
            qp.ctrl(qp.RX(0.4, wires=0), [2, 3])
            qp.ctrl(qp.adjoint(qp.RX(0.4, wires=0), lazy=False), [2, 3])
            qp.ctrl(qp.adjoint(qp.RX(0.4, wires=0), lazy=True), [2, 3])
            qp.adjoint(qp.ctrl(qp.RX(0.4, wires=0), [2, 3]))
            qp.adjoint(qp.ctrl(qp.adjoint(qp.RX(0.4, wires=0)), [2, 3]))
            qp.ctrl(qp.adjoint(ops), [4, 5])
            qp.adjoint(qp.ctrl(ops, [4, 5]))
            qp.adjoint(qp.ctrl(qp.adjoint(ops), [4, 5]))

        mat = np.array([])

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)
        s_expected = (
            (
                ["RX", "RX", "RX", "RX", "RX", "QubitUnitary", "QubitUnitary", "QubitUnitary"],
                [
                    np.array([0.4]),
                    np.array([-0.4]),
                    np.array([0.4]),
                    np.array([0.4]),
                    np.array([0.4]),
                    [0.0],
                    [0.0],
                    [0.0],
                ],
                [[0], [0], [0], [0], [0], list(ops.wires), list(ops.wires), list(ops.wires)],
                [False, False, True, True, False, False, True, True],
                [
                    mat,
                    mat,
                    mat,
                    mat,
                    mat,
                    np.array([qp.matrix(qp.adjoint(ops))]),
                    np.array([qp.matrix(ops)]),
                    np.array([qp.matrix(qp.adjoint(ops))]),
                ],
                [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [4, 5], [4, 5], [4, 5]],
            ),
            False,
        )
        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_ctrl_qubitunitary_inverse(self, wires_map):
        """Test expected serialization for controlled qubit unitary with and without inverse"""
        mat = qp.matrix(qp.RX(0.1234, wires=[0]))
        op = qp.QubitUnitary(mat, wires=[0])
        with qp.tape.QuantumTape() as tape:
            qp.ctrl(op, [4, 5])
            qp.adjoint(qp.ctrl(op, [4, 5]))
            qp.ctrl(qp.adjoint(op, lazy=True), [4, 5])
            qp.adjoint(qp.ctrl(qp.adjoint(op, lazy=True), [4, 5]), lazy=True)

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)
        s_expected = (
            (
                ["QubitUnitary", "QubitUnitary", "QubitUnitary", "QubitUnitary"],
                [[0.0], [0.0], [0.0], [0.0]],
                [list(op.wires), list(op.wires), list(op.wires), list(op.wires)],
                [False, True, False, True],
                [
                    np.array([qp.matrix(op)]),
                    np.array([qp.matrix(op)]),
                    np.array([qp.matrix(qp.adjoint(op))]),
                    np.array([qp.matrix(qp.adjoint(op))]),
                ],
                [[4, 5], [4, 5], [4, 5], [4, 5]],
            ),
            False,
        )

        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    def test_inverse(self, wires_map):
        """Test expected serialization for adjoint gates and qubitunitary"""
        mat = qp.matrix(qp.OrbitalRotation(0.1234, wires=range(4)))
        with qp.tape.QuantumTape() as tape:
            qp.adjoint(qp.SingleExcitationMinus(0.5, wires=[0, 1]), lazy=False)
            qp.adjoint(qp.SingleExcitationMinus(0.5, wires=[0, 1]), lazy=True)
            qp.adjoint(qp.QubitUnitary(mat, wires=range(4)), lazy=True)

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)
        s_expected = (
            (
                [
                    "SingleExcitationMinus",
                    "SingleExcitationMinus",
                    "QubitUnitary",
                ],
                [[-0.5], [0.5], [0.0]],
                [[0, 1], [0, 1], list(range(4))],
                [False, True, True],
                [np.array([]), np.array([]), np.array([mat])],
                [[], [], []],
                [[], [], []],
            ),
            False,
        )

        assert compare_serialized_ops(s, s_expected)

    @pytest.mark.parametrize("wires_map", [wires_dict, None])
    @pytest.mark.parametrize("C", [True, False])
    def test_integration(self, C, wires_map):
        """Test expected serialization for a random circuit"""
        with qp.tape.QuantumTape() as tape:
            qp.RX(0.4, wires=0)
            qp.RY(0.6, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.QubitUnitary(np.eye(4), wires=[0, 1])
            qp.templates.QFT(wires=[0, 1, 2])
            qp.DoubleExcitation(0.555, wires=[3, 2, 1, 0])
            qp.DoubleExcitationMinus(0.555, wires=[0, 1, 2, 3])
            qp.DoubleExcitationPlus(0.555, wires=[0, 1, 2, 3])

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, wires_map)

        dtype = np.complex64 if C else np.complex128
        mat = np.array([], dtype=dtype)
        s_expected = (
            (
                [
                    "RX",
                    "RY",
                    "CNOT",
                    "QubitUnitary",
                    "QFT",
                    "DoubleExcitation",
                    "DoubleExcitationMinus",
                    "DoubleExcitationPlus",
                ],
                [[0.4], [0.6], [], [0.0], [], [0.555], [0.555], [0.555]],
                [[0], [1], [0, 1], [0, 1], [0, 1, 2], [3, 2, 1, 0], [0, 1, 2, 3], [0, 1, 2, 3]],
                [False, False, False, False, False, False, False, False],
                [
                    mat,
                    mat,
                    mat,
                    np.array([qp.matrix(qp.QubitUnitary(np.eye(4, dtype=dtype), wires=[0, 1]))]),
                    np.array([qp.matrix(qp.templates.QFT(wires=[0, 1, 2]))]),
                    mat,
                    mat,
                    mat,
                ],
            ),
            False,
        )
        assert compare_serialized_ops(s, s_expected)


def check_global_phase_diagonal(par, wires, targets, controls, control_values):
    op = qp.ctrl(qp.GlobalPhase(par, wires=targets), controls, control_values=control_values)
    return np.diag(op.matrix(wires))


def test_global_phase():
    """Validate global_phase_diagonal with various combinations of num_qubits, targets and controls."""
    import itertools

    nmax = 7
    par = 0.1
    for nq in range(2, nmax):
        wires = range(nq)
        for nw in range(nq, nmax):
            wire_lists = list(itertools.permutations(wires, nw))
            for wire_list in wire_lists:
                for i in range(len(wire_list) - 1):
                    targets = wire_list[0:i]
                    controls = wire_list[i:]
                    control_values = [i % 2 == 0 for i in controls]
                    D0 = check_global_phase_diagonal(par, wires, targets, controls, control_values)
                    D1 = global_phase_diagonal(par, wires, controls, control_values)
                    assert np.allclose(D0, D1)
