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
Unit tests for the serialization helper functions.
"""
import pytest
from conftest import device_name, LightningDevice

import numpy as np
import pennylane as qml
from pennylane_lightning.core._serialize import QuantumScriptSerializer, global_phase_diagonal

if not LightningDevice._CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if device_name == "lightning.kokkos":
    from pennylane_lightning.lightning_kokkos_ops.observables import (
        NamedObsC64,
        NamedObsC128,
        HermitianObsC64,
        HermitianObsC128,
        TensorProdObsC64,
        TensorProdObsC128,
        HamiltonianC64,
        HamiltonianC128,
        SparseHamiltonianC64,
        SparseHamiltonianC128,
    )
elif device_name == "lightning.gpu":
    from pennylane_lightning.lightning_gpu_ops.observables import (
        NamedObsC64,
        NamedObsC128,
        HermitianObsC64,
        HermitianObsC128,
        TensorProdObsC64,
        TensorProdObsC128,
        HamiltonianC64,
        HamiltonianC128,
        SparseHamiltonianC64,
        SparseHamiltonianC128,
    )
else:
    from pennylane_lightning.lightning_qubit_ops.observables import (
        NamedObsC64,
        NamedObsC128,
        HermitianObsC64,
        HermitianObsC128,
        TensorProdObsC64,
        TensorProdObsC128,
        HamiltonianC64,
        HamiltonianC128,
        SparseHamiltonianC64,
        SparseHamiltonianC128,
    )


def test_wrong_device_name():
    """Test the device name is not a valid option"""

    with pytest.raises(qml.DeviceError, match="The device name"):
        QuantumScriptSerializer("thunder.qubit")


@pytest.mark.parametrize(
    "obs,obs_type",
    [
        (qml.PauliZ(0), NamedObsC128),
        (qml.PauliZ(0) @ qml.PauliZ(1), TensorProdObsC128),
        (qml.Hadamard(0), NamedObsC128),
        (qml.Hermitian(np.eye(2), wires=0), HermitianObsC128),
        (
            qml.PauliZ(0) @ qml.Hadamard(1) @ (0.1 * (qml.PauliZ(2) + qml.PauliX(3))),
            HamiltonianC128,
        ),
        (
            (
                qml.Hermitian(np.eye(2), wires=0)
                @ qml.Hermitian(np.eye(2), wires=1)
                @ qml.Projector([0], wires=2)
            ),
            TensorProdObsC128,
        ),
        (
            qml.PauliZ(0) @ qml.Hermitian(np.eye(2), wires=1) @ qml.Projector([0], wires=2),
            TensorProdObsC128,
        ),
        (qml.Projector([0], wires=0), HermitianObsC128),
        (qml.Hamiltonian([1], [qml.PauliZ(0)]), HamiltonianC128),
        (qml.sum(qml.Hadamard(0), qml.PauliX(1)), HermitianObsC128),
        (
            qml.SparseHamiltonian(qml.Hamiltonian([1], [qml.PauliZ(0)]).sparse_matrix(), wires=[0]),
            SparseHamiltonianC128,
        ),
    ],
)
def test_obs_returns_expected_type(obs, obs_type):
    """Tests that observables get serialized to the expected type."""
    assert isinstance(
        QuantumScriptSerializer(device_name)._ob(obs, dict(enumerate(obs.wires))), obs_type
    )


class TestSerializeObs:
    """Tests for the _observables function"""

    wires_dict = {i: i for i in range(10)}

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_tensor_non_tensor_return(self, use_csingle):
        """Test expected serialization for a mixture of tensor product and non-tensor product
        return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.Hadamard(1))

        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, self.wires_dict
        )

        s_expected = [
            tensor_prod_obs([named_obs("PauliZ", [0]), named_obs("PauliX", [1])]),
            named_obs("Hadamard", [1]),
        ]

        assert s == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_hermitian_return(self, use_csingle):
        """Test expected serialization for a Hermitian return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]))

        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        c_dtype = np.complex64 if use_csingle else np.complex128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, self.wires_dict
        )
        s_expected = hermitian_obs(
            np.array(
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=c_dtype,
            ),
            [0, 1],
        )
        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_hermitian_tensor_return(self, use_csingle):
        """Test expected serialization for a Hermitian return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.Hermitian(np.eye(2), wires=[2]))

        c_dtype = np.complex64 if use_csingle else np.complex128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, self.wires_dict
        )

        s_expected = tensor_prod_obs(
            [
                hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1]),
                hermitian_obs(np.eye(2, dtype=c_dtype).ravel(), [2]),
            ]
        )

        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_mixed_tensor_return(self, use_csingle):
        """Test expected serialization for a mixture of Hermitian and Pauli return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.PauliY(2))

        c_dtype = np.complex64 if use_csingle else np.complex128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, self.wires_dict
        )

        s_expected = tensor_prod_obs(
            [hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1]), named_obs("PauliY", [2])]
        )

        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_hamiltonian_return(self, use_csingle):
        """Test expected serialization for a Hamiltonian return"""

        ham = qml.Hamiltonian(
            [0.3, 0.5, 0.4],
            [
                qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.PauliY(2),
                qml.PauliX(0) @ qml.PauliY(2),
                qml.Hermitian(np.ones((8, 8)), wires=range(3)),
            ],
        )

        with qml.tape.QuantumTape() as tape:
            qml.expval(ham)

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64
        c_dtype = np.complex64 if use_csingle else np.complex128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, self.wires_dict
        )

        s_expected = hamiltonian_obs(
            np.array([0.3, 0.5, 0.4], dtype=r_dtype),
            [
                tensor_prod_obs(
                    [
                        hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1]),
                        named_obs("PauliY", [2]),
                    ]
                ),
                tensor_prod_obs([named_obs("PauliX", [0]), named_obs("PauliY", [2])]),
                hermitian_obs(np.ones(64, dtype=c_dtype), [0, 1, 2]),
            ],
        )

        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_hamiltonian_tensor_return(self, use_csingle):
        """Test expected serialization for a Hamiltonian return"""

        with qml.tape.QuantumTape() as tape:
            ham = qml.Hamiltonian(
                [0.3, 0.5, 0.4],
                [
                    qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.PauliY(2),
                    qml.PauliX(0) @ qml.PauliY(2),
                    qml.Hermitian(np.ones((8, 8)), wires=range(3)),
                ],
            )
            qml.expval(ham @ qml.PauliZ(3))

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64
        c_dtype = np.complex64 if use_csingle else np.complex128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, self.wires_dict
        )

        # Expression (ham @ obs) is converted internally by Pennylane
        # where obs is appended to each term of the ham
        s_expected = hamiltonian_obs(
            np.array([0.3, 0.5, 0.4], dtype=r_dtype),
            [
                tensor_prod_obs(
                    [
                        hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1]),
                        named_obs("PauliY", [2]),
                        named_obs("PauliZ", [3]),
                    ]
                ),
                tensor_prod_obs(
                    [named_obs("PauliX", [0]), named_obs("PauliY", [2]), named_obs("PauliZ", [3])]
                ),
                tensor_prod_obs(
                    [hermitian_obs(np.ones(64, dtype=c_dtype), [0, 1, 2]), named_obs("PauliZ", [3])]
                ),
            ],
        )

        assert s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_hamiltonian_mix_return(self, use_csingle):
        """Test expected serialization for a Hamiltonian return"""

        ham1 = qml.Hamiltonian(
            [0.3, 0.5, 0.4],
            [
                qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.PauliY(2),
                qml.PauliX(0) @ qml.PauliY(2),
                qml.Hermitian(np.ones((8, 8)), wires=range(3)),
            ],
        )
        ham2 = qml.Hamiltonian(
            [0.7, 0.3],
            [qml.PauliX(0) @ qml.Hermitian(np.eye(4), wires=[1, 2]), qml.PauliY(0) @ qml.PauliX(2)],
        )

        with qml.tape.QuantumTape() as tape:
            qml.expval(ham1)
            qml.expval(ham2)

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64
        c_dtype = np.complex64 if use_csingle else np.complex128

        s, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, self.wires_dict
        )

        s_expected1 = hamiltonian_obs(
            np.array([0.3, 0.5, 0.4], dtype=r_dtype),
            [
                tensor_prod_obs(
                    [
                        hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [0, 1]),
                        named_obs("PauliY", [2]),
                    ]
                ),
                tensor_prod_obs([named_obs("PauliX", [0]), named_obs("PauliY", [2])]),
                hermitian_obs(np.ones(64, dtype=c_dtype), [0, 1, 2]),
            ],
        )
        s_expected2 = hamiltonian_obs(
            np.array([0.7, 0.3], dtype=r_dtype),
            [
                tensor_prod_obs(
                    [
                        named_obs("PauliX", [0]),
                        hermitian_obs(np.eye(4, dtype=c_dtype).ravel(), [1, 2]),
                    ]
                ),
                tensor_prod_obs([named_obs("PauliY", [0]), named_obs("PauliX", [2])]),
            ],
        )

        assert s[0] == s_expected1
        assert s[1] == s_expected2

    @pytest.mark.parametrize(
        "obs,coeffs,terms",
        [
            (qml.prod(qml.PauliZ(0), qml.PauliX(1)), [1], [[("PauliX", 1), ("PauliZ", 0)]]),
            (qml.s_prod(0.1, qml.PauliX(0)), [0.1], ("PauliX", 0)),
            (
                qml.sum(
                    0.5 * qml.prod(qml.PauliX(0), qml.PauliZ(1)),
                    0.1 * qml.prod(qml.PauliZ(0), qml.PauliY(1)),
                ),
                [0.5, 0.1],
                [[("PauliZ", 1), ("PauliX", 0)], [("PauliY", 1), ("PauliZ", 0)]],
            ),
        ],
    )
    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_op_arithmetic_uses_hamiltonian(self, use_csingle, obs, coeffs, terms):
        """Tests that an arithmetic obs with a PauliRep serializes as a Hamiltonian."""
        tape = qml.tape.QuantumTape(measurements=[qml.expval(obs)])
        res, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, self.wires_dict
        )
        assert len(res) == 1
        assert isinstance(res[0], HamiltonianC64 if use_csingle else HamiltonianC128)

        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        tensor_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        rtype = np.float32 if use_csingle else np.float64
        term_shape = np.array(terms).shape

        if len(term_shape) == 1:  # just a single pauli op
            expected_terms = [named_obs(terms[0], [terms[1]])]
        elif len(term_shape) == 3:  # list of tensor products
            expected_terms = [
                tensor_obs([named_obs(pauli, [wire]) for pauli, wire in term]) for term in terms
            ]

        coeffs = np.array(coeffs).astype(rtype)
        assert res[0] == hamiltonian_obs(coeffs, expected_terms)

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_multi_wire_identity(self, use_csingle):
        """Tests that multi-wire Identity does not fail serialization."""
        tape = qml.tape.QuantumTape(measurements=[qml.expval(qml.Identity(wires=[1, 2]))])
        res, _ = QuantumScriptSerializer(device_name, use_csingle).serialize_observables(
            tape, self.wires_dict
        )
        assert len(res) == 1

        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        assert res[0] == named_obs("Identity", [1])


class TestSerializeOps:
    """Tests for the _ops function"""

    wires_dict = {i: i for i in range(10)}

    def test_basic_circuit(self):
        """Test expected serialization for a simple circuit"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, self.wires_dict)
        s_expected = (
            (
                ["RX", "RY", "CNOT"],
                [np.array([0.4]), np.array([0.6]), []],
                [[0], [1], [0, 1]],
                [False, False, False],
                [[], [], []],
                [[], [], []],
                [[], [], []],
            ),
            False,
        )
        assert s == s_expected

    def test_basic_circuit_not_implemented_ctrl_ops(self):
        """Test expected serialization for a simple circuit"""
        ops = qml.OrbitalRotation(0.1234, wires=range(4))
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.ctrl(ops, [4, 5])

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, self.wires_dict)
        s_expected = (
            (
                ["RX", "RY", "QubitUnitary"],
                [np.array([0.4]), np.array([0.6]), [0.0]],
                [[0], [1], list(ops.wires)],
                [False, False, False],
                [[], [], [qml.matrix(ops)]],
                [[], [], [4, 5]],
            ),
            False,
        )
        assert s[0][0] == s_expected[0][0]
        assert s[0][1] == s_expected[0][1]
        assert s[0][2] == s_expected[0][2]
        assert s[0][3] == s_expected[0][3]
        assert all(np.allclose(s0, s1) for s0, s1 in zip(s[0][4], s_expected[0][4]))
        assert s[0][5] == s_expected[0][5]
        assert s[1] == s_expected[1]

    def test_multicontrolledx(self):
        """Test expected serialization for a simple circuit"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.ctrl(qml.PauliX(wires=0), [1, 2, 3], control_values=[True, False, False])

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, self.wires_dict)
        s_expected = (
            (
                ["RX", "RY", "PauliX"],
                [np.array([0.4]), np.array([0.6]), []],
                [[0], [1], [0]],
                [False, False, False],
                [[], [], []],
                [[], [], [1, 2, 3]],
                [[], [], [True, False, False]],
            ),
            False,
        )
        assert s == s_expected

    @pytest.mark.parametrize("stateprep", [qml.QubitStateVector, qml.StatePrep])
    def test_skips_prep_circuit(self, stateprep):
        """Test expected serialization for a simple circuit with state preparation, such that
        the state preparation is skipped"""
        with qml.tape.QuantumTape() as tape:
            stateprep([1, 0], wires=0)
            qml.BasisState([1], wires=1)
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, self.wires_dict)
        s_expected = (
            (
                ["RX", "RY", "CNOT"],
                [[0.4], [0.6], []],
                [[0], [1], [0, 1]],
                [False, False, False],
                [[], [], []],
                [[], [], []],
                [[], [], []],
            ),
            True,
        )
        assert s == s_expected

    def test_unsupported_kernel_circuit(self):
        """Test expected serialization for a circuit including gates that do not have a dedicated
        kernel"""
        with qml.tape.QuantumTape() as tape:
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.2, wires=2)

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, self.wires_dict)
        s_expected = (
            (
                ["CNOT", "RZ"],
                [[], [0.2]],
                [[0, 1], [2]],
                [False, False],
            ),
            False,
        )
        assert s[0][0] == s_expected[0][0]
        assert s[0][1] == s_expected[0][1]

    def test_custom_wires_circuit(self):
        """Test expected serialization for a simple circuit with custom wire labels"""
        wires_dict = {"a": 0, 3.2: 1}
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires="a")
            qml.RY(0.6, wires=3.2)
            qml.CNOT(wires=["a", 3.2])
            qml.SingleExcitation(0.5, wires=["a", 3.2])
            qml.SingleExcitationPlus(0.4, wires=["a", 3.2])
            qml.adjoint(qml.SingleExcitationMinus(0.5, wires=["a", 3.2]), lazy=False)

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
                ],
                [[0.4], [0.6], [], [0.5], [0.4], [-0.5]],
                [[0], [1], [0, 1], [0, 1], [0, 1], [0, 1]],
                [False, False, False, False, False, False],
                [[], [], [], [], [], []],
                [[], [], [], [], [], []],
                [[], [], [], [], [], []],
            ),
            False,
        )
        assert s == s_expected

    @pytest.mark.parametrize("C", [True, False])
    def test_integration(self, C):
        """Test expected serialization for a random circuit"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.QubitUnitary(np.eye(4), wires=[0, 1])
            qml.templates.QFT(wires=[0, 1, 2])
            qml.DoubleExcitation(0.555, wires=[3, 2, 1, 0])
            qml.DoubleExcitationMinus(0.555, wires=[0, 1, 2, 3])
            qml.DoubleExcitationPlus(0.555, wires=[0, 1, 2, 3])

        s = QuantumScriptSerializer(device_name).serialize_ops(tape, self.wires_dict)

        dtype = np.complex64 if C else np.complex128
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
                    [],
                    [],
                    [],
                    qml.matrix(qml.QubitUnitary(np.eye(4, dtype=dtype), wires=[0, 1])),
                    qml.matrix(qml.templates.QFT(wires=[0, 1, 2])),
                    [],
                    [],
                    [],
                ],
            ),
            False,
        )
        assert s[0][0] == s_expected[0][0]
        assert s[0][1] == s_expected[0][1]
        assert s[0][2] == s_expected[0][2]
        assert s[0][3] == s_expected[0][3]
        assert s[1] == s_expected[1]

        assert all(np.allclose(s1, s2) for s1, s2 in zip(s[0][4], s_expected[0][4]))


def check_global_phase_diagonal(par, wires, targets, controls, control_values):
    op = qml.ctrl(qml.GlobalPhase(par, wires=targets), controls, control_values=control_values)
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
