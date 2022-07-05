# Copyright 2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the serialization helper functions
"""
import pennylane as qml
import numpy as np
import pennylane_lightning

from pennylane_lightning._serialize import (
    _serialize_observables,
    _serialize_ops,
    _obs_has_kernel,
)
import pytest
from unittest import mock

from pennylane_lightning.lightning_qubit import CPP_BINARY_AVAILABLE

if not CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

from pennylane_lightning.lightning_qubit_ops.adjoint_diff import (
    NamedObsC64,
    NamedObsC128,
    HermitianObsC64,
    HermitianObsC128,
    TensorProdObsC64,
    TensorProdObsC128,
    HamiltonianC64,
    HamiltonianC128,
)


class TestObsHasKernel:
    """Tests for the _obs_has_kernel function"""

    def test_pauli_z(self):
        """Tests if return is true for a PauliZ observable"""
        o = qml.PauliZ(0)
        assert _obs_has_kernel(o)

    def test_tensor_pauli(self):
        """Tests if return is true for a tensor product of Pauli terms"""
        o = qml.PauliZ(0) @ qml.PauliZ(1)
        assert _obs_has_kernel(o)

    def test_hadamard(self):
        """Tests if return is true for a Hadamard observable"""
        o = qml.Hadamard(0)
        assert _obs_has_kernel(o)

    def test_projector(self):
        """Tests if return is true for a Projector observable"""
        o = qml.Projector([0], wires=0)
        assert _obs_has_kernel(o)

    def test_hermitian(self):
        """Tests if return is false for a Hermitian observable"""
        o = qml.Hermitian(np.eye(2), wires=0)
        assert not _obs_has_kernel(o)

    def test_tensor_product_of_valid_terms(self):
        """Tests if return is true for a tensor product of Pauli, Hadamard, and Projector terms"""
        o = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.Projector([0], wires=2)
        assert _obs_has_kernel(o)

    def test_tensor_product_of_invalid_terms(self):
        """Tests if return is false for a tensor product of Hermitian terms"""
        o = qml.Hermitian(np.eye(2), wires=0) @ qml.Hermitian(np.eye(2), wires=1)
        assert not _obs_has_kernel(o)

    def test_tensor_product_of_mixed_terms(self):
        """Tests if return is false for a tensor product of valid and invalid terms"""
        o = qml.PauliZ(0) @ qml.Hermitian(np.eye(2), wires=1)
        assert not _obs_has_kernel(o)


class TestSerializeObs:
    """Tests for the _serialize_obs function"""

    wires_dict = {i: i for i in range(10)}

    @pytest.mark.parametrize("ObsFunc", [NamedObsC128, NamedObsC64])
    def test_basic_return(self, monkeypatch, ObsFunc):
        """Test expected serialization for a simple return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == NamedObsC64 else False
        obs_str = "NamedObsC64" if ObsFunc == NamedObsC64 else "NamedObsC128"

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args[0]
        s_expected = ("PauliZ", [0])
        ObsFunc(*s_expected)

        assert s == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_tensor_return(self, monkeypatch, use_csingle):
        """Test expected serialization for a tensor product return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        mock_obs = mock.MagicMock()

        ObsFunc = TensorProdObsC64 if use_csingle else TensorProdObsC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        obs_str = "TensorProdObsC64" if use_csingle else "TensorProdObsC128"

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args[0]
        s_expected = ([named_obs("PauliZ", [0]), named_obs("PauliZ", [1])],)
        ObsFunc(*s_expected)

        assert s == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_tensor_non_tensor_return(self, use_csingle):
        """Test expected serialization for a mixture of tensor product and non-tensor product
        return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.Hadamard(1))

        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128

        s = _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

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

        s = _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)
        s_expected = hermitian_obs(
            np.array(
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=c_dtype,
            ),
            [0, 1],
        )
        s[0] == s_expected

    @pytest.mark.parametrize("use_csingle", [True, False])
    def test_hermitian_tensor_return(self, use_csingle):
        """Test expected serialization for a Hermitian return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.Hermitian(np.eye(2), wires=[2]))

        c_dtype = np.complex64 if use_csingle else np.complex128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        s = _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

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

        s = _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

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

        obs_str = "HamiltonianC64" if use_csingle else "HamiltonianC128"
        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64
        c_dtype = np.complex64 if use_csingle else np.complex128

        s = _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

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

        obs_str = "HamiltonianC64" if use_csingle else "HamiltonianC128"
        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64
        c_dtype = np.complex64 if use_csingle else np.complex128

        s = _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

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

        obs_str = "HamiltonianC64" if use_csingle else "HamiltonianC128"
        hamiltonian_obs = HamiltonianC64 if use_csingle else HamiltonianC128
        named_obs = NamedObsC64 if use_csingle else NamedObsC128
        hermitian_obs = HermitianObsC64 if use_csingle else HermitianObsC128
        tensor_prod_obs = TensorProdObsC64 if use_csingle else TensorProdObsC128
        r_dtype = np.float32 if use_csingle else np.float64
        c_dtype = np.complex64 if use_csingle else np.complex128

        s = _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

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

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("ObsChunk", list(range(1, 5)))
    def test_chunk_obs(self, monkeypatch, use_csingle, ObsChunk):
        """Test chunking of observable array"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.PauliY(wires=1))
            qml.expval(qml.PauliX(0) @ qml.Hermitian([[0, 1], [1, 0]], wires=3) @ qml.Hadamard(2))
            qml.expval(qml.Hermitian(qml.PauliZ.compute_matrix(), wires=0) @ qml.Identity(1))

        s = _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

        obtained_chunks = pennylane_lightning.lightning_qubit._chunk_iterable(s, ObsChunk)
        assert len(list(obtained_chunks)) == int(np.ceil(len(s) / ObsChunk))


class TestSerializeOps:
    """Tests for the _serialize_ops function"""

    wires_dict = {i: i for i in range(10)}

    def test_basic_circuit(self):
        """Test expected serialization for a simple circuit"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])

        s = _serialize_ops(tape, self.wires_dict)
        s_expected = (
            (
                ["RX", "RY", "CNOT"],
                [np.array([0.4]), np.array([0.6]), []],
                [[0], [1], [0, 1]],
                [False, False, False],
                [[], [], []],
            ),
            False,
        )
        print(s == s_expected)
        assert s == s_expected

    def test_skips_prep_circuit(self):
        """Test expected serialization for a simple circuit with state preparation, such that
        the state preparation is skipped"""
        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector([1, 0], wires=0)
            qml.BasisState([1], wires=1)
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])

        s = _serialize_ops(tape, self.wires_dict)
        s_expected = (
            (
                ["RX", "RY", "CNOT"],
                [[0.4], [0.6], []],
                [[0], [1], [0, 1]],
                [False, False, False],
                [[], [], []],
            ),
            True,
        )
        assert s == s_expected

    def test_inverse_circuit(self):
        """Test expected serialization for a simple circuit that includes an inverse gate"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1).inv()
            qml.CNOT(wires=[0, 1])

        s = _serialize_ops(tape, self.wires_dict)
        s_expected = (
            (
                ["RX", "RY", "CNOT"],
                [[0.4], [0.6], []],
                [[0], [1], [0, 1]],
                [False, True, False],
                [[], [], []],
            ),
            False,
        )
        assert s == s_expected

    def test_unsupported_kernel_circuit(self):
        """Test expected serialization for a circuit including gates that do not have a dedicated
        kernel"""
        with qml.tape.QuantumTape() as tape:
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.2, wires=2)

        s = _serialize_ops(tape, self.wires_dict)
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
            qml.SingleExcitationMinus(0.5, wires=["a", 3.2]).inv()

        s = _serialize_ops(tape, wires_dict)
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
                [[0.4], [0.6], [], [0.5], [0.4], [0.5]],
                [[0], [1], [0, 1], [0, 1], [0, 1], [0, 1]],
                [False, False, False, False, False, True],
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
            qml.RY(0.6, wires=1).inv().inv()
            qml.CNOT(wires=[0, 1])
            qml.QubitUnitary(np.eye(4), wires=[0, 1])
            qml.templates.QFT(wires=[0, 1, 2]).inv()
            qml.DoubleExcitation(0.555, wires=[3, 2, 1, 0])
            qml.DoubleExcitationMinus(0.555, wires=[0, 1, 2, 3])
            qml.DoubleExcitationPlus(0.555, wires=[0, 1, 2, 3])

        s = _serialize_ops(tape, self.wires_dict)

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
                [[0.4], [0.6], [], [], [], [0.555], [0.555], [0.555]],
                [[0], [1], [0, 1], [0, 1], [0, 1, 2], [3, 2, 1, 0], [0, 1, 2, 3], [0, 1, 2, 3]],
                [False, False, False, False, False, False, False, False],
                [
                    [],
                    [],
                    [],
                    qml.matrix(qml.QubitUnitary(np.eye(4, dtype=dtype), wires=[0, 1])),
                    qml.matrix(qml.templates.QFT(wires=[0, 1, 2]).inv()),
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
