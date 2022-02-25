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
from pennylane import numpy as np
import pennylane_lightning

from pennylane_lightning._serialize import (
    _serialize_obs,
    _serialize_ops,
    _obs_has_kernel,
    _is_lightning_gate,
)
import pytest
from unittest import mock

try:
    from pennylane_lightning.lightning_qubit_ops import (
        ObsStructC64,
        ObsStructC128,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestIsLightningGate:
    """Tests for the _is_lightning_gate"""

    def test_gates(self):
        """Test if returns true for some gates"""
        for gate in [
            "PauliX",
            "PauliY",
            "PauliZ",
            "Hadamard",
            "S",
            "T",
            "PhaseShift",
            "RX",
            "RY",
            "RZ",
            "Rot",
            "CNOT",
            "CY",
            "CZ",
            "SWAP",
            "ControlledPhaseShift",
            "CRX",
            "CRY",
            "CRZ",
            "CRot",
            "Toffoli",
            "CSWAP",
        ]:
            assert _is_lightning_gate(gate)

    def test_matrix(self):
        assert not _is_lightning_gate("Matrix")

    def test_non_gates(self):
        for gate in ["Quantum", "computing", "in", "2022", "with", "Pennylane", "Lightning"]:
            assert not _is_lightning_gate(gate)


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

    @pytest.mark.skipif(
        "ObsStructC128" and "ObsStructC64" not in dir(pennylane_lightning.lightning_qubit_ops),
        reason="ObsStructC128 and ObsStructC64 are required",
    )
    @pytest.mark.parametrize("ObsFunc", [ObsStructC128, ObsStructC64])
    def test_basic_return(self, monkeypatch, ObsFunc):
        """Test expected serialization for a simple return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0))

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == ObsStructC64 else False
        obs_str = "ObsStructC64" if ObsFunc == ObsStructC64 else "ObsStructC128"

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_obs(tape, self.wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args[0]
        s_expected = (["PauliZ"], [], [[0]])
        ObsFunc(*s_expected)

        assert s == s_expected

    @pytest.mark.skipif(
        "ObsStructC128" and "ObsStructC64" not in dir(pennylane_lightning.lightning_qubit_ops),
        reason="ObsStructC128 and ObsStructC64 are required",
    )
    @pytest.mark.parametrize("ObsFunc", [ObsStructC128, ObsStructC64])
    def test_tensor_return(self, monkeypatch, ObsFunc):
        """Test expected serialization for a tensor product return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == ObsStructC64 else False
        obs_str = "ObsStructC64" if ObsFunc == ObsStructC64 else "ObsStructC128"

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_obs(tape, self.wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args[0]
        s_expected = (["PauliZ", "PauliZ"], [], [[0], [1]])
        ObsFunc(*s_expected)

        assert s == s_expected

    @pytest.mark.skipif(
        "ObsStructC128" and "ObsStructC64" not in dir(pennylane_lightning.lightning_qubit_ops),
        reason="ObsStructC128 and ObsStructC64 are required",
    )
    @pytest.mark.parametrize("ObsFunc", [ObsStructC128, ObsStructC64])
    def test_tensor_non_tensor_return(self, monkeypatch, ObsFunc):
        """Test expected serialization for a mixture of tensor product and non-tensor product
        return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.Hadamard(1))

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == ObsStructC64 else False
        obs_str = "ObsStructC64" if ObsFunc == ObsStructC64 else "ObsStructC128"

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_obs(tape, self.wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args_list

        s_expected = [
            (["PauliZ", "PauliX"], [], [[0], [1]]),
            (["Hadamard"], [], [[1]]),
        ]
        [ObsFunc(*s_expected) for s_expected in s_expected]

        assert s[0][0] == s_expected[0]
        assert s[1][0] == s_expected[1]

    @pytest.mark.skipif(
        "ObsStructC128" and "ObsStructC64" not in dir(pennylane_lightning.lightning_qubit_ops),
        reason="ObsStructC128 and ObsStructC64 are required",
    )
    @pytest.mark.parametrize("ObsFunc", [ObsStructC128, ObsStructC64])
    def test_hermitian_return(self, monkeypatch, ObsFunc):
        """Test expected serialization for a Hermitian return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]))

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == ObsStructC64 else False
        obs_str = "ObsStructC64" if ObsFunc == ObsStructC64 else "ObsStructC128"

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_obs(tape, self.wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args[0]
        s_expected = (["Hermitian"], [np.eye(4).ravel()], [[0, 1]])
        ObsFunc(*s_expected)

        assert s[0] == s_expected[0]
        assert np.allclose(s[1], s_expected[1])
        assert s[2] == s_expected[2]

    @pytest.mark.skipif(
        "ObsStructC128" and "ObsStructC64" not in dir(pennylane_lightning.lightning_qubit_ops),
        reason="ObsStructC128 and ObsStructC64 are required",
    )
    @pytest.mark.parametrize("ObsFunc", [ObsStructC128, ObsStructC64])
    def test_hermitian_tensor_return(self, monkeypatch, ObsFunc):
        """Test expected serialization for a Hermitian return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.Hermitian(np.eye(2), wires=[2]))

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == ObsStructC64 else False
        obs_str = "ObsStructC64" if ObsFunc == ObsStructC64 else "ObsStructC128"

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_obs(tape, self.wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args[0]
        s_expected = (
            ["Hermitian", "Hermitian"],
            [np.eye(4).ravel(), np.eye(2).ravel()],
            [[0, 1], [2]],
        )
        ObsFunc(*s_expected)

        assert s[0] == s_expected[0]
        assert np.allclose(s[1][0], s_expected[1][0])
        assert np.allclose(s[1][1], s_expected[1][1])
        assert s[2] == s_expected[2]

    @pytest.mark.skipif(
        "ObsStructC128" and "ObsStructC64" not in dir(pennylane_lightning.lightning_qubit_ops),
        reason="ObsStructC128 and ObsStructC64 are required",
    )
    @pytest.mark.parametrize("ObsFunc", [ObsStructC128, ObsStructC64])
    def test_mixed_tensor_return(self, monkeypatch, ObsFunc):
        """Test expected serialization for a mixture of Hermitian and Pauli return"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.Hermitian(np.eye(4), wires=[0, 1]) @ qml.PauliY(2))

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == ObsStructC64 else False
        obs_str = "ObsStructC64" if ObsFunc == ObsStructC64 else "ObsStructC128"

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_obs(tape, self.wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args[0]
        s_expected = (["Hermitian", "PauliY"], [np.eye(4).ravel()], [[0, 1], [2]])
        ObsFunc(*s_expected)

        assert s[0] == s_expected[0]
        assert np.allclose(s[1][0], s_expected[1][0])
        assert s[2] == s_expected[2]

    @pytest.mark.skipif(
        "ObsStructC128" and "ObsStructC64" not in dir(pennylane_lightning.lightning_qubit_ops),
        reason="ObsStructC128 and ObsStructC64 are required",
    )
    @pytest.mark.parametrize("ObsFunc", [ObsStructC128, ObsStructC64])
    def test_integration_c64(self, monkeypatch, ObsFunc):
        """Test for a comprehensive range of returns"""
        wires_dict = {"a": 0, 1: 1, "b": 2, -1: 3, 3.141: 4, "five": 5, 6: 6, 77: 7, 9: 8}
        I = np.eye(2).astype(np.complex64)
        X = qml.PauliX.compute_matrix().astype(np.complex64)
        Y = qml.PauliY.compute_matrix().astype(np.complex64)
        Z = qml.PauliZ.compute_matrix().astype(np.complex64)

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == ObsStructC64 else False
        obs_str = "ObsStructC64" if ObsFunc == ObsStructC64 else "ObsStructC128"

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ("a") @ qml.PauliX("b"))
            qml.expval(qml.Hermitian(I, wires=1))
            qml.expval(qml.PauliZ(-1) @ qml.Hermitian(X, wires=3.141) @ qml.Hadamard("five"))
            # qml.expval(qml.Projector([1, 1], wires=[6, 77]) @ qml.Hermitian(Y, wires=9))
            qml.expval(qml.Hermitian(Z, wires="a") @ qml.Identity(1))

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_obs(tape, wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args_list

        s_expected = [
            (["PauliZ", "PauliX"], [], [[0], [2]]),
            (["Hermitian"], [I.ravel()], [[1]]),
            (["PauliZ", "Hermitian", "Hadamard"], [[], X.ravel(), []], [[3], [4], [5]]),
            # (["Projector", "Hermitian"], [[],Y.ravel().astype(np.complex64)], [[6, 7], [8]]),
            (["Hermitian", "Identity"], [Z.ravel(), []], [[0], [1]]),
        ]
        [ObsFunc(*s_expected) for s_expected in s_expected]

        assert all(s1[0][0] == s2[0] for s1, s2 in zip(s, s_expected))
        for s1, s2 in zip(s, s_expected):
            for v1, v2 in zip(s1[0][1], s2[1]):
                assert np.allclose(v1, v2)
        assert all(s1[0][2] == s2[2] for s1, s2 in zip(s, s_expected))

    @pytest.mark.skipif(
        "ObsStructC128" and "ObsStructC64" not in dir(pennylane_lightning.lightning_qubit_ops),
        reason="ObsStructC128 and ObsStructC64 are required",
    )
    @pytest.mark.parametrize("ObsFunc", [ObsStructC128, ObsStructC64])
    def test_integration_c128(self, monkeypatch, ObsFunc):
        """Test for a comprehensive range of returns"""
        wires_dict = {"a": 0, 1: 1, "b": 2, -1: 3, 3.141: 4, "five": 5, 6: 6, 77: 7, 9: 8}
        I = np.eye(2).astype(np.complex128)
        X = qml.PauliX.compute_matrix().astype(np.complex128)
        Y = qml.PauliY.compute_matrix().astype(np.complex128)
        Z = qml.PauliZ.compute_matrix().astype(np.complex128)

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == ObsStructC64 else False
        obs_str = "ObsStructC64" if ObsFunc == ObsStructC64 else "ObsStructC128"

        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ("a") @ qml.PauliX("b"))
            qml.expval(qml.Hermitian(I, wires=1))
            qml.expval(qml.PauliZ(-1) @ qml.Hermitian(X, wires=3.141) @ qml.Hadamard("five"))
            # qml.expval(qml.Projector([1, 1], wires=[6, 77]) @ qml.Hermitian(Y, wires=9))
            qml.expval(qml.Hermitian(Z, wires="a") @ qml.Identity(1))

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_obs(tape, wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args_list

        s_expected = [
            (["PauliZ", "PauliX"], [], [[0], [2]]),
            (["Hermitian"], [I.ravel()], [[1]]),
            (["PauliZ", "Hermitian", "Hadamard"], [[], X.ravel(), []], [[3], [4], [5]]),
            # (["Projector", "Hermitian"], [[],Y.ravel().astype(np.complex128)], [[6, 7], [8]]),
            (["Hermitian", "Identity"], [Z.ravel(), []], [[0], [1]]),
        ]
        [ObsFunc(*s_expected) for s_expected in s_expected]

        assert all(s1[0][0] == s2[0] for s1, s2 in zip(s, s_expected))
        for s1, s2 in zip(s, s_expected):
            for v1, v2 in zip(s1[0][1], s2[1]):
                assert np.allclose(v1, v2)
        assert all(s1[0][2] == s2[2] for s1, s2 in zip(s, s_expected))

    @pytest.mark.skipif(
        "ObsStructC128" and "ObsStructC64" not in dir(pennylane_lightning.lightning_qubit_ops),
        reason="ObsStructC128 and ObsStructC64 are required",
    )
    @pytest.mark.parametrize("ObsFunc", [ObsStructC128, ObsStructC64])
    @pytest.mark.parametrize("ObsChunk", list(range(1, 5)))
    def test_chunk_obs(self, monkeypatch, ObsFunc, ObsChunk):
        """Test chunking of observable array"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.PauliY(wires=1))
            qml.expval(qml.PauliX(0) @ qml.Hermitian([[0, 1], [1, 0]], wires=3) @ qml.Hadamard(2))
            qml.expval(qml.Hermitian(qml.PauliZ.compute_matrix(), wires=1) @ qml.Identity(1))

        mock_obs = mock.MagicMock()

        use_csingle = True if ObsFunc == ObsStructC64 else False
        obs_str = "ObsStructC64" if ObsFunc == ObsStructC64 else "ObsStructC128"

        with monkeypatch.context() as m:
            m.setattr(pennylane_lightning._serialize, obs_str, mock_obs)
            _serialize_obs(tape, self.wires_dict, use_csingle=use_csingle)

        s = mock_obs.call_args_list

        obtained_chunks = pennylane_lightning.lightning_qubit._chunk_iterable(s, ObsChunk)
        assert len(list(obtained_chunks)) == int(np.ceil(len(s) / ObsChunk))


class TestSerializeOps:
    """Tests for the _serialize_ops function"""

    wires_dict = {i: i for i in range(10)}

    @pytest.mark.parametrize("C", [True, False])
    def test_basic_circuit(self, C):
        """Test expected serialization for a simple circuit"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])

        s = _serialize_ops(tape, self.wires_dict, use_csingle=C)
        s_expected = (
            (
                ["RX", "RY", "CNOT"],
                [[0.4], [0.6], []],
                [[0], [1], [0, 1]],
                [False, False, False],
                [[], [], []],
            ),
            False,
        )
        assert s == s_expected

    @pytest.mark.parametrize("C", [True, False])
    def test_skips_prep_circuit(self, C):
        """Test expected serialization for a simple circuit with state preparation, such that
        the state preparation is skipped"""
        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector([1, 0], wires=0)
            qml.BasisState([1], wires=1)
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1)
            qml.CNOT(wires=[0, 1])

        s = _serialize_ops(tape, self.wires_dict, use_csingle=C)
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

    @pytest.mark.parametrize("C", [True, False])
    def test_inverse_circuit(self, C):
        """Test expected serialization for a simple circuit that includes an inverse gate"""
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.6, wires=1).inv()
            qml.CNOT(wires=[0, 1])

        s = _serialize_ops(tape, self.wires_dict, use_csingle=C)
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

    @pytest.mark.parametrize("C", [True, False])
    def test_unsupported_kernel_circuit(self, C):
        """Test expected serialization for a circuit including gates that do not have a dedicated
        kernel"""
        with qml.tape.QuantumTape() as tape:
            qml.SingleExcitationPlus(0.4, wires=[0, 1])
            qml.SingleExcitationMinus(0.5, wires=[1, 2]).inv()
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.2, wires=2)

        s = _serialize_ops(tape, self.wires_dict, use_csingle=C)
        s_expected = (
            (
                ["SingleExcitationPlus", "SingleExcitationMinus", "CNOT", "RZ"],
                [[], [], [], [0.2]],
                [[0, 1], [1, 2], [0, 1], [2]],
                [False, False, False, False],
                [
                    qml.SingleExcitationPlus(0.4, wires=[0, 1]).matrix,
                    qml.SingleExcitationMinus(0.5, wires=[1, 2]).inv().matrix,
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

        assert all(np.allclose(s1, s2) for s1, s2 in zip(s[0][4], s_expected[0][4]))

    @pytest.mark.parametrize("C", [True, False])
    def test_custom_wires_circuit(self, C):
        """Test expected serialization for a simple circuit with custom wire labels"""
        wires_dict = {"a": 0, 3.2: 1}
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires="a")
            qml.RY(0.6, wires=3.2)
            qml.CNOT(wires=["a", 3.2])

        s = _serialize_ops(tape, wires_dict, use_csingle=C)
        s_expected = (
            (
                ["RX", "RY", "CNOT"],
                [[0.4], [0.6], []],
                [[0], [1], [0, 1]],
                [False, False, False],
                [[], [], []],
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

        s = _serialize_ops(tape, self.wires_dict, use_csingle=C)

        dtype = np.complex64 if C else np.complex128
        s_expected = (
            (
                ["RX", "RY", "CNOT", "QubitUnitary", "QFT", "DoubleExcitation"],
                [[0.4], [0.6], [], [], [], []],
                [[0], [1], [0, 1], [0, 1], [0, 1, 2], [3, 2, 1, 0]],
                [False, False, False, False, False, False],
                [
                    [],
                    [],
                    [],
                    qml.QubitUnitary(np.eye(4, dtype=dtype), wires=[0, 1]).matrix,
                    qml.templates.QFT(wires=[0, 1, 2]).inv().matrix,
                    qml.DoubleExcitation(0.555, wires=[3, 2, 1, 0]).matrix,
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
