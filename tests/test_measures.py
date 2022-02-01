# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Unit tests for Measures in lightning.qubit.
"""
import numpy as np
import pennylane as qml
import pytest


class TestProbs:
    """Test Probs"""

    @pytest.fixture
    def dev(self):
        return qml.device("lightning.qubit", wires=2)

    @pytest.mark.parametrize(
        "cases",
        [
            [None, [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [[], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [[0, 1], [0.9165164490394898, 0.0, 0.08348355096051052, 0.0]],
            [[1, 0], [0.9165164490394898, 0.08348355096051052, 0.0, 0.0]],
            [0, [0.9165164490394898, 0.08348355096051052]],
            [[0], [0.9165164490394898, 0.08348355096051052]],
        ],
    )
    @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    def test_probs_tape_wire0(self, cases, tol, dev, C):
        dev._state = dev._asarray(dev._state, C)

        x, y, z = [0.5, 0.3, -0.7]

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.4, wires=[0])
            qml.Rot(x, y, z, wires=[0])
            qml.RY(-0.2, wires=[0])
            p = qml.probs(cases[0])

        assert np.allclose(cases[1], p, atol=tol, rtol=0)

    # @pytest.mark.parametrize(
    #     "cases",
    #     [
    #         [
    #             None,
    #             [
    #                 0.9178264236525453,
    #                 0.02096485729264079,
    #                 0.059841820910257436,
    #                 0.0013668981445561978,
    #             ],
    #         ],
    #         [
    #             [],
    #             [
    #                 0.9178264236525453,
    #                 0.02096485729264079,
    #                 0.059841820910257436,
    #                 0.0013668981445561978,
    #             ],
    #         ],
    #         [
    #             [0, 1],
    #             [
    #                 0.9178264236525453,
    #                 0.02096485729264079,
    #                 0.059841820910257436,
    #                 0.0013668981445561978,
    #             ],
    #         ],
    #         [
    #             [1, 0],
    #             [
    #                 0.9178264236525453,
    #                 0.059841820910257436,
    #                 0.02096485729264079,
    #                 0.0013668981445561978,
    #             ],
    #         ],
    #         [0, [0.938791280945186, 0.061208719054813635]],
    #         [[0], [0.938791280945186, 0.061208719054813635]],
    #     ],
    # )
    # @pytest.mark.parametrize("C", [np.complex64, np.complex128])
    # def test_probs_tape_wire01(self, cases, tol, dev, C):
    #     dev._state = dev._asarray(dev._state, C)

    #     with qml.tape.JacobianTape() as tape:
    #         qml.RX(0.5, wires=[0])
    #         qml.RY(0.3, wires=[1])

    #     p = dev.probs(cases[0])

    #     assert np.allclose(cases[1], p, atol=tol, rtol=0)


# class TestExpval:
#     """Test Expval"""

#     @pytest.fixture
#     def dev(self):
#         return qml.device("lightning.qubit", wires=2)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0, 1]],
#             ["PauliY", [0, 1]],
#             ["PauliZ", [0, 1]],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_gate_error(self, cases, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         with pytest.raises(
#             ValueError, match="The supplied gate requires 1 wires, but 2 were supplied."
#         ):
#             dev.expval(cases[0], cases[1])

#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_wires_error(self, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         with pytest.raises(AttributeError, match="'str' object has no attribute 'name'"):
#             dev.expval("PauliX", wires=None)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0], -0.041892271271228736],
#             ["PauliX", [1], 0.0],
#             ["PauliY", [0], -0.5516350865364075],
#             ["PauliY", [1], 0.0],
#             ["PauliZ", [0], 0.8330328980789793],
#             ["PauliZ", [1], 1.0],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_tape_wire0(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         x, y, z = [0.5, 0.3, -0.7]

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(x, y, z, wires=[0])
#             qml.RY(-0.2, wires=[0])

#         e = dev.expval(cases[0], cases[1])
#         assert np.allclose(e, cases[2], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0], 0.0],
#             ["PauliX", [1], -0.19866933079506122],
#             ["PauliY", [0], -0.3894183423086505],
#             ["PauliY", [1], 0.0],
#             ["PauliZ", [0], 0.9210609940028852],
#             ["PauliZ", [1], 0.9800665778412417],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_tape_wire01(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         e = dev.expval(cases[0], cases[1])
#         assert np.allclose(e, cases[2], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             [qml.PauliX(0), -0.041892271271228736],
#             [qml.PauliX(1), 0.0],
#             [qml.PauliY(0), -0.5516350865364075],
#             [qml.PauliY(1), 0.0],
#             [qml.PauliZ(0), 0.8330328980789793],
#             [qml.PauliZ(1), 1.0],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_qml_tape_wire0(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         x, y, z = [0.5, 0.3, -0.7]

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(x, y, z, wires=[0])
#             qml.RY(-0.2, wires=[0])

#         e = dev.expval(cases[0])
#         assert np.allclose(e, cases[1], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             [qml.PauliX(0), 0.0],
#             [qml.PauliX(1), -0.19866933079506122],
#             [qml.PauliY(0), -0.3894183423086505],
#             [qml.PauliY(1), 0.0],
#             [qml.PauliZ(0), 0.9210609940028852],
#             [qml.PauliZ(1), 0.9800665778412417],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_expval_qml_tape_wire01(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         e = dev.expval(cases[0])
#         assert np.allclose(e, cases[1], atol=tol, rtol=0)


# class TestVar:
#     """Test Var"""

#     @pytest.fixture
#     def dev(self):
#         return qml.device("lightning.qubit", wires=2)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0, 1]],
#             ["PauliY", [0, 1]],
#             ["PauliZ", [0, 1]],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_gate_error(self, cases, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         with pytest.raises(
#             ValueError, match="The supplied gate requires 1 wires, but 2 were supplied."
#         ):
#             dev.var(cases[0], cases[1])

#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_wires_error(self, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         with pytest.raises(AttributeError, match="'str' object has no attribute 'name'"):
#             dev.var("PauliX", wires=None)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0], 0.9982450376077382],
#             ["PauliX", [1], 1.0],
#             ["PauliY", [0], 0.6956987716741251],
#             ["PauliY", [1], 1.0],
#             ["PauliZ", [0], 0.3060561907181374],
#             ["PauliZ", [1], -4.440892098500626e-16],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_tape_wire0(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         x, y, z = [0.5, 0.3, -0.7]

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(x, y, z, wires=[0])
#             qml.RY(-0.2, wires=[0])

#         e = dev.var(cases[0], cases[1])
#         assert np.allclose(e, cases[2], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             ["PauliX", [0], 1.0],
#             ["PauliX", [1], 0.9605304970014426],
#             ["PauliY", [0], 0.8483533546735826],
#             ["PauliY", [1], 1.0],
#             ["PauliZ", [0], 0.15164664532641725],
#             ["PauliZ", [1], 0.03946950299855745],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_tape_wire01(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         e = dev.var(cases[0], cases[1])
#         assert np.allclose(e, cases[2], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             [qml.PauliX(0), 0.9982450376077382],
#             [qml.PauliX(1), 1.0],
#             [qml.PauliY(0), 0.6956987716741251],
#             [qml.PauliY(1), 1.0],
#             [qml.PauliZ(0), 0.3060561907181374],
#             [qml.PauliZ(1), -4.440892098500626e-16],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_qml_tape_wire0(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         x, y, z = [0.5, 0.3, -0.7]

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.Rot(x, y, z, wires=[0])
#             qml.RY(-0.2, wires=[0])

#         e = dev.var(cases[0])
#         assert np.allclose(e, cases[1], atol=tol, rtol=0)

#     @pytest.mark.parametrize(
#         "cases",
#         [
#             [qml.PauliX(0), 1.0],
#             [qml.PauliX(1), 0.9605304970014426],
#             [qml.PauliY(0), 0.8483533546735826],
#             [qml.PauliY(1), 1.0],
#             [qml.PauliZ(0), 0.15164664532641725],
#             [qml.PauliZ(1), 0.03946950299855745],
#         ],
#     )
#     @pytest.mark.parametrize("C", [np.complex64, np.complex128])
#     def test_var_qml_tape_wire01(self, cases, tol, dev, C):
#         dev._state = dev._asarray(dev._state, C)

#         with qml.tape.JacobianTape() as tape:
#             qml.RX(0.4, wires=[0])
#             qml.RY(-0.2, wires=[1])

#         e = dev.var(cases[0])
#         assert np.allclose(e, cases[1], atol=tol, rtol=0)
