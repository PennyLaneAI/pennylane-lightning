# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane_lightning.LightningQubit` device with parameter broadcasting.
"""
# pylint: disable=protected-access,cell-var-from-loop
import math

import numpy as np
import pennylane as qml
import pytest

try:
    from pennylane_lightning.lightning_qubit_ops import (
        Kokkos_info,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)


class TestApply:
    """Tests that operations are applied correctly or that the proper errors are raised."""

    @pytest.mark.parametrize(
        "operation, par",
        [
            (qml.QubitStateVector, np.array([[0, 0, 1, 0], [0, 0, 1, 0]])),
        ],
    )
    def test_apply_operation_state_preparation(self, qubit_device, operation, par):
        """Tests that the statevector cannot be broadcasted during initialization"""

        par = np.array(par)
        dev = qubit_device(wires=2)
        dev.reset()

        with pytest.raises(
            ValueError, match="Lightning doesn't support broadcasted state vector initialization"
        ):
            dev.apply([operation(par, wires=[0, 1])])

    test_data_single_wire_with_single_parameters = [
        (qml.PhaseShift, [1, 0], [[1, 0], [1, 0]], [math.pi / 2, math.pi / 4]),
        (
            qml.PhaseShift,
            [0, 1],
            [[0, 1.0j], [0, 7.0710677e-01 + 0.70710677j]],
            [math.pi / 2, math.pi / 4],
        ),
        (
            qml.RX,
            [1, 0],
            [[1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [0.9238795, -0.38268346j]],
            [math.pi / 2, math.pi / 4],
        ),
        (
            qml.RX,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [
                [-0.70710677j, -0.70710677j],
                [4.9999997e-01 - 0.49999997j, 4.9999997e-01 - 0.49999997j],
                [6.5328145e-01 - 0.27059805j, 6.5328145e-01 - 0.27059805j],
            ],
            [math.pi, math.pi / 2, math.pi / 4],
        ),
        (qml.RY, [1, 0], [[0, 1], [7.0710677e-01, 7.0710677e-01]], [math.pi, math.pi / 2]),
        (
            qml.RY,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [[-0.7071068, 0.7071067], [0, 1], [0.3826834, 0.9238795]],
            [math.pi, math.pi / 2, math.pi / 4],
        ),
        (qml.RZ, [1, 0], [[-1.0j, 0], [7.0710677e-01 - 0.70710677j, 0]], [math.pi, math.pi / 2]),
        (
            qml.RZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [
                [-0.70710677j, 0.70710677j],
                [4.9999997e-01 - 0.49999997j, 4.9999997e-01 + 0.49999997j],
                [6.5328145e-01 - 0.27059805j, 6.5328145e-01 + 0.27059805j],
            ],
            [math.pi, math.pi / 2, math.pi / 4],
        ),
        (
            qml.MultiRZ,
            [1, 0],
            [[-1.0j, 0], [7.0710677e-01 - 0.70710677j, 0]],
            [math.pi, math.pi / 2],
        ),
        (
            qml.MultiRZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [
                [-0.70710677j, 0.70710677j],
                [4.9999997e-01 - 0.49999997j, 4.9999997e-01 + 0.49999997j],
                [6.5328145e-01 - 0.27059805j, 6.5328145e-01 + 0.27059805j],
            ],
            [math.pi, math.pi / 2, math.pi / 4],
        ),
    ]

    @pytest.mark.parametrize(
        "operation, state, expected_output, par", test_data_single_wire_with_single_parameters
    )
    def test_apply_single_param_single_wire_operation(
        self, qubit_device, tol, operation, state, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        dev = qubit_device(wires=1)
        dev._state = np.array(state).astype(dev.C_DTYPE)
        dev.apply([operation(np.array(par), wires=[0])])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev._state.dtype == dev.C_DTYPE

    test_data_single_wire_with_three_parameters = [
        (
            qml.Rot,
            [1, 0],
            [
                [-0.70710677j, 0.70710677],
                [0.56242222 - 0.7329629j, 0.37940952 - 0.04995021j],
                [0.83146960 - 0.55557024j, 0],
            ],
            [math.pi / 2, math.pi / 3, math.pi / 4],
            [math.pi / 2, math.pi / 4, 0],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
        (
            qml.Rot,
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [
                [-0.49999997 - 0.49999997j, 0.49999997 + 0.49999997j],
                [0.23911762 - 0.09904575j, 0.8923991 + 0.3696438j],
                [0.37533027 - 0.07465784j, 0.9061274 + 0.18023996j],
            ],
            [math.pi / 2, 0, 0],
            [math.pi / 2, math.pi / 3, math.pi / 4],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
        (
            qml.Rot,
            [1 / math.sqrt(2), -1 / math.sqrt(2)],
            [
                [0.21850802 - 0.6724985j, 0.21850802 - 0.6724985j],
                [0.8001031 - 0.33141357j, -0.46193975 - 0.1913417j],
                [0.8304546 - 0.16518769j, -0.5218092 - 0.10379431j],
            ],
            [math.pi / 2, 0, 0],
            [math.pi / 5, math.pi / 6, math.pi / 7],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
    ]

    @pytest.mark.parametrize(
        "operation, state, expected_output, phi, theta, omega",
        test_data_single_wire_with_three_parameters,
    )
    def test_apply_three_param_single_wire_operation(
        self, qubit_device, tol, operation, state, expected_output, phi, theta, omega
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        dev = qubit_device(wires=1)
        dev._state = np.array(state).astype(dev.C_DTYPE)
        dev.apply([operation(np.array(phi), np.array(theta), np.array(omega), wires=[0])])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev._state.dtype == dev.C_DTYPE

    test_data_two_wires_with_single_parameter = [
        (
            qml.IsingXX,
            [1, 0, 0, 0],
            [[0.70710677, 0, 0, -0.70710677j], [0.98078525, 0, 0, -0.19509032j]],
            [math.pi / 2, math.pi / 8],
        ),
        (
            qml.IsingXX,
            [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
            [
                [0.0 - 0.49999997j, 0.49999997 + 0.0j, 0.0 - 0.49999997j, 0.49999997 + 0.0j],
                [0.0 - 0.35355338j, 0.6123724 + 0.0j, 0.0 - 0.35355338j, 0.6123724 + 0.0j],
                [0.0 - 0.27059805j, 0.65328145 + 0.0j, 0.0 - 0.27059805j, 0.65328145 + 0.0j],
            ],
            [math.pi / 2, math.pi / 3, math.pi / 4],
        ),
        (
            qml.IsingXY,
            [1, 0, 0, 0],
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            [math.pi / 2, math.pi / 8],
        ),
        (
            qml.IsingXY,
            [0, 1 / math.sqrt(2), 0, 1 / math.sqrt(2)],
            [
                [0, 0.49999997, 0.49999997j, 0.70710677],
                [0, 0.65328145, 0.27059805j, 0.70710677],
                [0, 0.6935199, 0.13794969j, 0.70710677],
            ],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
        (
            qml.IsingYY,
            [1, 0, 0, 0],
            [[0.70710677, 0, 0, 0.70710677j], [0.98078525, 0, 0, 0.19509032j]],
            [math.pi / 2, math.pi / 8],
        ),
        (
            qml.IsingYY,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [
                [0.49999997 + 0.49999997j, 0, 0, 0.49999997 + 0.49999997j],
                [0.65328145 + 0.27059805j, 0, 0, 0.65328145 + 0.27059805j],
                [0.6935199 + 0.13794969j, 0, 0, 0.6935199 + 0.13794969j],
            ],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
        (
            qml.IsingZZ,
            [1, 0, 0, 0],
            [[0.70710677 - 0.70710677j, 0, 0, 0], [0.98078525 - 0.19509032j, 0, 0, 0]],
            [math.pi / 2, math.pi / 8],
        ),
        (
            qml.IsingZZ,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [
                [0.49999997 - 0.49999997j, 0, 0, 0.49999997 - 0.49999997j],
                [0.65328145 - 0.27059805j, 0, 0, 0.65328145 - 0.27059805j],
                [0.6935199 - 0.13794969j, 0, 0, 0.6935199 - 0.13794969j],
            ],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
        (
            qml.MultiRZ,
            [1, 0, 0, 0],
            [[0.70710677 - 0.70710677j, 0, 0, 0], [0.98078525 - 0.19509032j, 0, 0, 0]],
            [math.pi / 2, math.pi / 8],
        ),
        (
            qml.MultiRZ,
            [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            [
                [0.49999997 - 0.49999997j, 0, 0, 0.49999997 - 0.49999997j],
                [0.65328145 - 0.27059805j, 0, 0, 0.65328145 - 0.27059805j],
                [0.6935199 - 0.13794969j, 0, 0, 0.6935199 - 0.13794969j],
            ],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
        (qml.CRX, [0, 1, 0, 0], [[0, 1, 0, 0], [0, 1, 0, 0]], [math.pi / 2, math.pi / 8]),
        (
            qml.CRX,
            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
            [
                [0, 0.70710677, 0.49999997, -0.49999997j],
                [0, 0.70710677, 0.65328145, -0.27059805j],
                [0, 0.70710677, 0.6935199, -0.13794969j],
            ],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
        (
            qml.CRY,
            [0, 0, 0, 1],
            [[0, 0, -0.70710678, 0.70710678], [0, 0, -0.19509032, 0.98078528]],
            [math.pi / 2, math.pi / 8],
        ),
        (
            qml.CRY,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [
                [0.70710678, 0.70710678, 0, 0],
                [0.70710678, 0.70710678, 0, 0],
                [0.70710678, 0.70710678, 0, 0],
            ],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
        (
            qml.CRZ,
            [0, 0, 0, 1],
            [[0, 0, 0, 0.70710678 + 0.70710678j], [0, 0, 0, 0.98078528 + 0.19509032j]],
            [math.pi / 2, math.pi / 8],
        ),
        (
            qml.CRZ,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
            [
                [0.70710678, 0.70710678, 0, 0],
                [0.70710678, 0.70710678, 0, 0],
                [0.70710678, 0.70710678, 0, 0],
            ],
            [math.pi / 2, math.pi / 4, math.pi / 8],
        ),
        (
            qml.ControlledPhaseShift,
            [1, 0, 0, 0],
            [[1, 0, 0, 0], [1, 0, 0, 0]],
            [math.pi / 2, math.pi / 8],
        ),
        (
            qml.ControlledPhaseShift,
            [1 / math.sqrt(2), 1 / math.sqrt(2), 1 / math.sqrt(2), 1 / math.sqrt(2)],
            [
                [0.707106781, 0.707106781, 0.707106781, -0.707106781],
                [0.707106781, 0.707106781, 0.707106781, 0.707106781j],
                [0.707106781, 0.707106781, 0.707106781, 0.500000000 + 0.500000000j],
            ],
            [math.pi, math.pi / 2, math.pi / 4],
        ),
    ]

    @pytest.mark.parametrize(
        "operation, state, expected_output, par", test_data_two_wires_with_single_parameter
    )
    def test_apply_operation_two_wires_with_parameters(
        self, qubit_device, tol, operation, state, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        dev = qubit_device(wires=2)
        dev._state = np.array(state).reshape(2 * [2]).astype(dev.C_DTYPE)
        dev.apply([operation(np.array(par), wires=[0, 1])])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev._state.dtype == dev.C_DTYPE

    test_data_two_wires_with_three_parameter = [
        (
            qml.CRot,
            [0, 0, 0, 1],
            [
                [0, 0, -0.6532815 - 0.27059805j, 0.27059805 + 0.6532815j],
                [0, 0, -0.3743209 - 0.07956436j, 0.61819607 + 0.6865763j],
                [0, 0, -0.1934213 - 0.0254644j, 0.77810925 + 0.59706426j],
            ],
            [math.pi / 2, math.pi / 3, math.pi / 4],
            [math.pi / 2, math.pi / 4, math.pi / 8],
            [math.pi / 4, math.pi / 5, math.pi / 6],
        ),
        (
            qml.CRot,
            [0, 0, 1 / math.sqrt(2), 1j / math.sqrt(2)],
            [
                [0, 0, 3.82683432e-01 - 9.23879533e-01j, 0],
                [0, 0, 4.93391133e-01 - 7.50167587e-01j, -2.20797920e-01 + 3.80870136e-01j],
                [0, 0, 5.68212395e-01 - 5.58957690e-01j, -2.85418668e-01 + 5.32200299e-01j],
            ],
            [math.pi / 2, math.pi / 3, math.pi / 4],
            [math.pi / 2, math.pi / 4, math.pi / 8],
            [math.pi / 4, math.pi / 5, math.pi / 6],
        ),
        (
            qml.CRot,
            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
            [
                [0, 0.70710678, 0.19134172 - 0.46193977j, 0.46193977 - 0.19134172j],
                [0, 0.70710678, 0.43713063 - 0.48548275j, 0.26468483 - 0.0562605j],
                [0, 0.70710678, 0.55020635 - 0.42218818j, 0.13676951 - 0.01800605j],
            ],
            [math.pi / 2, math.pi / 3, math.pi / 4],
            [math.pi / 2, math.pi / 4, math.pi / 8],
            [math.pi / 4, math.pi / 5, math.pi / 6],
        ),
    ]

    @pytest.mark.parametrize(
        "operation, state, expected_output, phi, theta, omega",
        test_data_two_wires_with_three_parameter,
    )
    def test_apply_operation_two_wires_with_parameters(
        self, qubit_device, tol, operation, state, expected_output, phi, theta, omega
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        dev = qubit_device(wires=2)
        dev._state = np.array(state).reshape(2 * [2]).astype(dev.C_DTYPE)
        dev.apply([operation(np.array(phi), np.array(theta), np.array(omega), wires=[0, 1])])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev._state.dtype == dev.C_DTYPE

    test_data_unitaries = [
        (
            qml.DiagonalQubitUnitary,
            [0.70710677, -0.70710677],
            [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0j], [1.0, -1.0j], [1.0j, 1.0], [1.0j, -1.0]],
            [0],
            [
                [0.70710677, -0.70710677],
                [0.70710677, 0.70710677],
                [0.70710677, -0.70710677j],
                [0.70710677, 0.70710677j],
                [0.70710677j, -0.70710677],
                [0.70710677j, 0.70710677],
            ],
        ),
        (
            qml.DiagonalQubitUnitary,
            [0.5, -0.5, 0.5, -0.5],
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, -1.0],
                [1.0, 1.0j, 1.0, 1.0],
                [1.0, 1.0, 1.0, -1.0j],
                [1.0j, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0j, -1.0],
            ],
            [0, 1],
            [
                [0.5, -0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5, 0.5],
                [0.5, -0.5j, 0.5, -0.5],
                [0.5, -0.5, 0.5, 0.5j],
                [0.5j, -0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5j, 0.5],
            ],
        ),
        (
            qml.QubitUnitary,
            [0.70710677, -0.70710677],
            [np.diag([1.0, 1.0]), np.diag([1.0, -1.0]), np.diag([1.0, 1.0j]), np.diag([1.0, -1.0j]), np.diag([1.0j, 1.0]), np.diag([1.0j, -1.0])],
            [0],
            [
                [0.70710677, -0.70710677],
                [0.70710677, 0.70710677],
                [0.70710677, -0.70710677j],
                [0.70710677, 0.70710677j],
                [0.70710677j, -0.70710677],
                [0.70710677j, 0.70710677],
            ],
        ),
        (
            qml.QubitUnitary,
            [0.5, -0.5, 0.5, -0.5],
            [
                np.diag([1.0, 1.0, 1.0, 1.0]),
                np.diag([1.0, 1.0, 1.0, -1.0]),
                np.diag([1.0, 1.0j, 1.0, 1.0]),
                np.diag([1.0, 1.0, 1.0, -1.0j]),
                np.diag([1.0j, 1.0, 1.0, 1.0]),
                np.diag([1.0, 1.0, 1.0j, -1.0]),
            ],
            [0, 1],
            [
                [0.5, -0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5, 0.5],
                [0.5, -0.5j, 0.5, -0.5],
                [0.5, -0.5, 0.5, 0.5j],
                [0.5j, -0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5j, 0.5],
            ],
        ),
    ]

    @pytest.mark.parametrize(
        "operation, state, param, wires, expected_output",
        test_data_unitaries,
    )
    def test_apply_unitaries(
        self, qubit_device, tol, operation, state, param, wires, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        dev = qubit_device(wires=len(wires))
        dev._state = np.array(state).astype(dev.C_DTYPE)
        dev.apply([operation(np.array(param), wires=wires)])

        assert np.allclose(dev.state, np.array(expected_output), atol=tol, rtol=0)
        assert dev._state.dtype == dev.C_DTYPE


class TestProbs:
    """Tests probabilities with parameter broadcasting in Lightning"""

    @pytest.mark.parametrize(
        "wires, param, expected",
        [
            [
                [1, 0],
                [0.2, 0.3],
                [[0.89179386, 0.10820614, 0.0, 0.0], [0.87556968, 0.12443032, 0.0, 0.0]],
            ],
            [
                0,
                [0.2, 0.3, 0.4],
                [[0.89179386, 0.10820614], [0.87556968, 0.12443032], [0.85559294, 0.14440706]],
            ],
            [
                [0],
                [0.2, 0.3, 0.4, 0.5],
                [
                    [0.89179386, 0.10820614],
                    [0.87556968, 0.12443032],
                    [0.85559294, 0.14440706],
                    [0.83206322, 0.16793678],
                ],
            ],
        ],
    )
    def test_probs_broadcast_one_operation(self, qubit_device, wires, param, expected, tol):
        """Tests the broadcasted probability for a circuit where some operations have non-broadcasted parameters and
        one operation has broadcasted parameters. In all cases, operations have wires=[0]"""

        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(0.5, 0.3, -0.7, wires=[0])
            qml.RY(np.array(param), wires=[0])
            return qml.probs(wires=wires)

        assert np.allclose(circuit(), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "wires, param, expected",
        [
            [
                [0, 1],
                [0.2, 0.3],
                [
                    [9.44563983e-01, 9.50896947e-03, 4.54693055e-02, 4.57741609e-04],
                    [9.14510154e-01, 2.08891076e-02, 6.31580901e-02, 1.44264788e-03],
                ],
            ],
            [
                [1, 0],
                [0.2, 0.3, 0.4],
                [
                    [9.44563983e-01, 4.54693055e-02, 9.50896947e-03, 4.57741609e-04],
                    [9.14510154e-01, 6.31580901e-02, 2.08891076e-02, 1.44264788e-03],
                    [8.76364210e-01, 8.41662873e-02, 3.60109959e-02, 3.45850709e-03],
                ],
            ],
            [
                [0],
                [0.2, 0.3, 0.4, 0.5],
                [
                    [0.95407295, 0.04592705],
                    [0.93539926, 0.06460074],
                    [0.91237521, 0.08762479],
                    [0.88523083, 0.11476917],
                ],
            ],
        ],
    )
    def test_probs_broadcast_two_operations(self, qubit_device, wires, param, expected, tol):
        """Tests the broadcasted probability for a circuit where a operation has non-broadcasted parameters,
        and some operations have broadcasted parameters.
        Broadcasted operations are applied to different wires."""

        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.array(param), wires=[0])
            qml.Rot(0.5, 0.3, -0.7, wires=[0])
            qml.RY(np.array(param), wires=[1])
            return qml.probs(wires=wires)

        assert np.allclose(circuit(), np.array(expected), atol=tol, rtol=0)


class TestExpval:
    """Tests expval with parameter broadcasting in Lightning"""

    @pytest.mark.parametrize(
        "op, param, expected",
        [
            [qml.PauliX(0), [-0.1, -0.2], [0.04148154, -0.04189227]],
            [qml.PauliX(1), [-0.1, -0.2, 0.3], [0.0, 0.0, 0.0]],
            [
                qml.PauliY(0),
                [-0.1, -0.2, 0.3, 0.4],
                [-0.55163505, -0.55163505, -0.55163505, -0.55163505],
            ],
            [qml.PauliY(1), [-0.1, -0.2, 0.3, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0, 0.0]],
            [
                qml.PauliZ(0),
                [-0.1, -0.2, 0.3, 0.4, 0.5, 0.6],
                [0.83305345, 0.8330329, 0.75113937, 0.71118587, 0.66412645, 0.61043129],
            ],
            [
                qml.PauliZ(1),
                [-0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ],
    )
    def test_expval_broadcast_one_operation(self, qubit_device, op, param, expected, tol):
        """Tests the broadcasted expval for a circuit where some operations have non-broadcasted parameters,
        and one operation has broadcasted parameters. In all cases, operations have wires=[0]"""

        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(0.5, 0.3, -0.7, wires=[0])
            qml.RY(np.array(param), wires=[0])
            return qml.expval(op)

        assert np.allclose(circuit(), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op, param, expected",
        [
            [qml.PauliX(0), [-0.1, -0.2], [0.24636598, 0.26424404]],
            [qml.PauliX(1), [-0.1, -0.2, 0.3], [-0.09983342, -0.19866933, 0.29552021]],
            [
                qml.PauliY(0),
                [-0.1, -0.2, 0.3, 0.4],
                [-0.092962, 0.00538419, -0.46742925, -0.55163505],
            ],
            [qml.PauliY(1), [-0.1, -0.2, 0.3, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0, 0.0]],
            [
                qml.PauliZ(0),
                [-0.1, -0.2, 0.3, 0.4, 0.5, 0.6],
                [0.96470818, 0.96444082, 0.87079852, 0.82475041, 0.77046166, 0.70847472],
            ],
            [
                qml.PauliZ(1),
                [-0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [
                    0.99500417,
                    0.98006658,
                    0.95533649,
                    0.92106099,
                    0.87758256,
                    0.82533561,
                    0.76484219,
                ],
            ],
        ],
    )
    def test_expval_broadcast_two_operations(self, qubit_device, op, param, expected, tol):
        """Tests the broadcasted expval for a circuit where a operation has non-broadcasted parameters,
        and some operations have broadcasted parameters.
        Broadcasted operations are applied to different wires."""
        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.array(param), wires=[0])
            qml.Rot(0.5, 0.3, -0.7, wires=[0])
            qml.RY(np.array(param), wires=[1])
            return qml.expval(op)

        assert np.allclose(circuit(), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "obs, coeffs, expected",
        [
            ([qml.PauliX(0) @ qml.PauliZ(1)], [1.0], [0.0, 0.0]),
            ([qml.PauliZ(0) @ qml.PauliZ(1)], [1.0], [0.90270109, 0.96053064]),
            (
                [
                    qml.PauliX(0) @ qml.PauliZ(1),
                    qml.Hermitian(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 1.0],
                            [0.0, 0.0, 1.0, -2.0],
                        ],
                        wires=[0, 1],
                    ),
                ],
                [0.3, 1.0],
                [0.93197289, 0.99772200],
            ),
        ],
    )
    def test_expval_broadcast_Hamiltonian(self, qubit_device, obs, coeffs, expected, tol):
        """Tests parameter broadcasting for a circuit returning the expval of a Hamiltonian."""
        ham = qml.Hamiltonian(np.array(coeffs), np.array(obs))
        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.array([0.4, 0.2]), wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(ham)

        assert np.allclose(circuit(), expected, atol=tol, rtol=0)


class TestSparseExpval:
    """Tests for the expval function"""

    @pytest.mark.parametrize(
        "obs, expected",
        [
            [qml.PauliX(0) @ qml.Identity(1), [0, 0]],
            [qml.Identity(0) @ qml.PauliX(1), [-0.19866933, -0.38941833]],
            [qml.PauliY(0) @ qml.Identity(1), [-0.38941836, -0.38941836]],
            [qml.Identity(0) @ qml.PauliY(1), [0, 0]],
            [qml.PauliZ(0) @ qml.Identity(1), [0.921060994, 0.92106104]],
            [qml.Identity(0) @ qml.PauliZ(1), [0.980066577, 0.92106104]],
        ],
    )
    @pytest.mark.skipif(
        Kokkos_info()["USE_KOKKOS"] == False, reason="Requires Kokkos and Kokkos Kernels."
    )
    def test_sparse_Pauli_words(self, qubit_device, obs, expected, tol):
        """Test expval of some simple sparse Hamiltonian"""
        dev = qubit_device(wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(np.array([-0.2, -0.4]), wires=[1])
            return qml.expval(
                qml.SparseHamiltonian(
                    qml.utils.sparse_hamiltonian(qml.Hamiltonian([1], [obs])), wires=[0, 1]
                )
            )

        assert np.allclose(circuit(), expected, atol=tol, rtol=0)


class TestVar:
    """Tests var with parameter broadcasting in Lightning"""

    @pytest.mark.parametrize(
        "op, param, expected",
        [
            [qml.PauliX(0), [-0.1, -0.2], [0.99827928, 0.99824504]],
            [qml.PauliX(1), [-0.1, -0.2, 0.3], [1.0, 1.0, 1.0]],
            [
                qml.PauliY(0),
                [-0.1, -0.2, 0.3, 0.4],
                [0.69569877, 0.69569877, 0.69569877, 0.69569877],
            ],
            [qml.PauliY(1), [-0.1, -0.2, 0.3, 0.4, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]],
            [
                qml.PauliZ(0),
                [-0.1, -0.2, 0.3, 0.4, 0.5, 0.6],
                [0.30602195, 0.30605619, 0.43578965, 0.49421465, 0.55893606, 0.62737365],
            ],
            [
                qml.PauliZ(1),
                [-0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ],
    )
    def test_var_broadcast_one_operation(self, qubit_device, op, param, expected, tol):
        """Tests the broadcasted var for a circuit where some operations have non-broadcasted parameters,
        and one operation has broadcasted parameters. In all cases, operations have wires=[0]"""

        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.Rot(0.5, 0.3, -0.7, wires=[0])
            qml.RY(np.array(param), wires=[0])
            return qml.var(op)

        assert np.allclose(circuit(), expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op, param, expected",
        [
            [qml.PauliX(0), [-0.1, -0.2], [0.9393038, 0.93017509]],
            [qml.PauliX(1), [-0.1, -0.2, 0.3], [0.99003329, 0.9605305, 0.91266781]],
            [
                qml.PauliY(0),
                [-0.1, -0.2, 0.3, 0.4],
                [0.99135807, 0.99997101, 0.78150989, 0.69569877],
            ],
            [qml.PauliY(1), [-0.1, -0.2, 0.3, 0.4, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]],
            [
                qml.PauliZ(0),
                [-0.1, -0.2, 0.3, 0.4, 0.5, 0.6],
                [0.06933813, 0.0698539, 0.24170993, 0.31978676, 0.40638882, 0.49806357],
            ],
            [
                qml.PauliZ(1),
                [-0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.00996671, 0.0394695, 0.08733219, 0.15164665, 0.22984885, 0.31882112, 0.41501643],
            ],
        ],
    )
    def test_var_broadcast_two_operations(self, qubit_device, op, param, expected, tol):
        """Tests the broadcasted var for a circuit where a operation has non-broadcasted parameters,
        and some operations have broadcasted parameters.
        Broadcasted operations are applied to different wires."""
        dev = qubit_device(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.array(param), wires=[0])
            qml.Rot(0.5, 0.3, -0.7, wires=[0])
            qml.RY(np.array(param), wires=[1])
            return qml.var(op)

        assert np.allclose(circuit(), expected, atol=tol, rtol=0)


class TestSample:
    """Tests that samples are properly calculated."""

    @pytest.mark.parametrize(
        "param, shots, wires",
        [
            [[-0.1, -0.2], 10, [0]],
            [[-0.1, -0.2, -0.3], 12, [1]],
            [[-0.1, -0.2, -0.3, -0.4], 17, [0, 1]],
        ],
    )
    def test_sample_broadcast_dimensions(self, qubit_device, param, shots, wires):
        """Tests if the samples returned by the broadcasted sample function have
        the correct dimensions.
        """
        dev = qubit_device(wires=2)
        dev.apply([qml.RX(1.5708, wires=[0]), qml.RX(np.array(param), wires=[1])])

        dev.shots = shots
        dev._wires_measured = wires
        dev._samples = dev.generate_samples()
        s1 = dev.sample(qml.PauliZ(wires=[0]))
        assert np.array_equal(
            s1.shape,
            (
                len(param),
                dev.shots,
            ),
        )

    def test_sample_values(self, qubit_device, tol):
        """Tests if the samples returned by the broadcasted sample calculation have
        the correct values
        """
        dev = qubit_device(wires=2)

        dev.shots = 1000
        dev.apply([qml.RX(np.array([1.5708, 2.5708, 3.5708]), wires=[0])])
        dev._wires_measured = {0}
        dev._samples = dev.generate_samples()

        s1 = dev.sample(qml.PauliZ(0))

        # Each s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1[0] ** 2, 1, atol=tol, rtol=0)
        assert np.allclose(s1[1] ** 2, 1, atol=tol, rtol=0)
        assert np.allclose(s1[2] ** 2, 1, atol=tol, rtol=0)
