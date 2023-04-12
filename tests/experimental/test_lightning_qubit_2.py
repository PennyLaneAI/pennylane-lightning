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
"""General tests for Lightning Qubit 2."""
import pytest

from pennylane import numpy as np

import pennylane as qml
from pennylane.devices.experimental import ExecutionConfig
from pennylane_lightning.experimental import LightningQubit2

from pennylane_lightning.experimental.lightning_qubit_2 import CPP_BINARY_AVAILABLE


@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
def test_name():
    """Tests the name of LightningQubit2."""
    assert LightningQubit2().name == "lightning.qubit.2"


@pytest.mark.skipif(CPP_BINARY_AVAILABLE, reason="Only when there is no binary")
def test_name_no_binary():
    """Tests that we are offloading to the default qubit."""
    assert LightningQubit2().name == "default.qubit.2"


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="Numpy only defines complex256 in Linux-like system"
)
def test_create_device_with_unsupported_dtype():
    with pytest.raises(TypeError, match="Unsupported complex Type:"):
        LightningQubit2(c_dtype=np.complex256)


@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
def test_no_jvp_functionality():
    """Test that jvp is not supported on LightningQubit2."""
    dev = LightningQubit2()

    assert not dev.supports_jvp(ExecutionConfig())

    with pytest.raises(NotImplementedError):
        dev.compute_jvp(qml.tape.QuantumScript(), (10, 10))

    with pytest.raises(NotImplementedError):
        dev.execute_and_compute_jvp(qml.tape.QuantumScript(), (10, 10))


@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
def test_no_vjp_functionality():
    """Test that vjp is not supported on LightningQubit2."""
    dev = LightningQubit2()

    assert not dev.supports_vjp(ExecutionConfig())

    with pytest.raises(NotImplementedError):
        dev.compute_vjp(qml.tape.QuantumScript(), (10.0, 10.0))

    with pytest.raises(NotImplementedError):
        dev.execute_and_compute_vjp(qml.tape.QuantumScript(), (10.0, 10.0))


@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
class TestTracking:
    """Testing the tracking capabilities of LightningQubit2."""

    def test_tracker_set_upon_initialization(self):
        """Test that a new tracker is initialized with each device."""
        assert LightningQubit2.tracker is not LightningQubit2().tracker

    def test_tracker_not_updated_if_not_active(self):
        """Test that the tracker is not updated if not active."""
        dev = LightningQubit2()
        assert len(dev.tracker.totals) == 0

        dev.execute(qml.tape.QuantumScript())
        assert len(dev.tracker.totals) == 0
        assert len(dev.tracker.history) == 0

    def test_tracking_batch(self):
        """Test that the experimental qubit integrates with the tracker."""

        qs = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])

        dev = LightningQubit2()
        with qml.Tracker(dev) as tracker:
            dev.execute(qs)
            dev.execute([qs, qs])  # and a second time

        assert tracker.history == {"batches": [1, 1], "executions": [1, 2]}
        assert tracker.totals == {"batches": 2, "executions": 3}
        assert tracker.latest == {"batches": 1, "executions": 2}


@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
class TestSupportsDerivatives:
    """Test that LightningQubit2 states what kind of derivatives it supports."""

    def test_supports_adjoint(self):
        """Test that LightningQubit2 says that it supports adjoint method."""
        dev = LightningQubit2()
        config = ExecutionConfig(gradient_method="adjoint")
        assert dev.supports_derivatives(config) is True

        qs = qml.tape.QuantumScript([], [qml.state()])
        assert dev.supports_derivatives(config, qs)

    @pytest.mark.parametrize(
        "gradient_method", ["backprop", "parameter-shift", "finite-diff", "device"]
    )
    def test_does_not_support_other_gradient_methods(self, gradient_method):
        """Test that LightningQubit2 currently does not support other gradient methods natively."""
        dev = LightningQubit2()
        config = ExecutionConfig(gradient_method=gradient_method)
        assert not dev.supports_derivatives(config)


@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
class TestBasicCircuit:
    """Tests a basic circuit with one RX gate and two simple expectation values."""

    def test_basic_circuit_numpy(self):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        dev = LightningQubit2()
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))


@pytest.mark.xfail  # [Return before Merge]: xfail for now. Waiting for core update.
@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
class TestExecutingBatches:
    @staticmethod
    def f(phi):
        """A function that executes a batch of scripts on LightningQubit2 without preprocessing."""
        ops = [
            qml.PauliX("a"),
            qml.PauliX("b"),
            qml.ctrl(qml.RX(phi, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
        ]

        qs1 = qml.tape.QuantumScript(
            ops,
            [
                qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
            ],
        )

        ops = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        qs2 = qml.tape.QuantumScript(ops, [qml.probs(wires=(0, 1))])
        return LightningQubit2().execute((qs1, qs2))

    @staticmethod
    def expected(phi):
        out1 = (-qml.math.sin(phi) - 1, 3 * qml.math.cos(phi))

        x1 = qml.math.cos(phi / 2) ** 2 / 2
        x2 = qml.math.sin(phi / 2) ** 2 / 2
        out2 = x1 * np.array([1, 0, 1, 0]) + x2 * np.array([0, 1, 0, 1])
        return (out1, out2)

    @staticmethod
    def nested_compare(x1, x2):
        assert len(x1) == len(x2)
        assert len(x1[0]) == len(x2[0])
        assert qml.math.allclose(x1[0][0], x2[0][0])
        assert qml.math.allclose(x1[0][1], x2[0][1])
        assert qml.math.allclose(x1[1], x2[1])

    def test_numpy(self):
        """Tests that results are expected when the parameter does not have a parameter."""
        phi = 0.892
        results = self.f(phi)
        expected = self.expected(phi)

        self.nested_compare(results, expected)


@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
class TestSumOfTermsDifferentiability:
    """Basically a copy of the `qubit.simulate` test but using the device instead."""

    @staticmethod
    def f(scale, n_wires=10, offset=0.1, convert_to_hamiltonian=False):
        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]

        t1 = 2.5 * qml.prod(*(qml.PauliZ(i) for i in range(n_wires)))
        t2 = 6.2 * qml.prod(*(qml.PauliY(i) for i in range(n_wires)))
        H = t1 + t2
        if convert_to_hamiltonian:
            H = H._pauli_rep.hamiltonian()  # pylint: disable=protected-access
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)])
        return LightningQubit2().execute(qs)

    @staticmethod
    def expected(scale, n_wires=10, offset=0.1, like="numpy"):
        phase = offset + scale * qml.math.asarray(range(n_wires), like=like)
        cosines = qml.math.cos(phase)
        sines = qml.math.sin(phase)
        return 2.5 * qml.math.prod(cosines) + 6.2 * qml.math.prod(sines)


@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
class TestPreprocessingIntegration:
    def test_preprocess_single_circuit(self):
        """Test integration between preprocessing and execution with numpy parameters."""

        # pylint: disable=too-few-public-methods
        class MyTemplate(qml.operation.Operation):
            num_wires = 2

            def decomposition(self):
                return [
                    qml.RX(self.data[0], self.wires[0]),
                    qml.RY(self.data[1], self.wires[1]),
                    qml.CNOT(self.wires),
                ]

        x = 0.928
        y = -0.792
        qscript = qml.tape.QuantumScript(
            [MyTemplate(x, y, ("a", "b"))],
            [qml.expval(qml.PauliY("a")), qml.expval(qml.PauliZ("a")), qml.expval(qml.PauliX("b"))],
        )

        dev = LightningQubit2()

        batch, post_processing_fn = dev.preprocess(qscript)

        assert len(batch) == 1
        execute_circuit = batch[0]
        assert qml.equal(execute_circuit[0], qml.RX(x, "a"))
        assert qml.equal(execute_circuit[1], qml.RY(y, "b"))
        assert qml.equal(execute_circuit[2], qml.CNOT(("a", "b")))
        assert qml.equal(execute_circuit[3], qml.expval(qml.PauliY("a")))
        assert qml.equal(execute_circuit[4], qml.expval(qml.PauliZ("a")))
        assert qml.equal(execute_circuit[5], qml.expval(qml.PauliX("b")))

        results = dev.execute(batch)
        assert len(results) == 1
        assert len(results[0]) == 3

        processed_results = post_processing_fn(results)
        assert len(processed_results) == 3
        assert qml.math.allclose(processed_results[0], -np.sin(x) * np.sin(y))
        assert qml.math.allclose(processed_results[1], np.cos(x))
        assert qml.math.allclose(processed_results[2], np.sin(y))

    def test_preprocess_batch_circuit(self):
        """Test preprocess integrates with lightning qubit when we start with a batch of circuits."""

        # pylint: disable=too-few-public-methods
        class CustomIsingXX(qml.operation.Operation):
            num_wires = 2

            def decomposition(self):
                return [qml.IsingXX(self.data[0], self.wires)]

        x = 0.692

        measurements1 = [qml.density_matrix("a"), qml.vn_entropy("a")]
        qs1 = qml.tape.QuantumScript([CustomIsingXX(x, ("a", "b"))], measurements1)

        y = -0.923

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(wires=1)
            m_0 = qml.measure(1)
            qml.cond(m_0, qml.RY)(y, wires=0)
            qml.expval(qml.PauliZ(0))

        qs2 = qml.tape.QuantumScript.from_queue(q)

        initial_batch = [qs1, qs2]

        dev = LightningQubit2()
        batch, post_processing_fn = dev.preprocess(initial_batch)

        results = dev.execute(batch)
        processed_results = post_processing_fn(results)

        assert len(processed_results) == 2
        assert len(processed_results[0]) == 2

        expected_density_mat = np.array([[np.cos(x / 2) ** 2, 0], [0, np.sin(x / 2) ** 2]])
        assert qml.math.allclose(processed_results[0][0], expected_density_mat)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(x / 2) ** 2 * np.sin(x / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(x / 2) ** 2 * np.sin(x / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = -np.sum(eigs * np.log(eigs))
        assert qml.math.allclose(processed_results[0][1], expected_entropy)

        expected_expval = np.cos(y)
        assert qml.math.allclose(expected_expval, processed_results[1])


@pytest.mark.skipif(not CPP_BINARY_AVAILABLE, reason="Lightning binary required")
def test_broadcasted_parameter():
    """Test that LightningQubit2 handles broadcasted parameters as expected."""
    dev = LightningQubit2()
    x = np.array([0.536, 0.894])
    qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
    batch, post_processing_fn = dev.preprocess(qs)
    assert len(batch) == 2
    results = dev.execute(batch)
    processed_results = post_processing_fn(results)
    assert qml.math.allclose(processed_results, np.cos(x))
