# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests specific to ``lightning.tensornet`` (Kokkos MPS / MPDO simulator):
bond-dimension options, truncation, pure/mixed modes and circuits on 30-40
qubits. Large-n references are analytic or come from the exact light cone of
the measured observable simulated on ``default.mixed``.
"""

# pylint: disable=missing-function-docstring
import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice, device_name
from pennylane.exceptions import DeviceError

if device_name != "lightning.tensornet":
    pytest.skip("Tests for lightning.tensornet.", allow_module_level=True)

if LightningDevice._ops_module() is None:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

# pylint: disable=wrong-import-position
from pennylane_lightning.lightning_tensornet_ops import MPSC64, MPSC128

pytestmark = pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")

NOISE = qp.NoiseModel(
    {
        qp.noise.op_eq("CNOT"): qp.noise.partial_wires(qp.DepolarizingChannel, 0.02),
        qp.noise.op_eq("RY"): qp.noise.partial_wires(qp.AmplitudeDamping, 0.05),
    }
)


def dev_tn(wires, **kwargs):
    return qp.device("lightning.tensornet", wires=wires, **kwargs)


class TestOptions:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_bond_dim": 0}, "max_bond_dim"),
            ({"max_bond_dim": -3}, "max_bond_dim"),
            ({"max_bond_dim": 2.5}, "max_bond_dim"),
            ({"max_bond_dim": True}, "max_bond_dim"),
            ({"cutoff": -1e-3}, "cutoff"),
            ({"cutoff": float("nan")}, "cutoff"),
            ({"cutoff": "1e-3"}, "cutoff"),
            ({"cutoff_mode": "max"}, "cutoff_mode"),
        ],
    )
    def test_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            dev_tn(2, **kwargs)

    def test_properties_and_device_kwargs(self):
        dev = dev_tn(3, max_bond_dim=np.int64(16), cutoff=1e-10, cutoff_mode="rel")
        assert (dev.max_bond_dim, dev.cutoff, dev.cutoff_mode) == (16, 1e-10, "rel")
        assert dev.device_kwargs == {"max_bond_dim": 16, "cutoff": 1e-10, "cutoff_mode": "rel"}
        assert dev.truncation_error == 0.0
        noisy = dev_tn(3, noise_model=NOISE, noise_mode="native")
        assert set(noisy.device_kwargs) == {"noise_model", "max_bond_dim", "cutoff", "cutoff_mode"}

    def test_bindings(self):
        for cls in (MPSC64, MPSC128):
            s = cls(4, 8, 0.0, "abs", False)
            assert s.num_qubits == 4 and s.max_bond_dim == 8 and not s.is_mixed()
            assert s.bond_dims() == [1, 1, 1, 1, 1]
            s.apply_operation("Hadamard", [0], False, [])
            s.apply_operation("CNOT", [0, 3], False, [])
            assert s.bond_dims() == [1, 2, 2, 2, 1]
            sv = s.state_vector()
            assert np.allclose(np.abs(sv[[0, 9]]) ** 2, 0.5, atol=1e-6)
            s.to_mixed()
            assert s.is_mixed() and s.bond_dims() == [1, 4, 4, 4, 1]
            assert np.isclose(s.purity(), 1.0, atol=1e-5)
            with pytest.raises(ValueError, match="mixed"):
                s.state_vector()
            assert len(s.site_tensors()) == 4
            with pytest.raises(ValueError):
                cls(2, 0)
            with pytest.raises(ValueError):
                cls(2, 4, -1.0)
            with pytest.raises(ValueError):
                cls(2, 4, 0.0, "x")
        assert MPSC128(3, mixed=True).is_mixed()


class TestPureAndMixedModes:
    def test_state_pure(self):
        def circ():
            qp.H(0)
            qp.CNOT([0, 3])
            qp.RY(0.3, 2)
            qp.CRX(0.5, [2, 1])
            return qp.state()

        res = qp.QNode(circ, dev_tn(4))()
        ref = qp.QNode(circ, qp.device("default.qubit", wires=4))()
        assert np.allclose(res, ref, atol=1e-12)

    @pytest.mark.parametrize("noise", ["channel", "model"])
    def test_state_mixed_raises(self, noise):
        def circ():
            qp.H(0)
            qp.CNOT([0, 1])
            if noise == "channel":
                qp.PhaseDamping(0.1, 1)
            return qp.state()

        dev = dev_tn(2, noise_model=NOISE if noise == "model" else None)
        with pytest.raises(DeviceError, match="qml.density_matrix"):
            qp.QNode(circ, dev)()

    def test_unitary_channel_stays_pure(self):
        """A channel with a single (unitary) Kraus operator does not switch to MPDO mode."""
        dev = dev_tn(2)

        def circ():
            qp.H(0)
            qp.QubitChannel([qp.matrix(qp.RX(0.4, 0))], wires=0)
            return qp.state()

        ref = qp.QNode(lambda: (qp.H(0), qp.RX(0.4, 0), qp.state())[-1], dev_tn(2))()
        assert np.allclose(qp.QNode(circ, dev)(), ref, atol=1e-12)
        assert not dev._state.is_mixed()  # pylint: disable=protected-access

    def test_dense_operation_limit(self):
        u6 = np.linalg.qr(np.random.default_rng(1).normal(size=(64, 64)))[0]

        def circ():
            qp.BitFlip(0.1, 0)
            qp.QubitUnitary(u6, wires=range(6))
            return qp.probs(wires=[0])

        with pytest.raises(RuntimeError, match="too many wires"):
            qp.QNode(circ, dev_tn(7))()


def _layered(n, layers=3):
    for i in range(n):
        qp.RY(0.3 + 0.17 * i, i)
    for layer in range(layers):
        for i in range(layer % 2, n - 1, 2):
            qp.CNOT([i, i + 1])
        for i in range(n):
            qp.RX(0.4 + 0.05 * i + 0.1 * layer, i)


class TestTruncation:
    def test_exact_without_truncation(self):
        circ = lambda: (_layered(8), qp.expval(qp.Z(3) @ qp.Z(4)), qp.probs(wires=[0, 7]))[1:]
        dev = dev_tn(8, max_bond_dim=16)
        res = qp.QNode(circ, dev)()
        ref = qp.QNode(circ, qp.device("default.qubit", wires=8))()
        assert dev.truncation_error == 0.0
        assert np.allclose(res[0], ref[0], atol=1e-12)
        assert np.allclose(res[1], ref[1], atol=1e-12)

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_max_bond_dim_truncates(self, c_dtype):
        circ = lambda: (_layered(8), qp.expval(qp.Z(3) @ qp.Z(4)))[1]
        ref = qp.QNode(circ, qp.device("default.qubit", wires=8))()
        dev = dev_tn(8, max_bond_dim=2, c_dtype=c_dtype)
        res = qp.QNode(circ, dev)()
        bonds = dev._state.bond_dims()  # pylint: disable=protected-access
        assert max(bonds) == 2
        assert 1e-4 < dev.truncation_error < 1
        assert not np.isclose(res, ref, atol=1e-4)
        # the truncation error is recomputed for every circuit
        qp.QNode(lambda: (qp.H(0), qp.expval(qp.Z(0)))[1], dev)()
        assert dev.truncation_error == 0.0

    def test_cutoff(self):
        def circ():
            qp.RY(0.02, 0)
            qp.CNOT([0, 1])
            qp.H(2)
            qp.CNOT([2, 3])
            qp.CNOT([1, 2])
            return qp.expval(qp.Z(0) @ qp.Z(1)), qp.probs(wires=[0])

        ref = qp.QNode(circ, dev_tn(4))()
        for mode in ("abs", "rel"):
            dev = dev_tn(4, cutoff=0.05, cutoff_mode=mode)
            res = qp.QNode(circ, dev)()
            assert dev.truncation_error > 0
            assert not np.allclose(res[1], ref[1], atol=1e-6)
            # the truncated state is renormalized
            assert np.isclose(np.sum(res[1]), 1.0)


class TestLargeCircuits:
    @pytest.mark.parametrize("n", [30, 40])
    def test_ghz(self, n):
        def circ():
            qp.H(0)
            for i in range(n - 1):
                qp.CNOT([i, i + 1])
            return (
                qp.expval(qp.Z(0) @ qp.Z(n - 1)),
                qp.probs(wires=[0, n - 1]),
                qp.density_matrix([0, n - 1]),
                qp.vn_entropy([n // 2]),
            )

        dev = dev_tn(n)
        res = qp.QNode(circ, dev)()
        assert np.isclose(res[0], 1.0)
        assert np.allclose(res[1], [0.5, 0, 0, 0.5])
        rho = np.zeros((4, 4))
        rho[0, 0] = rho[3, 3] = 0.5
        assert np.allclose(res[2], rho, atol=1e-12)
        assert np.isclose(res[3], np.log(2))
        assert max(dev._state.bond_dims()) == 2  # pylint: disable=protected-access

    def test_ghz_parity_observable(self):
        """<X...X> = 1 on the GHZ state (a 40-qubit Pauli string)."""
        n = 40

        def circ():
            qp.H(0)
            for i in range(n - 1):
                qp.CNOT([i, i + 1])
            return qp.expval(qp.prod(*[qp.X(i) for i in range(n)]))

        assert np.isclose(qp.QNode(circ, dev_tn(n))(), 1.0)

    def test_ghz_shots(self):
        n = 40

        def circ():
            qp.H(0)
            for i in range(n - 1):
                qp.CNOT([i, i + 1])
            return qp.sample(wires=[0, 13, 39]), qp.expval(qp.Z(0) @ qp.Z(39))

        samples, ez = qp.set_shots(qp.QNode(circ, dev_tn(n, seed=3)), 500)()
        samples = np.asarray(samples)
        assert samples.shape == (500, 3)
        assert np.all(samples == samples[:, :1])
        assert 0.35 < samples[:, 0].mean() < 0.65
        assert ez == 1.0

    @staticmethod
    def _brickwork(n, keep=None):
        """RY layer, CNOTs on (even, odd) pairs, RZ layer, CNOTs on (odd, even)
        pairs. With ``keep`` only the gates acting inside ``keep`` are applied:
        this is exact for observables on wires 15 and 16 when
        keep = {14, 15, 16, 17} (their backward light cone)."""

        def ok(*w):
            return keep is None or all(x in keep for x in w)

        for i in range(n):
            if ok(i):
                qp.RY(0.2 + 0.07 * i, i)
        for i in range(0, n - 1, 2):
            if ok(i, i + 1):
                qp.CNOT([i, i + 1])
        for i in range(n):
            if ok(i):
                qp.RZ(0.5 + 0.03 * i, i)
                qp.RX(0.3, i)
        for i in range(1, n - 1, 2):
            if ok(i, i + 1):
                qp.CNOT([i, i + 1])

    @pytest.mark.parametrize("mode", ["transform", "native"])
    def test_noisy_40_qubits_light_cone(self, mode):
        n = 40
        keep = [14, 15, 16, 17]

        def meas():
            return qp.expval(qp.Z(15)), qp.expval(qp.Y(15) @ qp.X(16)), qp.probs(wires=[16])

        def full():
            self._brickwork(n)
            return meas()

        def cone():
            self._brickwork(n, keep=set(keep))
            return meas()

        dev = dev_tn(n, noise_model=NOISE, noise_mode=mode)
        res = qp.QNode(full, dev)()
        assert dev._state.is_mixed()  # pylint: disable=protected-access
        assert dev.truncation_error == 0.0
        ref_dev = qp.noise.add_noise(qp.device("default.mixed", wires=keep), NOISE)
        ref = qp.QNode(cone, ref_dev)()
        for a, b in zip(res, ref):
            assert np.allclose(a, b, atol=1e-10)

    def test_noisy_40_qubits_qjit(self):
        pytest.importorskip("catalyst")
        n = 40

        def full():
            self._brickwork(n)
            return qp.expval(qp.Z(15)), qp.probs(wires=[15, 16])

        dev = dev_tn(n, noise_model=NOISE, noise_mode="native")
        res = qp.qjit(qp.QNode(full, dev))()
        ref = qp.QNode(full, dev)()
        for a, b in zip(res, ref):
            assert np.allclose(a, b, atol=1e-10)

    def test_mcm_large(self):
        """Mid-circuit measurement on a 30-qubit GHZ state (one-shot)."""
        n = 30

        def circ():
            qp.H(0)
            for i in range(n - 1):
                qp.CNOT([i, i + 1])
            m = qp.measure(n // 2)
            qp.cond(m, qp.X)(0)
            return qp.expval(qp.Z(0)), qp.expval(m)

        ez, em = qp.set_shots(qp.QNode(circ, dev_tn(n, seed=9), mcm_method="one-shot"), 300)()
        # X on wire 0 is applied exactly when wire 0 was |1>, so wire 0 ends in |0>
        assert np.isclose(ez, 1.0)
        assert 0.35 < em < 0.65


def test_matches_density_matrix_simulator():
    """MPDO and dense density-matrix simulation of the same noisy circuit."""
    n = 8
    ref_dev = qp.noise.add_noise(qp.device("default.mixed", wires=n), NOISE)

    def circ():
        _layered(n)
        return qp.expval(qp.Z(2) @ qp.X(5)), qp.probs(wires=[0, 4]), qp.purity([1, 2, 3])

    res = qp.QNode(circ, dev_tn(n, noise_model=NOISE, noise_mode="native"))()
    ref = qp.QNode(circ, ref_dev)()
    for x, y in zip(res, ref):
        assert np.allclose(x, y, atol=1e-10)
