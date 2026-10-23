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
Device-level tests of ``lightning.tensornet`` (run with
``PL_DEVICE=lightning.tensornet``): gates, channels, noise models, measurements
and mid-circuit measurements. References are ``default.qubit`` /
``default.mixed`` or an explicit NumPy density-matrix calculation.
"""

# pylint: disable=missing-function-docstring,too-few-public-methods
import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice, device_name
from pennylane import numpy as pnp
from pennylane.exceptions import DeviceError, QuantumFunctionError

if device_name != "lightning.tensornet":
    pytest.skip("Tests for lightning.tensornet.", allow_module_level=True)

if LightningDevice._ops_module() is None:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

# pylint: disable=wrong-import-position
from pennylane_lightning.lightning_base._noise import NativeNoiseError

# PennyLane's PauliError Kraus construction casts the (unused) Y matrix to a real
# dtype for Pauli words without Y, which emits a harmless ComplexWarning.
pytestmark = pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")

TOL = {np.complex128: 1e-10, np.complex64: 2e-5}


def make_dev(wires, **kwargs):
    return qp.device(device_name, wires=wires, **kwargs)


def assert_all_close(res, ref, atol):
    assert len(res) == len(ref)
    for a, b in zip(res, ref):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=atol), (a, b)


def random_unitary(dim, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def prepare(n):
    """Entangled, non-symmetric initial state on n wires."""
    for i in range(n):
        qp.RY(0.4 + 0.3 * i, i)
        qp.RX(0.1 * (i + 1), i)
    for i in range(n - 1):
        qp.CNOT([i, i + 1])
    qp.RZ(0.7, 0)


# ---------------------------------------------------------------------------
# Device construction and errors
# ---------------------------------------------------------------------------


class TestDevice:
    def test_name_and_dtype(self):
        dev = make_dev(2)
        assert dev.name == device_name
        assert dev.c_dtype == np.complex128
        assert make_dev(2, c_dtype=np.complex64).c_dtype == np.complex64

    def test_invalid_dtype(self):
        with pytest.raises(TypeError, match="Unsupported complex type"):
            make_dev(2, c_dtype=np.float64)

    def test_backend_info(self):
        info = LightningDevice._ops_module().backend_info()  # pylint: disable=protected-access
        assert info["NAME"] == device_name
        assert info["EXECUTION_SPACE"]

    def test_noise_mode_errors(self):
        with pytest.raises(ValueError, match="noise_mode must be one of"):
            make_dev(2, noise_mode="fast")
        with pytest.raises(ValueError, match="requires noise_mode='native'"):
            make_dev(2, noise_model="LNM1;", noise_mode="transform")
        with pytest.raises(NativeNoiseError):
            make_dev(2, noise_model="LNM1;garbage")
        with pytest.raises(TypeError, match="noise_model must be"):
            make_dev(2, noise_model=3)

    def test_kokkos_args_type(self):
        with pytest.raises(TypeError):
            make_dev(2, kokkos_args=3)

    def test_device_options(self):
        cfg = qp.devices.ExecutionConfig(device_options={"a": 1})
        with pytest.raises(DeviceError, match="device option a not present"):
            make_dev(1).setup_execution_config(cfg)

    @pytest.mark.parametrize("method", ["adjoint", "backprop", "device"])
    def test_device_derivatives_not_supported(self, method):
        with pytest.raises(QuantumFunctionError, match=f"does not support {method}"):
            qp.QNode(lambda: qp.expval(qp.Z(0)), make_dev(1), diff_method=method)

    def test_tree_traversal_not_supported(self):
        def circ():
            m = qp.measure(0)
            qp.cond(m, qp.X)(1)
            return qp.expval(qp.Z(1))

        qnode = qp.set_shots(qp.QNode(circ, make_dev(2), mcm_method="tree-traversal"), 10)
        with pytest.raises(DeviceError, match="not supported"):
            qnode()

    def test_parameter_dependent_native_noise(self):
        def pdep(op, **_):
            qp.DepolarizingChannel(abs(op.parameters[0]) / 10, op.wires[0])

        nm = qp.NoiseModel({qp.noise.op_eq("RX"): pdep})
        circ = lambda: (qp.RX(0.3, 0), qp.expval(qp.Z(0)))[1]
        with pytest.raises(NativeNoiseError):
            qp.QNode(circ, make_dev(1, noise_model=nm, noise_mode="native"))()
        # the transform mode handles it (PennyLane inserts the channels)
        res = qp.QNode(circ, make_dev(1, noise_model=nm))()
        ref = qp.QNode(circ, qp.noise.add_noise(qp.device("default.mixed", wires=1), nm))()
        assert np.allclose(res, ref)

    def test_native_noise_model_string(self):
        nm = qp.NoiseModel({qp.noise.op_eq("CNOT"): qp.noise.partial_wires(qp.BitFlip, 0.1)})
        dev = make_dev(3, noise_model=nm, noise_mode="native")
        spec = dev.native_noise_model()
        assert spec.startswith("LNM1;")
        assert dev.device_kwargs["noise_model"] == spec
        # the serialized model can be handed back to a device
        dev2 = make_dev(3, noise_model=spec)
        assert dev2.noise_mode == "native"
        circ = lambda: (qp.H(0), qp.CNOT([0, 1]), qp.CNOT([1, 2]), qp.probs())[-1]
        assert np.allclose(qp.QNode(circ, dev)(), qp.QNode(circ, dev2)())
        assert make_dev(3).native_noise_model() == ""


# ---------------------------------------------------------------------------
# Unitary simulation
# ---------------------------------------------------------------------------

N_GATE = 4
GATES = [
    qp.CNOT([0, 2]),
    qp.ControlledPhaseShift(0.3, [1, 0]),
    qp.CRot(0.1, 0.2, 0.3, [2, 1]),
    qp.CRX(0.4, [3, 0]),
    qp.CRY(0.5, [0, 3]),
    qp.CRZ(0.6, [1, 2]),
    qp.CSWAP([3, 0, 2]),
    qp.CY([2, 0]),
    qp.CZ([1, 3]),
    qp.DoubleExcitation(0.4, [0, 1, 2, 3]),
    qp.DoubleExcitationMinus(0.3, [3, 1, 2, 0]),
    qp.DoubleExcitationPlus(0.2, [0, 2, 1, 3]),
    qp.GlobalPhase(0.3),
    qp.Hadamard(2),
    qp.Identity(1),
    qp.IsingXX(0.3, [0, 3]),
    qp.IsingXY(0.4, [2, 1]),
    qp.IsingYY(0.5, [1, 3]),
    qp.IsingZZ(0.6, [3, 2]),
    qp.MultiRZ(0.5, [0, 1, 3]),
    qp.PauliX(1),
    qp.PauliY(2),
    qp.PauliZ(0),
    qp.PauliRot(0.4, "XYIZ", [0, 1, 2, 3]),
    qp.PauliRot(0.4, "YZ", [3, 1]),
    qp.PauliRot(0.4, "IIII", [0, 1, 2, 3]),
    qp.PhaseShift(0.3, 3),
    qp.PSWAP(0.4, [0, 2]),
    qp.QubitUnitary(random_unitary(4, 1), [3, 1]),
    qp.QubitUnitary(random_unitary(8, 2), [2, 0, 3]),
    qp.Rot(0.1, 0.2, 0.3, 1),
    qp.RX(0.3, 0),
    qp.RY(0.4, 1),
    qp.RZ(0.5, 2),
    qp.SingleExcitation(0.3, [0, 3]),
    qp.SingleExcitationMinus(0.4, [2, 1]),
    qp.SingleExcitationPlus(0.5, [1, 3]),
    qp.S(3),
    qp.SWAP([0, 3]),
    qp.Toffoli([3, 1, 0]),
    qp.T(2),
    qp.SX(1),
    qp.adjoint(qp.S(0)),
    qp.adjoint(qp.IsingXY(0.3, [1, 2])),
    qp.ctrl(qp.RX(0.2, 0), 3),
    qp.ctrl(qp.Hadamard(1), [2, 3], control_values=[0, 1]),
    qp.ctrl(qp.QubitUnitary(random_unitary(2, 3), 2), [0, 1], control_values=[1, 0]),
    qp.MultiControlledX(wires=[0, 1, 2, 3], control_values=[1, 0, 1]),
    qp.ControlledQubitUnitary(random_unitary(2, 4), wires=[3, 0]),
    qp.DiagonalQubitUnitary(np.exp(1j * np.arange(4)), [2, 0]),
    qp.ECR([1, 2]),
    qp.ISWAP([3, 0]),
    qp.OrbitalRotation(0.3, [0, 1, 2, 3]),
    qp.QubitCarry([0, 1, 2, 3]),
    qp.QubitSum([3, 1, 2]),
    qp.SISWAP([0, 2]),
    qp.SQISW([1, 3]),
]


class TestUnitary:
    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    @pytest.mark.parametrize("gate", GATES, ids=lambda g: g.name)
    def test_gate(self, gate, c_dtype):
        def circ():
            prepare(N_GATE)
            qp.apply(gate)
            return qp.density_matrix(range(N_GATE))

        def ref_circ():
            prepare(N_GATE)
            qp.apply(gate)
            return qp.state()

        psi = qp.QNode(ref_circ, qp.device("default.qubit", wires=N_GATE))()
        rho = qp.QNode(circ, make_dev(N_GATE, c_dtype=c_dtype))()
        assert np.allclose(rho, np.outer(psi, psi.conj()), atol=TOL[c_dtype])

    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    def test_measurements(self, c_dtype):
        herm = np.array([[1.0, 0.5 - 0.2j], [0.5 + 0.2j, -0.3]])

        def circ(x):
            prepare(4)
            qp.CRY(x, [1, 2])
            qp.Toffoli([0, 1, 3])
            return (
                qp.expval(qp.Z(0) @ qp.Z(1)),
                qp.expval(qp.X(2) @ qp.Y(3)),
                qp.expval(0.3 * qp.Z(0) + 0.5 * qp.X(1) @ qp.X(3) - 0.2 * qp.Y(2)),
                qp.expval(qp.Hermitian(herm, 2)),
                qp.expval(qp.Hadamard(1)),
                qp.expval(qp.Projector([1, 0], [0, 3])),
                qp.expval(qp.Identity(0)),
                qp.var(qp.Y(3)),
                qp.var(qp.X(0) @ qp.Z(2) + 0.4 * qp.Y(1)),
                qp.probs(wires=[1, 2]),
                qp.probs(wires=[3, 0]),
                qp.probs(),
                qp.probs(op=qp.X(1)),
                qp.density_matrix([0, 3]),
                qp.purity([0, 1]),
                qp.vn_entropy([2]),
                qp.mutual_info([0], [3]),
            )

        res = qp.QNode(circ, make_dev(4, c_dtype=c_dtype))(0.8)
        ref = qp.QNode(circ, qp.device("default.mixed", wires=4))(0.8)
        assert_all_close(res, ref, atol=50 * TOL[c_dtype])

    def test_wire_labels(self):
        dev = make_dev(["a", 3, "c"])

        def circ():
            qp.H("a")
            qp.CNOT(["a", "c"])
            qp.RY(0.3, 3)
            return qp.probs(wires=["c", 3]), qp.expval(qp.Z("a") @ qp.Z("c"))

        res = qp.QNode(circ, dev)()
        ref = qp.QNode(circ, qp.device("default.mixed", wires=["a", 3, "c"]))()
        assert_all_close(res, ref, atol=1e-10)

    def test_wires_none(self):
        def circ():
            qp.H(0)
            qp.CNOT([0, 4])
            return qp.probs(wires=[0, 4])

        assert np.allclose(qp.QNode(circ, make_dev(None))(), [0.5, 0, 0, 0.5])

    def test_state_preparation(self):
        vec = np.arange(1, 9) / np.linalg.norm(np.arange(1, 9))

        def circ():
            qp.StatePrep(vec, wires=[2, 0, 1])
            qp.CNOT([0, 1])
            return qp.probs(wires=[0, 1, 2])

        assert np.allclose(
            qp.QNode(circ, make_dev(3))(), qp.QNode(circ, qp.device("default.qubit", wires=3))()
        )

        def circ_b():
            qp.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
            qp.H(1)
            qp.StatePrep(np.array([0, 1, 1, 0]) / np.sqrt(2), wires=[0, 2])
            return qp.probs(wires=[0, 1, 2])

        assert np.allclose(
            qp.QNode(circ_b, make_dev(3))(),
            qp.QNode(circ_b, qp.device("default.qubit", wires=3))(),
        )

    def test_broadcasting(self):
        def circ(x):
            qp.RX(x, 0)
            qp.CNOT([0, 1])
            return qp.expval(qp.Z(1))

        x = np.array([0.1, 0.5, 1.2])
        assert np.allclose(qp.QNode(circ, make_dev(2))(x), np.cos(x))


# ---------------------------------------------------------------------------
# Channels used directly in the circuit
# ---------------------------------------------------------------------------


def _random_channel(nq, nk, seed):
    rng = np.random.default_rng(seed)
    dim = 2**nq
    a = rng.normal(size=(nk * dim, dim)) + 1j * rng.normal(size=(nk * dim, dim))
    q, _ = np.linalg.qr(a)
    return [q[k * dim : (k + 1) * dim] for k in range(nk)]


CHANNELS = [
    qp.AmplitudeDamping(0.3, 1),
    qp.GeneralizedAmplitudeDamping(0.2, 0.4, 2),
    qp.PhaseDamping(0.25, 0),
    qp.DepolarizingChannel(0.15, 3),
    qp.BitFlip(0.1, 2),
    qp.PhaseFlip(0.2, 1),
    qp.PauliError("XY", 0.2, wires=[0, 2]),
    qp.PauliError("ZIX", 0.3, wires=[3, 1, 0]),
    qp.ResetError(0.1, 0.2, 1),
    qp.ThermalRelaxationError(0.1, 100, 80, 20, 0),
    qp.ThermalRelaxationError(0.2, 100, 180, 20, 2),
    qp.QubitChannel(_random_channel(1, 3, 5), wires=3),
    qp.QubitChannel(_random_channel(2, 2, 6), wires=[2, 0]),
]


class TestChannels:
    @pytest.mark.parametrize("c_dtype", [np.complex128, np.complex64])
    @pytest.mark.parametrize("channel", CHANNELS, ids=lambda c: c.name)
    def test_channel(self, channel, c_dtype):
        def circ():
            prepare(4)
            qp.apply(channel)
            qp.CNOT([1, 3])
            qp.RX(0.3, 2)
            return qp.density_matrix(range(4)), qp.expval(qp.Z(1) @ qp.X(2)), qp.purity(range(4))

        res = qp.QNode(circ, make_dev(4, c_dtype=c_dtype))()
        ref = qp.QNode(circ, qp.device("default.mixed", wires=4))()
        assert_all_close(res, ref, atol=10 * TOL[c_dtype])
        assert res[2] < 1 - 1e-3


# ---------------------------------------------------------------------------
# Noise models (transform and native modes)
# ---------------------------------------------------------------------------


def _noise_circ(x):
    qp.Hadamard(0)
    qp.RX(x, 1)
    qp.CNOT([0, 1])
    qp.CRY(0.4, [1, 2])
    qp.RZ(0.3, 3)
    qp.Toffoli([0, 1, 3])
    qp.adjoint(qp.S(2))
    qp.ctrl(qp.RX(0.2, 0), 3)
    qp.IsingXX(0.7, [2, 3])
    qp.MultiRZ(0.5, [0, 1, 2])
    qp.PauliRot(0.3, "XYZ", [0, 2, 3])
    qp.T(1)
    qp.SWAP([0, 3])
    return (
        qp.expval(qp.Z(0) @ qp.Z(1)),
        qp.probs(wires=[1, 2]),
        qp.expval(qp.X(2)),
        qp.var(qp.Y(3)),
        qp.density_matrix([0, 3]),
    )


def _n4(op, **_):
    qp.PhaseDamping(0.05, op.wires[0])
    qp.apply(op)
    qp.BitFlip(0.02, op.wires[-1])


def _n3(op, **_):
    qp.PauliError("X" * len(op.wires), 0.1, op.wires)


@qp.BooleanFn
def _is_pauli_rot(op):
    return op.name == "PauliRot"


def _n5(op, **_):
    for w in op.wires:
        qp.DepolarizingChannel(0.03, w)


_C0 = qp.noise.op_eq(qp.CNOT)
NOISE_MODELS = {
    "basic": qp.NoiseModel({_C0: qp.noise.partial_wires(qp.DepolarizingChannel, 0.1)}),
    "multi": qp.NoiseModel(
        {
            _C0: qp.noise.partial_wires(qp.AmplitudeDamping, 0.2),
            qp.noise.op_in(["RX", "RZ"])
            & qp.noise.wires_in([1, 3]): qp.noise.partial_wires(qp.PhaseFlip, 0.15),
            qp.noise.wires_eq(2): qp.noise.partial_wires(qp.BitFlip, 0.05),
            qp.noise.op_eq("Toffoli") | qp.noise.op_eq("IsingXX"): _n3,
        }
    ),
    "apply": qp.NoiseModel(
        {
            qp.noise.op_eq("MultiRZ"): _n4,
            _is_pauli_rot: _n5,
            _C0: qp.noise.partial_wires(qp.GeneralizedAmplitudeDamping, 0.1, 0.3),
        }
    ),
    "thermal": qp.NoiseModel(
        {
            qp.noise.op_in(["Hadamard", "T", "SWAP"]): qp.noise.partial_wires(
                qp.ThermalRelaxationError, 0.1, 100, 80, 20
            )
        }
    ),
}


def _reference(circ, n, nm, *args, **kwargs):
    """Noise inserted by ``add_noise`` on the circuit decomposed to the device
    gate set (the noise semantics of the devices), executed on default.mixed."""
    tape = qp.workflow.construct_tape(qp.QNode(circ, qp.device("default.mixed", wires=n)))(
        *args, **kwargs
    )
    (dec,), _ = make_dev(n).preprocess_transforms()((tape,))
    dec = qp.tape.QuantumScript(dec.operations, tape.measurements, shots=tape.shots)
    tapes, fn = qp.noise.add_noise(dec, nm)
    return fn(qp.execute(tapes, qp.device("default.mixed", wires=n)))


class TestNoiseModels:
    @pytest.mark.parametrize("mode", ["transform", "native"])
    @pytest.mark.parametrize("name", list(NOISE_MODELS))
    def test_gate_noise(self, name, mode):
        nm = NOISE_MODELS[name]
        ref = _reference(_noise_circ, 4, nm, 0.7)
        res = qp.QNode(_noise_circ, make_dev(4, noise_model=nm, noise_mode=mode))(0.7)
        assert_all_close(res, ref, atol=1e-9)

    def test_native_matches_transform_complex64(self):
        nm = NOISE_MODELS["multi"]
        a = qp.QNode(_noise_circ, make_dev(4, noise_model=nm, c_dtype=np.complex64))(0.7)
        b = qp.QNode(
            _noise_circ,
            make_dev(4, noise_model=nm, noise_mode="native", c_dtype=np.complex64),
        )(0.7)
        assert_all_close(a, b, atol=1e-5)

    @pytest.mark.parametrize("mode", ["transform", "native"])
    def test_readout_noise(self, mode):
        def circ():
            qp.RY(0.8, 0)
            qp.CNOT([0, 1])
            qp.RX(0.5, 2)
            qp.CZ([1, 2])
            return (
                qp.expval(qp.Z(0)),
                qp.expval(qp.Z(1) @ qp.Z(2)),
                qp.probs(wires=[0, 2]),
                qp.var(qp.X(2)),
                qp.expval(qp.Z(2)),
                qp.density_matrix([0, 1]),
            )

        gate = {qp.noise.op_eq("CNOT"): qp.noise.partial_wires(qp.DepolarizingChannel, 0.05)}
        meas = {
            qp.noise.meas_eq(qp.expval)
            & qp.noise.wires_in([0, 1]): qp.noise.partial_wires(qp.BitFlip, 0.2),
            qp.noise.meas_eq(qp.probs): qp.noise.partial_wires(qp.AmplitudeDamping, 0.1),
            qp.noise.meas_eq(qp.var): qp.noise.partial_wires(qp.PhaseFlip, 0.3),
        }
        nm = qp.NoiseModel(gate, meas_map=meas)
        tape = qp.workflow.construct_tape(qp.QNode(circ, make_dev(3)))()
        tapes, fn = qp.noise.add_noise(tape, nm)
        ref = fn(qp.execute(tapes, qp.device("default.mixed", wires=3)))
        res = qp.QNode(circ, make_dev(3, noise_model=nm, noise_mode=mode))()
        assert_all_close(res, ref, atol=1e-9)

    def test_noise_changes_results(self):
        nm = NOISE_MODELS["basic"]
        clean = qp.QNode(_noise_circ, make_dev(4))(0.7)
        noisy = qp.QNode(_noise_circ, make_dev(4, noise_model=nm, noise_mode="native"))(0.7)
        assert not np.allclose(clean[0], noisy[0])


# ---------------------------------------------------------------------------
# Finite shots
# ---------------------------------------------------------------------------


def _shot_prep():
    qp.RY(0.9, 0)
    qp.RX(0.4, 1)
    qp.CNOT([0, 2])
    qp.AmplitudeDamping(0.3, 2)
    qp.BitFlip(0.1, 1)


class TestShots:
    def test_statistics(self):
        def circ():
            _shot_prep()
            return (
                qp.expval(qp.Z(0)),
                qp.expval(qp.X(1) @ qp.Z(2)),
                qp.var(qp.Y(1)),
                qp.probs(wires=[0, 2]),
                qp.expval(0.3 * qp.Z(0) + 0.5 * qp.X(1)),
                qp.expval(qp.Hermitian(np.array([[1, 0.5], [0.5, -1]]), 2)),
            )

        exact = qp.QNode(circ, qp.device("default.mixed", wires=3))()
        res = qp.set_shots(qp.QNode(circ, make_dev(3, seed=11)), 40000)()
        assert_all_close(res, exact, atol=0.03)

    def test_sample_and_counts_shapes(self):
        def circ():
            _shot_prep()
            return (
                qp.sample(wires=[0, 1]),
                qp.counts(wires=[2]),
                qp.sample(qp.X(1)),
                qp.counts(qp.Z(0) @ qp.Z(2)),
                qp.sample(),
            )

        res = qp.set_shots(qp.QNode(circ, make_dev(3, seed=5)), 20)()
        ref = qp.set_shots(qp.QNode(circ, qp.device("default.mixed", wires=3, seed=5)), 20)()
        for a, b in zip(res, ref):
            if isinstance(b, dict):
                assert isinstance(a, dict)
                assert set(a) <= {"0", "1", 1, -1} | set(b)
                assert sum(a.values()) == 20
            else:
                assert np.shape(a) == np.shape(b)
        assert set(np.unique(res[2])) <= {-1, 1}

    def test_shot_vector(self):
        def circ():
            _shot_prep()
            return qp.expval(qp.Z(0)), qp.probs(wires=[1])

        exact = qp.QNode(circ, qp.device("default.mixed", wires=3))()
        res = qp.set_shots(qp.QNode(circ, make_dev(3, seed=1)), [(20000, 2), 30000])()
        assert len(res) == 3
        for r in res:
            assert_all_close(r, exact, atol=0.03)

    def test_seed_reproducible(self):
        circ = lambda: (_shot_prep(), qp.sample(wires=[0, 1, 2]))[1]
        a = qp.set_shots(qp.QNode(circ, make_dev(3, seed=7)), 50)()
        b = qp.set_shots(qp.QNode(circ, make_dev(3, seed=7)), 50)()
        c = qp.set_shots(qp.QNode(circ, make_dev(3, seed=8)), 50)()
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    @pytest.mark.parametrize("mode", ["transform", "native"])
    def test_noise_model_with_shots(self, mode):
        nm = qp.NoiseModel(
            {qp.noise.op_eq("CNOT"): qp.noise.partial_wires(qp.DepolarizingChannel, 0.1)},
            meas_map={qp.noise.meas_eq(qp.expval): qp.noise.partial_wires(qp.BitFlip, 0.1)},
        )

        def circ():
            qp.H(0)
            qp.CNOT([0, 1])
            return qp.expval(qp.Z(0) @ qp.Z(1)), qp.probs(wires=[0, 1])

        exact = qp.QNode(circ, make_dev(2, noise_model=nm, noise_mode=mode))()
        res = qp.set_shots(
            qp.QNode(circ, make_dev(2, noise_model=nm, noise_mode=mode, seed=3)), 40000
        )()
        assert_all_close(res, exact, atol=0.03)


# ---------------------------------------------------------------------------
# Mid-circuit measurements with noise
# ---------------------------------------------------------------------------

MCM_NOISE = qp.NoiseModel(
    {
        qp.noise.op_in(["RX", "CNOT"]): qp.noise.partial_wires(qp.DepolarizingChannel, 0.1),
        qp.noise.op_eq("Hadamard"): qp.noise.partial_wires(qp.AmplitudeDamping, 0.3),
    }
)


def _mcm_circ():
    qp.RY(1.1, 0)
    qp.CNOT([0, 1])
    m0 = qp.measure(0)
    qp.cond(m0, qp.RX)(0.7, 2)
    qp.cond(~m0, qp.Hadamard)(1)
    return qp.expval(qp.Z(2)), qp.probs(wires=[1, 2]), qp.expval(m0)


def _mcm_exact():
    """NumPy reference: conditional gates get conditional noise."""
    n = 3

    def full(mat, wires):
        return qp.math.expand_matrix(mat, wires, wire_order=range(n))

    def kraus(rho, ks, wires):
        return sum(full(k, wires) @ rho @ full(k, wires).conj().T for k in ks)

    def gate(rho, op):
        u = full(qp.matrix(op), op.wires)
        return u @ rho @ u.conj().T

    rho = np.zeros((8, 8), complex)
    rho[0, 0] = 1
    rho = gate(rho, qp.RY(1.1, 0))
    rho = gate(rho, qp.CNOT([0, 1]))
    for w in (0, 1):
        rho = kraus(rho, qp.DepolarizingChannel(0.1, w).kraus_matrices(), [w])
    p0 = full(np.diag([1, 0]), [0])
    p1 = full(np.diag([0, 1]), [0])
    r1 = p1 @ rho @ p1
    r0 = p0 @ rho @ p0
    r1 = gate(r1, qp.RX(0.7, 2))
    r1 = kraus(r1, qp.DepolarizingChannel(0.1, 2).kraus_matrices(), [2])
    r0 = gate(r0, qp.Hadamard(1))
    r0 = kraus(r0, qp.AmplitudeDamping(0.3, 1).kraus_matrices(), [1])
    rf = r0 + r1
    ez2 = np.trace(full(np.diag([1, -1]), [2]) @ rf).real
    probs = np.real(np.diag(rf)).reshape(2, 2, 2).sum(axis=0).ravel()
    return ez2, probs, np.trace(r1).real


class TestMidCircuitMeasurements:
    @pytest.mark.parametrize("wires", [5, None])
    @pytest.mark.parametrize("mode", ["transform", "native"])
    def test_deferred(self, mode, wires):
        dev = make_dev(wires, noise_model=MCM_NOISE, noise_mode=mode)
        res = qp.QNode(_mcm_circ, dev, mcm_method="deferred")()
        assert_all_close(res, _mcm_exact(), atol=1e-10)

    @pytest.mark.parametrize("mode", ["transform", "native"])
    def test_one_shot(self, mode):
        dev = make_dev(3, noise_model=MCM_NOISE, noise_mode=mode, seed=4)
        res = qp.set_shots(qp.QNode(_mcm_circ, dev, mcm_method="one-shot"), 6000)()
        assert_all_close(res, _mcm_exact(), atol=0.04)

    def test_reset_and_postselect(self):
        def circ():
            qp.H(0)
            qp.CNOT([0, 1])
            m = qp.measure(0, reset=True)
            qp.cond(m, qp.X)(2)
            return qp.probs(wires=[0, 1, 2])

        # the deferred measurement needs one extra wire
        res = qp.QNode(circ, make_dev(4), mcm_method="deferred")()
        expected = np.zeros(8)
        expected[0b000] = 0.5
        expected[0b011] = 0.5
        assert np.allclose(res, expected)

        def circ_ps():
            qp.RY(1.0, 0)
            qp.CNOT([0, 1])
            qp.measure(0, postselect=1)
            return qp.expval(qp.Z(1))

        # like lightning.qubit, deferred measurements do not postselect
        with pytest.raises(ValueError, match="Postselection is not allowed"):
            qp.QNode(circ_ps, make_dev(3), mcm_method="deferred")()
        qnode = qp.QNode(
            circ_ps, make_dev(2, seed=2), mcm_method="one-shot", postselect_mode="hw-like"
        )
        assert np.allclose(qp.set_shots(qnode, 200)(), -1)
        with pytest.raises(DeviceError, match="fill-shots"):
            qp.set_shots(
                qp.QNode(circ_ps, make_dev(2), mcm_method="one-shot", postselect_mode="fill-shots"),
                10,
            )()

    def test_mcm_requires_shots_or_deferred(self):
        tape = qp.tape.QuantumScript([qp.H(0), qp.ops.MidMeasure(0)], [qp.expval(qp.Z(0))])
        with pytest.raises(DeviceError, match="mid-circuit measurements with shots"):
            make_dev(1).execute([tape])


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


class TestGradients:
    @pytest.mark.parametrize("method", ["parameter-shift", "finite-diff"])
    def test_noisy_gradient(self, method):
        def f(x):
            qp.RX(x, 0)
            qp.CNOT([0, 1])
            qp.Hadamard(1)
            return qp.expval(qp.Z(0) @ qp.Z(1))

        x = pnp.array(0.4, requires_grad=True)
        kwargs = {"gradient_kwargs": {"h": 1e-6}} if method == "finite-diff" else {}
        dev = make_dev(2, noise_model=MCM_NOISE)
        g = qp.grad(qp.QNode(f, dev, diff_method=method, **kwargs))(x)
        ref_dev = qp.noise.add_noise(qp.device("default.mixed", wires=2), MCM_NOISE)
        g_ref = qp.grad(qp.QNode(f, ref_dev, diff_method="backprop"))(x)
        assert np.allclose(g, g_ref, atol=1e-8 if method == "parameter-shift" else 1e-5)

    def test_jax_parameter_shift(self):
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)

        def f(x):
            qp.RX(x, 0)
            qp.AmplitudeDamping(0.2, 0)
            return qp.expval(qp.Z(0))

        g = jax.grad(qp.QNode(f, make_dev(1), diff_method="parameter-shift", interface="jax"))(0.4)
        # <Z> = 1 - (1 - p)(1 - cos x)  ->  d/dx = -(1 - p) sin x
        assert np.allclose(g, -0.8 * np.sin(0.4), atol=1e-8)


# ---------------------------------------------------------------------------
# Larger system (10 qubits, compared with default.mixed)
# ---------------------------------------------------------------------------


def test_ten_qubits_noisy():
    n = 10
    nm = qp.NoiseModel(
        {
            qp.noise.op_eq("CNOT"): qp.noise.partial_wires(qp.DepolarizingChannel, 0.02),
            qp.noise.op_eq("RY"): qp.noise.partial_wires(qp.AmplitudeDamping, 0.05),
        }
    )

    def circ():
        for layer in range(2):
            for i in range(n):
                qp.RY(0.3 + 0.1 * i + layer, i)
            for i in range(layer % 2, n - 1, 2):
                qp.CNOT([i, i + 1])
        return qp.expval(qp.Z(0) @ qp.Z(n - 1)), qp.probs(wires=[3, 4]), qp.expval(qp.X(5))

    res = qp.QNode(circ, make_dev(n, noise_model=nm, noise_mode="native"))()
    ref = qp.QNode(circ, qp.noise.add_noise(qp.device("default.mixed", wires=n), nm))()
    assert_all_close(res, ref, atol=1e-9)
