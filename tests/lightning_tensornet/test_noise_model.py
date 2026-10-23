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
Tests for the lowering of PennyLane noise models to the native ``LNM1``
format (``pennylane_lightning.lightning_base._noise``) and for the C++ parser.
"""

# pylint: disable=missing-function-docstring
import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice, device_name

if device_name != "lightning.tensornet":
    pytest.skip("Tests for lightning.tensornet.", allow_module_level=True)

if LightningDevice._ops_module() is None:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

# pylint: disable=wrong-import-position
from pennylane_lightning.lightning_base._noise import (
    CHANNEL_NAMES,
    READOUT_KINDS,
    NativeNoiseError,
    NoiseModelConverter,
    gates_from_capabilities,
    is_native_spec,
    to_native_noise_model,
    validate_native_spec,
)

pytestmark = pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")

CAPS = LightningDevice.capabilities
OPS = LightningDevice._ops_module()  # pylint: disable=protected-access


def convert(nm, wires=range(3), **kwargs):
    return to_native_noise_model(nm, list(wires), CAPS, **kwargs)


def parse_channels(spec):
    """Kraus channels of a serialized model: list of (nq, [matrices])."""
    out = []
    for part in spec.split(";"):
        if part.startswith("K="):
            fields = part[2:].split("/")
            nq, nk = int(fields[0]), int(fields[1])
            vals = np.array([float(v) for v in fields[2:]])
            z = vals[0::2] + 1j * vals[1::2]
            d = 2**nq
            out.append((nq, list(z.reshape(nk, d, d))))
    return out


class TestSpecStrings:
    def test_is_native_spec(self):
        assert is_native_spec("LNM1;")
        assert not is_native_spec("LNM2;")
        assert not is_native_spec(None)

    @pytest.mark.parametrize(
        "bad", ["", "foo", "LNM1;G=a,b", "LNM1; ", "LNM1;{", "LNM1;a:b", "LNM1'"]
    )
    def test_validate_native_spec(self, bad):
        with pytest.raises(NativeNoiseError):
            validate_native_spec(bad)

    def test_passthrough(self):
        assert to_native_noise_model(None, [0], CAPS) == ""
        assert to_native_noise_model("LNM1;", [0], CAPS) == "LNM1;"

    def test_not_a_noise_model(self):
        with pytest.raises(NativeNoiseError, match="NoiseModel"):
            NoiseModelConverter(object(), [0], {})

    def test_gates_from_capabilities(self):
        gates = gates_from_capabilities(CAPS)
        for name in ("CNOT", "RX", "PauliRot", "QubitUnitary", *CHANNEL_NAMES):
            assert name in gates
        assert "controllable" in gates["RX"] and "invertible" in gates["RX"]
        assert set(READOUT_KINDS) == {"expval", "var", "probs", "sample", "counts"}


class TestConversion:
    def test_kraus_full_precision(self):
        p = 0.1234567890123
        nm = qp.NoiseModel({qp.noise.op_eq("CNOT"): qp.noise.partial_wires(qp.BitFlip, p)})
        spec = convert(nm)
        ((nq, kraus),) = parse_channels(spec)
        assert nq == 1
        ref = qp.BitFlip(p, 0).kraus_matrices()
        assert all(np.array_equal(a, b) for a, b in zip(kraus, ref))
        assert any(part.startswith("G=CNOT/") for part in spec.split(";"))

    def test_channels_are_shared(self):
        nm = qp.NoiseModel(
            {
                qp.noise.op_eq("CNOT"): qp.noise.partial_wires(qp.DepolarizingChannel, 0.1),
                qp.noise.op_eq("CZ"): qp.noise.partial_wires(qp.DepolarizingChannel, 0.1),
                qp.noise.op_eq("RX"): qp.noise.partial_wires(qp.PhaseFlip, 0.2),
            }
        )
        assert len(parse_channels(convert(nm))) == 2

    def test_all_channel_types_serialize(self):
        # built outside the noise function (operators created inside it are queued as noise)
        k_flip = qp.PhaseFlip.compute_kraus_matrices(0.1)
        k_ad = qp.AmplitudeDamping.compute_kraus_matrices(0.4)[::-1]

        def noise(op, **_):
            w = op.wires[0]
            qp.AmplitudeDamping(0.1, w)
            qp.GeneralizedAmplitudeDamping(0.1, 0.2, w)
            qp.PhaseDamping(0.1, w)
            qp.DepolarizingChannel(0.1, w)
            qp.BitFlip(0.1, w)
            qp.PhaseFlip(0.1, w)
            qp.PauliError("Y", 0.1, w)
            qp.ResetError(0.1, 0.1, w)
            qp.ThermalRelaxationError(0.1, 100, 80, 20, w)
            # same Kraus operators as PhaseFlip(0.1): stored once
            qp.QubitChannel(k_flip, w)
            qp.QubitChannel(k_ad, w)

        spec = convert(qp.NoiseModel({qp.noise.op_eq("Hadamard"): noise}))
        validate_native_spec(spec)
        channels = parse_channels(spec)
        assert len(channels) == 10
        for _, kraus in channels:
            s = sum(k.conj().T @ k for k in kraus)
            assert np.allclose(s, np.eye(2), atol=1e-12)
        nat = getattr(OPS, "NativeNoiseModelC128")(spec)
        assert nat.num_channels == 10 and not nat.empty() and not nat.has_readout_noise

    def test_parameter_dependent_noise(self):
        def noise(op, **_):
            qp.DepolarizingChannel(abs(op.parameters[0]) / 10, op.wires)

        with pytest.raises(NativeNoiseError, match="depends on the gate parameters"):
            convert(qp.NoiseModel({qp.noise.op_eq("RX"): noise}))

    def test_noise_outside_device_wires(self):
        def noise(op, **_):  # pylint: disable=unused-argument
            qp.BitFlip(0.1, wires=99)

        with pytest.raises(NativeNoiseError, match="not a device wire"):
            convert(qp.NoiseModel({qp.noise.op_eq("Hadamard"): noise}))

    def test_wire_labels(self):
        nm = qp.NoiseModel({qp.noise.wires_in(["b"]): qp.noise.partial_wires(qp.PhaseFlip, 0.3)})
        dev = qp.device(device_name, wires=["a", "b", "c"], noise_model=nm, noise_mode="native")

        def circ():
            qp.H("a")
            qp.H("b")
            qp.H("c")
            return qp.expval(qp.X("a")), qp.expval(qp.X("b")), qp.expval(qp.X("c"))

        assert np.allclose(qp.QNode(circ, dev)(), [1, 0.4, 1])

    def test_bare_wires_in_condition(self):
        """wires_in also matches a wire-less GlobalPhase; the conversion must not fail."""
        nm = qp.NoiseModel({qp.noise.wires_in([1]): qp.noise.partial_wires(qp.PhaseFlip, 0.3)})
        dev = qp.device(device_name, wires=3, noise_model=nm, noise_mode="native")

        def circ():
            qp.H(0)
            qp.H(1)
            return qp.expval(qp.X(0)), qp.expval(qp.X(1))

        assert np.allclose(qp.QNode(circ, dev)(), [1, 0.4])

    def test_failing_noise_function(self):
        def noise(op, **_):
            raise RuntimeError("boom")

        with pytest.raises(NativeNoiseError, match="boom"):
            convert(qp.NoiseModel({qp.noise.op_eq("Hadamard"): noise}))

    def test_wires_none_uses_condition_wires(self):
        nm = qp.NoiseModel({qp.noise.wires_eq(11): qp.noise.partial_wires(qp.BitFlip, 0.2)})
        spec = to_native_noise_model(nm, None, CAPS)
        assert "11" in spec
        dev = qp.device(device_name, noise_model=nm, noise_mode="native")

        def circ():
            qp.X(11)
            qp.X(0)
            return qp.expval(qp.Z(11)), qp.expval(qp.Z(0))

        assert np.allclose(qp.QNode(circ, dev)(), [-0.6, -1])

    def test_observable_dependent_readout_warns(self):
        @qp.BooleanFn
        def is_x(mp):
            return getattr(mp, "obs", None) is not None and mp.obs.name == "PauliX"

        nm = qp.NoiseModel({}, meas_map={is_x: qp.noise.partial_wires(qp.BitFlip, 0.1)})
        with pytest.warns(UserWarning, match="depends on the observable"):
            convert(nm)

    def test_readout_spec(self):
        nm = qp.NoiseModel(
            {},
            meas_map={qp.noise.meas_eq(qp.probs): qp.noise.partial_wires(qp.BitFlip, 0.1)},
        )
        spec = convert(nm)
        assert any(part.startswith("M=probs/") for part in spec.split(";"))
        assert not any(part.startswith("M=expval/") for part in spec.split(";"))
        nat = getattr(OPS, "NativeNoiseModelC64")(spec)
        assert nat.has_readout_noise

    @pytest.mark.parametrize(
        "spec",
        ["LNM1;K=1/1/1/0", "LNM1;K=1/2/1/0/0/0/0/0/1/0;G=CNOT/*2/a0@r0", "LNM1;G=CNOT/*2/a0@r0"],
    )
    def test_cpp_parser_rejects_malformed(self, spec):
        with pytest.raises(NativeNoiseError):
            qp.device(device_name, wires=2, noise_model=spec)

    def test_hand_written_spec(self):
        """A bit flip written by hand (a = sqrt(1-p), b = sqrt(p), 17 digits)."""
        p = 0.2
        a, b = repr(float(np.sqrt(1 - p))), repr(float(np.sqrt(p)))
        spec = f"LNM1;K=1/2/{a}/0/0/0/0/0/{a}/0/0/0/{b}/0/{b}/0/0/0;G=Hadamard/*1/a0@r0"
        dev = qp.device(device_name, wires=1, noise_model=spec)
        res = qp.QNode(lambda: (qp.H(0), qp.expval(qp.X(0)))[1], dev)()
        # X commutes with a bit flip
        assert np.isclose(res, 1.0)
        res = qp.QNode(lambda: (qp.H(0), qp.H(0), qp.expval(qp.Z(0)))[-1], dev)()
        # H, flip, H, flip: <Z> = (1 - 2p) * <X after first flip> = (1 - 2p) * 1
        assert np.isclose(res, 1 - 2 * p)
