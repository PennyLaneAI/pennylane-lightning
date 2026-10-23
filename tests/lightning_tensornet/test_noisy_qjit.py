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
Catalyst (``qp.qjit``) tests for ``lightning.tensornet``: the compiled program
runs on the Catalyst runtime device ``LightningTensornetSimulator``, with the
noise model handed to the runtime through ``device_kwargs``.
"""

# pylint: disable=missing-function-docstring
import numpy as np
import pennylane as qp
import pytest
from conftest import LightningDevice, device_name
from pennylane import numpy as pnp

if device_name != "lightning.tensornet":
    pytest.skip("Tests for lightning.tensornet.", allow_module_level=True)

if LightningDevice._ops_module() is None:  # pylint: disable=protected-access
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

catalyst = pytest.importorskip("catalyst")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

try:
    LightningDevice.get_c_interface()
except RuntimeError:  # pragma: no cover
    pytest.skip("The Catalyst runtime library was not built.", allow_module_level=True)

pytestmark = pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")

NM = qp.NoiseModel(
    {
        qp.noise.op_eq("CNOT"): qp.noise.partial_wires(qp.DepolarizingChannel, 0.05),
        qp.noise.op_in(["RX", "RY"])
        & qp.noise.wires_in([0, 2]): qp.noise.partial_wires(qp.AmplitudeDamping, 0.1),
        qp.noise.op_eq("Hadamard"): qp.noise.partial_wires(qp.PhaseDamping, 0.2),
    },
    meas_map={qp.noise.meas_eq(qp.expval): qp.noise.partial_wires(qp.BitFlip, 0.03)},
)

MCM_NM = qp.NoiseModel(
    {
        qp.noise.op_in(["RX", "CNOT"]): qp.noise.partial_wires(qp.DepolarizingChannel, 0.1),
        qp.noise.op_eq("Hadamard"): qp.noise.partial_wires(qp.AmplitudeDamping, 0.3),
    }
)


def make_dev(wires, **kwargs):
    return qp.device(device_name, wires=wires, **kwargs)


def assert_all_close(res, ref, atol):
    assert len(res) == len(ref)
    for a, b in zip(res, ref):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=atol), (a, b)


def circ(x):
    qp.Hadamard(0)
    qp.RX(x, 0)
    qp.CNOT([0, 1])
    qp.RY(0.3 * x, 2)
    qp.CNOT([1, 2])
    qp.IsingZZ(0.4, [2, 3])
    qp.adjoint(qp.S(3))
    qp.ctrl(qp.RZ(0.5, 3), 0)
    qp.CRX(0.2, [3, 1])
    qp.PauliRot(0.3, "XIY", [0, 1, 3])
    qp.SWAP([0, 3])
    return (
        qp.expval(qp.Z(0)),
        qp.expval(qp.X(1) @ qp.Y(3)),
        qp.probs(wires=[1, 2]),
        qp.var(qp.Z(2)),
    )


def mcm_circ():
    qp.RY(1.1, 0)
    qp.CNOT([0, 1])
    m0 = qp.measure(0)
    qp.cond(m0, qp.RX)(0.7, 2)
    qp.cond(~m0, qp.Hadamard)(1)
    return qp.expval(qp.Z(2)), qp.probs(wires=[1, 2]), qp.expval(m0)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"noise_model": NM, "noise_mode": "native"}, {"noise_model": NM}],
    ids=["clean", "native", "transform"],
)
def test_analytic(kwargs):
    """qjit results match the Python device (a transform-mode device is run
    with the native noise model under qjit)."""
    dev = make_dev(4, **kwargs)
    jit = qp.qjit(qp.QNode(circ, dev))
    for x in (0.6, 1.1):
        assert_all_close(jit(x), qp.QNode(circ, dev)(x), atol=1e-10)


def test_noise_reaches_runtime():
    dev = make_dev(4, noise_model=NM, noise_mode="native")
    jit = qp.qjit(qp.QNode(circ, dev))
    clean = qp.qjit(qp.QNode(circ, make_dev(4)))(0.6)
    noisy = jit(0.6)
    assert not np.allclose(clean[0], noisy[0])
    spec = dev.device_kwargs["noise_model"]
    assert spec.startswith("LNM1;")
    assert "LNM1" in jit.mlir


def test_explicit_channel_not_supported():
    """Catalyst has no channel instruction: noise under qjit comes from the
    device noise model only."""

    @qp.qjit
    @qp.qnode(make_dev(1))
    def f(x):
        qp.RX(x, 0)
        qp.BitFlip(0.1, 0)
        return qp.expval(qp.Z(0))

    with pytest.raises(catalyst.utils.exceptions.CompileError, match="BitFlip"):
        f(0.3)


def test_shots():
    def circ_s(x):
        qp.Hadamard(0)
        qp.RX(x, 0)
        qp.CNOT([0, 1])
        qp.RY(0.3 * x, 2)
        qp.CNOT([1, 2])
        return qp.expval(qp.Z(0)), qp.probs(wires=[1, 2]), qp.sample(wires=[0, 1])

    def exact(x):
        qp.Hadamard(0)
        qp.RX(x, 0)
        qp.CNOT([0, 1])
        qp.RY(0.3 * x, 2)
        qp.CNOT([1, 2])
        return qp.expval(qp.Z(0)), qp.probs(wires=[1, 2]), qp.probs(wires=[0, 1])

    dev = make_dev(3, noise_model=NM, noise_mode="native")
    ref = qp.QNode(exact, dev)(0.6)
    res = qp.qjit(qp.set_shots(qp.QNode(circ_s, dev), 40000), seed=3)(0.6)
    samples = np.asarray(res[2])
    assert samples.shape == (40000, 2)
    p01 = np.bincount(samples[:, 0] * 2 + samples[:, 1], minlength=4) / len(samples)
    assert np.allclose(res[0], ref[0], atol=0.03)
    assert np.allclose(res[1], ref[1], atol=0.015)
    assert np.allclose(p01, ref[2], atol=0.015)


def test_seeded_qjit_is_reproducible():
    dev = make_dev(2, noise_model=MCM_NM)

    @qp.qjit(seed=17)
    @qp.set_shots(30)
    @qp.qnode(dev)
    def f():
        qp.Hadamard(0)
        qp.CNOT([0, 1])
        return qp.sample(wires=[0, 1])

    assert np.array_equal(np.asarray(f()), np.asarray(f()))


def test_mcm_one_shot():
    exact = qp.QNode(mcm_circ, make_dev(5, noise_model=MCM_NM), mcm_method="deferred")()
    dev = make_dev(3, noise_model=MCM_NM)
    res = qp.qjit(qp.set_shots(qp.QNode(mcm_circ, dev, mcm_method="one-shot"), 20000), seed=37)()
    assert_all_close(res, exact, atol=0.025)


def test_mcm_device_trajectories():
    """mcm_method='device': each call of a 1-shot program samples one trajectory
    (noise channels and mid-circuit measurements are sampled by the runtime)."""
    exact = qp.QNode(mcm_circ, make_dev(5, noise_model=MCM_NM), mcm_method="deferred")()[1]
    dev = make_dev(3, noise_model=MCM_NM)

    @qp.qjit
    @qp.set_shots(1)
    @qp.qnode(dev, mcm_method="device")
    def traj():
        qp.RY(1.1, 0)
        qp.CNOT([0, 1])
        m0 = qp.measure(0)
        qp.cond(m0, qp.RX)(0.7, 2)
        qp.cond(~m0, qp.Hadamard)(1)
        return qp.sample(wires=[1, 2])

    s = np.array([np.asarray(traj()).ravel() for _ in range(3000)])
    p = np.bincount(s[:, 0] * 2 + s[:, 1], minlength=4) / len(s)
    assert np.allclose(p, exact, atol=0.035)


def _grad_circ(x):
    qp.RX(x, 0)
    qp.CNOT([0, 1])
    qp.Hadamard(1)
    return qp.expval(qp.Z(0) @ qp.Z(1))


def test_grad_finite_diff():
    dev = make_dev(2, noise_model=MCM_NM)
    g = qp.qjit(catalyst.grad(qp.QNode(_grad_circ, dev), method="fd", h=1e-7))(0.4)
    ref = qp.grad(qp.QNode(_grad_circ, dev, diff_method="parameter-shift"))(
        pnp.array(0.4, requires_grad=True)
    )
    assert np.allclose(g, ref, atol=1e-5)


def test_jacobian_parameter_shift():
    dev = make_dev(2, noise_model=MCM_NM)

    def f(x):
        qp.RX(x[0], 0)
        qp.RY(x[1], 1)
        qp.CNOT([0, 1])
        return qp.expval(qp.Z(1)), qp.probs(wires=[0])

    jac = qp.qjit(qp.jacobian(qp.QNode(f, dev, diff_method="parameter-shift")))(
        jax.numpy.array([0.3, 0.5])
    )

    def stacked(x):
        return qp.math.hstack(qp.QNode(f, dev, diff_method="parameter-shift")(x))

    ref = qp.jacobian(stacked)(pnp.array([0.3, 0.5], requires_grad=True))
    jac = np.vstack([np.atleast_2d(np.asarray(j)) for j in jac])
    assert np.allclose(jac, ref, atol=1e-8)


def test_dynamic_qubit_allocation():
    """``qp.allocate`` under program capture: the runtime appends a qubit."""

    def ref(x):
        qp.RX(x, 0)
        qp.CNOT([0, 2])
        qp.Hadamard(2)
        qp.CNOT([2, 1])
        return qp.expval(qp.Z(1)), qp.probs(wires=[0, 1])

    expected = qp.QNode(ref, make_dev(3, noise_model=MCM_NM, noise_mode="native"))(0.8)

    qp.capture.enable()
    try:
        dev = make_dev(2, noise_model=MCM_NM)

        @qp.qjit
        @qp.qnode(dev)
        def dyn(x):
            qp.RX(x, 0)
            q = qp.allocate(1)
            qp.CNOT([0, q[0]])
            qp.Hadamard(q[0])
            qp.CNOT([q[0], 1])
            qp.deallocate(q)
            return qp.expval(qp.Z(1)), qp.probs(wires=[0, 1])

        res = dyn(0.8)
    finally:
        qp.capture.disable()
    assert_all_close(res, expected, atol=1e-10)
