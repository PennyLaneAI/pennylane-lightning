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
r"""
Python device logic for the Kokkos noisy simulators, currently
``lightning.tensornet`` (matrix product states / operators).

A backend exposes the common C++ state API through nanobind (see
``core/noise/bindings/NoisyBindings.hpp``), so the execution, measurement and
noise logic lives here; the device classes only provide the backend module,
the state class and the backend-specific measurements.

Noise models
------------
A device accepts ``noise_model=`` either as a :class:`~pennylane.NoiseModel` or
as a serialized native model (an ``"LNM1;..."`` string, see
:mod:`pennylane_lightning.lightning_base._noise`). Two execution modes exist:

* ``noise_mode="transform"`` (default for :class:`~pennylane.NoiseModel`): the
  model is inserted into every tape by the device preprocessing, with the same
  per-operation rules as :func:`pennylane.noise.add_noise`, after the circuit
  has been decomposed to the device gate set.
* ``noise_mode="native"`` (default and only choice for native strings): the
  model is compiled to a native rule table and applied by the C++ backend
  while the circuit runs. This is the representation handed to the Catalyst
  runtime under :func:`~pennylane.qjit` (through ``device_kwargs``), whatever
  the Python-side mode is.

In both modes the noise is not applied to state preparation, to mid-circuit
measurements, or to measurements other than expectation values, variances,
probabilities, samples and counts; noise attached to a gate inside
``qml.cond`` is conditioned on the same classical value as the gate.
"""

# pylint: disable=too-many-lines
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial
from numbers import Number
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pennylane as qp
from pennylane.core.qscript import make_qscript
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.preprocess import (
    decompose,
    device_resolve_dynamic_wires,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.devices.qubit.sampling import _group_measurements
from pennylane.exceptions import DeviceError
from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    DensityMatrixMP,
    ExpectationMP,
    MeasurementProcess,
    MutualInfoMP,
    ProbabilityMP,
    PurityMP,
    SampleMP,
    ShadowExpvalMP,
    Shots,
    StateMP,
    VarianceMP,
    VnEntropyMP,
)
from pennylane.operation import StatePrepBase
from pennylane.ops import Conditional, MidMeasure, SProd, Sum
from pennylane.tape import QuantumScript
from pennylane.transforms import defer_measurements, dynamic_one_shot
from pennylane.typing import Result, ResultBatch
from pennylane.wires import Wires

from pennylane_lightning.lightning_base._noise import (
    NAMED_CONTROLLED_GATES,
    NativeNoiseError,
    is_native_spec,
    runtime_key,
    to_native_noise_model,
    validate_native_spec,
)

Result_or_ResultBatch = Union[Result, ResultBatch]

NOISE_MODES = ("transform", "native")

# Gates the C++ backends apply through a named kernel (no matrix is built).
# QubitUnitary and PauliRot have dedicated entry points.
_NAMED_KERNELS = frozenset(
    {
        "Identity",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "T",
        "SX",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "PhaseShift",
        "GlobalPhase",
        "CNOT",
        "CY",
        "CZ",
        "SWAP",
        "CSWAP",
        "Toffoli",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "ControlledPhaseShift",
        "IsingXX",
        "IsingXY",
        "IsingYY",
        "IsingZZ",
        "MultiRZ",
        "PSWAP",
        "SingleExcitation",
        "SingleExcitationMinus",
        "SingleExcitationPlus",
        "DoubleExcitation",
        "DoubleExcitationMinus",
        "DoubleExcitationPlus",
    }
)

# Base gates that the backends can control with a named kernel.
_CONTROLLABLE_KERNELS = frozenset(
    {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "T",
        "SX",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "PhaseShift",
        "GlobalPhase",
        "SWAP",
        "IsingXX",
        "IsingXY",
        "IsingYY",
        "IsingZZ",
        "MultiRZ",
        "PSWAP",
        "SingleExcitation",
        "SingleExcitationMinus",
        "SingleExcitationPlus",
        "DoubleExcitation",
        "DoubleExcitationMinus",
        "DoubleExcitationPlus",
    }
)

_MEASUREMENT_KIND = (
    (ExpectationMP, "expval"),
    (VarianceMP, "var"),
    (ProbabilityMP, "probs"),
    (SampleMP, "sample"),
    (CountsMP, "counts"),
)


def readout_kind(mp: MeasurementProcess) -> Optional[str]:
    """Native readout-noise kind of a measurement, or ``None`` if readout noise
    does not apply to it (state-like measurements, measurements of
    mid-circuit measurement values)."""
    if getattr(mp, "mv", None) is not None:
        return None
    for cls, kind in _MEASUREMENT_KIND:
        if isinstance(mp, cls):
            return kind
    return None


def _is_state_prep(op) -> bool:
    return isinstance(op, (StatePrepBase, qp.QubitDensityMatrix))


def _unwrap(x):
    return qp.math.unwrap(x) if not isinstance(x, Number) else x


def _kokkos_dict(kokkos_args) -> dict:
    """Kokkos settings as a plain dict (keys of ``Kokkos::InitializationSettings``)."""
    if kokkos_args is None:
        return {}
    if isinstance(kokkos_args, dict):
        return dict(kokkos_args)
    raise TypeError(
        "kokkos_args must be a dict with Kokkos::InitializationSettings keys "
        "(num_threads, device_id, map_device_id_by, disable_warnings, "
        "print_configuration, tune_internals, tools_libs, tools_help, tools_args)."
    )


# ---------------------------------------------------------------------------
# Preprocessing transforms
# ---------------------------------------------------------------------------


def _simplify_adjoint(op):
    if isinstance(op, qp.ops.Adjoint):
        new = qp.adjoint(_simplify_adjoint(op.base), lazy=False)
        return new
    if isinstance(op, qp.ops.Controlled) and op.name not in NAMED_CONTROLLED_GATES:
        if isinstance(op.base, qp.ops.Adjoint) and not isinstance(op, qp.ControlledQubitUnitary):
            base = _simplify_adjoint(op.base)
            if base is not op.base:
                return qp.ctrl(
                    base,
                    control=op.control_wires,
                    control_values=op.control_values,
                    work_wires=op.work_wires,
                )
    return op


@qp.transform
def simplify_adjoints(tape: QuantumScript):
    """Replace ``Adjoint(op)`` by ``qml.adjoint(op, lazy=False)`` so that noise
    rules see the same operation in Python, in the native rule tables and
    under qjit (for example ``Adjoint(RX(x))`` becomes ``RX(-x)``)."""
    new_ops = []
    changed = False
    for op in tape.operations:
        new = _simplify_adjoint(op) if not isinstance(op, (Conditional, MidMeasure)) else op
        if isinstance(op, Conditional):
            base = _simplify_adjoint(op.base)
            if base is not op.base:
                new = Conditional(op.meas_val, base)
        changed |= new is not op
        new_ops.append(new)
    if not changed:
        return [tape], lambda res: res[0]
    return [tape.copy(operations=new_ops)], lambda res: res[0]


@qp.transform
def expand_wireless_measurements(tape: QuantumScript, device_wires=None):
    """Give explicit wires to ``probs()``, ``sample()`` and ``counts()`` without
    wires, so that readout-noise rules see the wires that are measured."""
    wires = device_wires if device_wires is not None else tape.wires
    new_mps = []
    changed = False
    for mp in tape.measurements:
        if (
            isinstance(mp, (ProbabilityMP, SampleMP, CountsMP))
            and mp.obs is None
            and getattr(mp, "mv", None) is None
            and len(mp.wires) == 0
            and len(wires) > 0
        ):
            if isinstance(mp, ProbabilityMP):
                mp = qp.probs(wires=wires)
            elif isinstance(mp, CountsMP):
                mp = qp.counts(wires=wires, all_outcomes=mp.all_outcomes)
            else:
                mp = qp.sample(wires=wires)
            changed = True
        new_mps.append(mp)
    if not changed:
        return [tape], lambda res: res[0]
    return [tape.copy(measurements=new_mps)], lambda res: res[0]


def _noise_rules(model_map):
    rules = []
    for condition, noise in model_map.items():
        rules.append((lru_cache(maxsize=512)(condition), make_qscript(noise)))
    return rules


def _noisy_operation(op, rules, metadata) -> list:
    """Same per-operation rule as :func:`pennylane.noise.add_noise`."""
    curr = [op]
    for condition, noise in rules:
        if condition(op):
            noise_ops = noise(op, **metadata).operations
            if any(qp.equal(op, o) for o in noise_ops):
                idx = noise_ops.index(op)
                curr = noise_ops[:idx] + curr + noise_ops[idx + 1 :]
            else:
                curr.extend(noise_ops)
    return curr


def _split_results(result, n_mps: int, shots: Shots) -> list:
    """Result of one tape as ``[bin][measurement]``."""
    bins = result if shots.has_partitioned_shots else (result,)
    out = []
    for b in bins:
        out.append([b] if n_mps == 1 else list(b))
    return out


@qp.transform
def apply_noise_model(tape: QuantumScript, noise_model):  # pylint: disable=too-many-branches
    """Insert the noise of ``noise_model`` into a device-level tape.

    Gate noise follows :func:`pennylane.noise.add_noise` operation by operation;
    state preparation and mid-circuit measurements are left untouched and the
    noise of an operation inside ``qml.cond`` is conditioned on the same value.
    Measurements with different readout noise are split into separate tapes.
    """
    metadata = noise_model.metadata
    rules = _noise_rules(noise_model.model_map)
    new_ops = []
    for op in tape.operations:
        if _is_state_prep(op) or isinstance(op, MidMeasure):
            new_ops.append(op)
        elif isinstance(op, Conditional):
            for o in _noisy_operation(op.base, rules, metadata):
                new_ops.append(Conditional(op.meas_val, o))
        else:
            new_ops.extend(_noisy_operation(op, rules, metadata))

    if not noise_model.meas_map:
        return [tape.copy(operations=new_ops)], lambda res: res[0]

    meas_rules = _noise_rules(noise_model.meas_map)
    groups: list[tuple[list, list[int]]] = []
    for i, mp in enumerate(tape.measurements):
        readout = []
        if readout_kind(mp) is not None:
            for condition, noise in meas_rules:
                if condition(mp):
                    readout.extend(noise(mp, **metadata).operations)
        for ops, idxs in groups:
            if len(ops) == len(readout) and all(qp.equal(a, b) for a, b in zip(ops, readout)):
                idxs.append(i)
                break
        else:
            groups.append((readout, [i]))

    if len(groups) == 1:
        return [tape.copy(operations=new_ops + groups[0][0])], lambda res: res[0]

    tapes = [
        tape.copy(
            operations=new_ops + ops,
            measurements=[tape.measurements[i] for i in idxs],
        )
        for ops, idxs in groups
    ]
    n_total = len(tape.measurements)
    shots = tape.shots

    def postprocessing(results):
        n_bins = shots.num_copies if shots.has_partitioned_shots else 1
        per_bin = [[None] * n_total for _ in range(n_bins)]
        for (_, idxs), res in zip(groups, results):
            split = _split_results(res, len(idxs), shots)
            for b, vals in enumerate(split):
                for i, v in zip(idxs, vals):
                    per_bin[b][i] = v
        out = [tuple(v) if n_total > 1 else v[0] for v in per_bin]
        return tuple(out) if shots.has_partitioned_shots else out[0]

    return tapes, postprocessing


def controlled_kraus(kraus, control_values: Sequence[bool], dtype=np.complex128) -> list:
    """Kraus operators of a channel applied only if the control qubits are in
    ``|control_values>`` (control wires first):

    ``K'_0 = (I - P) (x) I + P (x) K_0`` and ``K'_i = P (x) K_i`` for ``i > 0``,
    with ``P`` the projector on the control values. This is trace preserving when
    the channel is.
    """
    nc = len(control_values)
    idx = int("".join("1" if v else "0" for v in control_values) or "0", 2)
    proj = np.zeros((2**nc, 2**nc))
    proj[idx, idx] = 1.0
    ks = [np.asarray(_unwrap(k)) for k in kraus]
    eye = np.eye(ks[0].shape[0])
    out = [np.kron(np.eye(2**nc) - proj, eye) + np.kron(proj, ks[0])]
    out += [np.kron(proj, k) for k in ks[1:]]
    return [np.ascontiguousarray(k, dtype=dtype) for k in out]


class NativeNoiseApplied(qp.operation.Operation):
    """Marker (no wires, no effect) recording that the native gate noise of a tape
    has already been inserted as explicit channels, so the backend must not apply
    it again. Used for mid-circuit measurements with ``mcm_method="deferred"``."""

    num_wires = None
    num_params = 0
    grad_method = None

    def __init__(self):
        super().__init__(wires=[])

    def decomposition(self):
        return []

    def adjoint(self):
        return NativeNoiseApplied()


def _gate_noise(native, op):
    """Native gate noise ``(before, after)`` of ``op`` (device-index wires)."""
    key, wires = runtime_key(op)
    return native.lookup_gate(key, [int(w) for w in wires])


@qp.transform
def insert_native_noise(tape: QuantumScript, device):
    """Insert the native gate noise of ``device`` as explicit channels.

    Only acts on tapes with mid-circuit measurements; it runs before
    :func:`~pennylane.defer_measurements` so that the noise follows the (conditional)
    gates of the circuit and not the operations introduced by the deferral. The
    readout noise is still applied by the backend.
    """
    native = device._native_model()  # pylint: disable=protected-access
    if native is None or not any(isinstance(op, MidMeasure) for op in tape.operations):
        return [tape], lambda res: res[0]
    wire_map = device._wire_map  # pylint: disable=protected-access
    to_label = {i: w for w, i in wire_map.items()} if wire_map else None

    def label(i):
        return to_label[i] if to_label is not None else i

    def index_op(op):
        if wire_map:
            return op.map_wires({w: wire_map[w] for w in op.wires})
        for w in op.wires:
            if not isinstance(w, (int, np.integer)) or w < 0:
                raise DeviceError(
                    f"{device.name} with noise_mode='native' and wires=None requires "
                    "non-negative integer wire labels; pass wires= to the device."
                )
        return op

    def channels(noise, meas_val):
        out = []
        for ch, wires in noise:
            with qp.QueuingManager.stop_recording():
                dim = 2 ** len(wires)
                kraus = [np.asarray(k, dtype=complex).reshape(dim, dim) for k in native.kraus(ch)]
                o = qp.QubitChannel(kraus, wires=[label(int(w)) for w in wires])
                out.append(Conditional(meas_val, o) if meas_val is not None else o)
        return out

    new_ops = [NativeNoiseApplied()]
    for op in tape.operations:
        base, meas_val = (op.base, op.meas_val) if isinstance(op, Conditional) else (op, None)
        if isinstance(op, MidMeasure) or _is_state_prep(base):
            new_ops.append(op)
            continue
        before, after = _gate_noise(native, index_op(base))
        new_ops.extend(channels(before, meas_val))
        new_ops.append(op)
        new_ops.extend(channels(after, meas_val))
    return [tape.copy(operations=new_ops)], lambda res: res[0]


def _stopping_condition(op, supports_operation: Callable, allow_mcms: bool) -> bool:
    if isinstance(op, MidMeasure):
        return allow_mcms
    if isinstance(op, (qp.PauliRot, NativeNoiseApplied)):
        return True
    if isinstance(op, qp.ops.Controlled) and isinstance(op.base, qp.operation.Channel):
        return True
    return supports_operation(op.name)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


class LightningNoisyBase(Device):
    """Base class of the Kokkos noisy simulators.

    Subclasses define ``config_filepath``, :attr:`name`, ``_ops_module()``,
    ``_state_class_name`` and the backend hooks ``_state_measurement``,
    ``_apply_density_matrix`` and ``_expval_sparse``.

    Args:
        wires (int | Iterable | None): device wires. ``None`` allocates the wires
            of each circuit.
        shots (int | list | None): default shots (deprecated in PennyLane in
            favour of ``qml.set_shots``).
        seed (str | None | int | array_like | SeedSequence | BitGenerator | Generator):
            seed for ``numpy.random.default_rng``; ``"global"`` draws the seed from
            NumPy's global generator.
        c_dtype: ``np.complex64`` or ``np.complex128``.
        noise_model (~pennylane.NoiseModel | str | None): noise model, or a
            serialized native model (``"LNM1;..."``).
        noise_mode (str | None): ``"transform"`` or ``"native"``; see the module
            documentation. ``None`` selects ``"transform"`` for a
            :class:`~pennylane.NoiseModel` and ``"native"`` for a string.
        kokkos_args (dict | None): Kokkos initialization settings, used the first
            time Kokkos is initialized in the process.
    """

    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    _state_class_name: str = ""
    _catalyst_device_name: str = ""
    _catalyst_lib_name: str = ""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires=None,
        *,
        shots=None,
        seed="global",
        c_dtype=np.complex128,
        noise_model=None,
        noise_mode: Optional[str] = None,
        kokkos_args: Optional[dict] = None,
    ):
        if self._ops_module() is None:
            raise ImportError(
                f"Pre-compiled binaries for {self.name} are not available. Build the "
                f"'{self._catalyst_lib_name}' backend (PL_BACKEND={self._catalyst_lib_name})."
            )
        super().__init__(wires=wires, shots=shots)
        if c_dtype not in (np.complex64, np.complex128):
            raise TypeError(f"Unsupported complex type: {c_dtype}")
        self._c_dtype = c_dtype
        self._rng = np.random.default_rng(
            np.random.randint(2**31 - 1) if isinstance(seed, str) and seed == "global" else seed
        )
        if isinstance(wires, int) or wires is None:
            self._wire_map = None
        else:
            self._wire_map = {w: i for i, w in enumerate(self.wires)}
        self._kokkos_args = _kokkos_dict(kokkos_args)
        self._state = None

        self._noise_model = None
        self._noise_mode = None
        self._native_spec: Optional[str] = None
        self._native_models = {}
        self._set_noise_model(noise_model, noise_mode)

    # ---- backend hooks ---------------------------------------------------

    @staticmethod
    def _ops_module():  # pragma: no cover - overridden
        return None

    def _state_measurement(self, state, mp, num_wires):
        """State-like measurements (``qml.state``, ``qml.density_matrix``, ...)."""
        raise DeviceError(f"Measurement {mp} not supported on {self.name}")

    def _apply_density_matrix(self, state, op, num_wires):
        raise DeviceError(f"{op.name} is not supported on {self.name}")

    def _expval_sparse(self, state, obs, num_wires, square=False):
        raise DeviceError(f"Sparse observables are not supported on {self.name}")

    def _new_state(self, num_wires: int):
        cls = getattr(
            self._ops_module(),
            self._state_class_name + ("C64" if self._c_dtype == np.complex64 else "C128"),
        )
        return cls(num_wires, self._kokkos_args)

    # ---- properties -------------------------------------------------------

    @property
    def c_dtype(self):
        """Complex data type of the simulation."""
        return self._c_dtype

    @property
    def noise_model(self):
        """The noise model of the device (``None``, a NoiseModel or a native string)."""
        return self._noise_model

    @property
    def noise_mode(self) -> Optional[str]:
        """``"transform"``, ``"native"`` or ``None`` (no noise)."""
        return self._noise_mode

    def _set_noise_model(self, noise_model, noise_mode):
        if noise_mode is not None and noise_mode not in NOISE_MODES:
            raise ValueError(f"noise_mode must be one of {NOISE_MODES}, got {noise_mode!r}")
        self._native_spec = None
        self._native_models = {}
        if noise_model is None:
            self._noise_model = None
            self._noise_mode = None
            return
        if isinstance(noise_model, str):
            if noise_mode == "transform":
                raise ValueError("A native noise model string requires noise_mode='native'.")
            self._native_spec = validate_native_spec(noise_model)
            self._noise_model = noise_model
            self._noise_mode = "native"
            try:
                self._native_model()  # parse eagerly to surface malformed specifications
            except RuntimeError as e:
                raise NativeNoiseError(f"Invalid native noise model: {e}") from e
            return
        if not hasattr(noise_model, "model_map") or not hasattr(noise_model, "metadata"):
            raise TypeError(
                "noise_model must be a pennylane.NoiseModel, a native noise model string or None."
            )
        self._noise_model = noise_model
        self._noise_mode = noise_mode or "transform"
        if self._noise_mode == "native":
            self.native_noise_model()  # convert eagerly to surface errors

    def native_noise_model(self) -> str:
        """The noise model serialized for the native backends (``""`` if none).

        This is the string handed to the Catalyst runtime device under qjit.
        """
        if self._noise_model is None:
            return ""
        if self._native_spec is None:
            # The conversion builds PennyLane operators; it may run while qjit is
            # tracing (``device_kwargs``), so program capture must be paused.
            with qp.capture.pause():
                self._native_spec = to_native_noise_model(
                    self._noise_model, self.wires, self.capabilities
                )
        return self._native_spec

    @property
    def device_kwargs(self) -> dict:
        """Keyword arguments passed to the Catalyst runtime device."""
        spec = self.native_noise_model()
        return {"noise_model": spec} if spec else {}

    def _native_model(self):
        """C++ noise model for native execution, or ``None``."""
        if self._noise_mode != "native":
            return None
        spec = self.native_noise_model()
        if not spec:
            return None
        key = self._c_dtype
        if key not in self._native_models:
            cls = getattr(
                self._ops_module(),
                "NativeNoiseModelC64" if self._c_dtype == np.complex64 else "NativeNoiseModelC128",
            )
            nm = cls(spec)
            self._native_models[key] = None if nm.empty() else nm
        return self._native_models[key]

    # ---- configuration ----------------------------------------------------

    def setup_execution_config(
        self, config: ExecutionConfig | None = None, circuit: QuantumScript | None = None
    ) -> ExecutionConfig:
        if config is None:
            config = ExecutionConfig()
        for option in config.device_options:
            raise DeviceError(f"device option {option} not present on {self}")

        updated = {}
        if config.use_device_gradient is None:
            updated["use_device_gradient"] = False
        if config.use_device_jacobian_product is None:
            updated["use_device_jacobian_product"] = False
        if config.grad_on_execution is None:
            updated["grad_on_execution"] = False

        mcm_config = config.mcm_config
        method = mcm_config.mcm_method
        allowed = (None, "device", "deferred", "one-shot")
        if qp.capture.enabled():
            allowed = (None, "device", "deferred", "single-branch-statistics")
        if method not in allowed:
            raise DeviceError(f"mcm_method='{method}' is not supported with {self.name}.")
        shots = getattr(circuit, "shots", None)
        if method in (None, "device") and not qp.capture.enabled():
            method = "one-shot" if shots else "deferred"
        elif method in (None, "device"):
            method = "deferred"
        if mcm_config.postselect_mode == "fill-shots" and method != "deferred":
            raise DeviceError("Using postselect_mode='fill-shots' is not supported.")
        updated["mcm_config"] = replace(mcm_config, mcm_method=method)
        return replace(config, **updated)

    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> qp.CompilePipeline:
        """Device preprocessing: decomposition to the device gate set, noise
        insertion (``noise_mode="transform"``), mid-circuit measurement handling
        and wire validation."""
        if execution_config is None:
            execution_config = ExecutionConfig()
        mcm_method = execution_config.mcm_config.mcm_method
        pipeline = qp.CompilePipeline()

        deferred = mcm_method == "deferred"
        base_gates = set(self.capabilities.gate_set())

        def _stop(allow_mcms):
            return partial(
                _stopping_condition,
                supports_operation=self.capabilities.supports_operation,
                allow_mcms=allow_mcms,
            )

        if qp.capture.enabled():
            if deferred:
                pipeline.add_transform(qp.defer_measurements, num_wires=len(self.wires))
            gate_set = base_gates if deferred else base_gates | {"MidMeasureMP"}
            pipeline.add_transform(
                qp.transforms.decompose, gate_set=gate_set, stopping_condition=_stop(not deferred)
            )
            return pipeline

        def _decompose(p, allow_mcms=True):
            p.add_transform(
                decompose,
                stopping_condition=_stop(allow_mcms),
                skip_initial_state_prep=True,
                name=self.name,
                device_wires=self.wires,
                target_gates=base_gates | {"MidMeasureMP"} if allow_mcms else base_gates,
            )
            p.add_transform(simplify_adjoints)

        def _resolve(p):
            p.add_transform(
                device_resolve_dynamic_wires, wires=self.wires, allow_resets=not deferred
            )

        pipeline.add_transform(validate_measurements, name=self.name)
        pipeline.add_transform(
            validate_observables, self.capabilities.supports_observable, name=self.name
        )
        # Noise is inserted on the circuit decomposed to the device gate set and,
        # with mcm_method="deferred", *before* the deferral: the noise of a
        # conditional gate stays conditional (and becomes a controlled channel)
        # and the operations added by the deferral are noiseless. This matches
        # the one-shot and qjit semantics.
        _decompose(pipeline)
        _resolve(pipeline)
        pipeline.add_transform(expand_wireless_measurements, device_wires=self.wires)
        if self._noise_mode == "transform":
            pipeline.add_transform(apply_noise_model, noise_model=self._noise_model)
            _decompose(pipeline)
            _resolve(pipeline)
        if deferred:
            if self._noise_mode == "native":
                pipeline.add_transform(insert_native_noise, device=self)
            pipeline.add_transform(defer_measurements, allow_postselect=False)
            _decompose(pipeline, allow_mcms=False)
            _resolve(pipeline)
        pipeline.add_transform(validate_device_wires, self.wires, name=self.name)
        if mcm_method == "one-shot":
            pipeline.add_transform(
                dynamic_one_shot, postselect_mode=execution_config.mcm_config.postselect_mode
            )
        pipeline.add_transform(qp.transforms.broadcast_expand)
        return pipeline

    def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
        """Device derivatives are not available; use parameter-shift or finite-diff."""
        return False

    # ---- execution --------------------------------------------------------

    def execute(self, circuits, execution_config: ExecutionConfig | None = None):
        if execution_config is None:
            execution_config = ExecutionConfig()
        return tuple(self._simulate(circuit) for circuit in circuits)

    def _seed(self) -> int:
        return int(self._rng.integers(0, 2**63 - 1))

    def _prepare_circuit(self, circuit: QuantumScript):
        """Map wires to indices and return ``(circuit, num_wires)``."""
        if self._wire_map is not None:
            [circuit], _ = qp.map_wires(circuit, self._wire_map)
            return circuit, len(self.wires)
        if self.wires is not None:
            return circuit, len(self.wires)
        if self._native_model() is not None:
            labels = list(circuit.wires)
            if not all(isinstance(w, (int, np.integer)) and w >= 0 for w in labels):
                raise DeviceError(
                    f"{self.name} with noise_mode='native' and wires=None requires "
                    "non-negative integer wire labels; pass wires= to the device."
                )
            return circuit, (max(labels) + 1 if labels else 0)
        num = circuit.num_wires
        return circuit.map_to_standard_wires(), num

    def _fresh_state(self, num_wires: int):
        num_wires = max(num_wires, 1)
        if self._state is None or self._state.num_qubits != num_wires:
            self._state = self._new_state(num_wires)
        else:
            self._state.reset_state()
        return self._state

    def _simulate(self, circuit: QuantumScript):
        circuit, num_wires = self._prepare_circuit(circuit)
        state = self._fresh_state(num_wires)
        native = self._native_model()
        gate_noise = native
        if any(isinstance(op, NativeNoiseApplied) for op in circuit.operations):
            # gate noise already inserted as channels (deferred mid-circuit measurements)
            gate_noise = None
            circuit = circuit.copy(
                operations=[
                    op for op in circuit.operations if not isinstance(op, NativeNoiseApplied)
                ]
            )

        if any(isinstance(op, MidMeasure) for op in circuit.operations):
            if not circuit.shots:
                raise DeviceError(
                    f"{self.name} executes mid-circuit measurements with shots "
                    "(mcm_method='one-shot') or with mcm_method='deferred'."
                )
            aux = QuantumScript(
                circuit.operations,
                circuit.measurements,
                shots=[1],
                trainable_params=circuit.trainable_params,
            )
            results = []
            for _ in range(circuit.shots.total_shots):
                state.reset_state()
                mids = {}
                self._apply_operations(state, aux.operations, gate_noise, mids)
                results.append(self._measure(state, aux, native, num_wires, mids))
            return tuple(results)

        self._apply_operations(state, circuit.operations, gate_noise, None)
        return self._measure(state, circuit, native, num_wires, None)

    # ---- operations -------------------------------------------------------

    def _apply_operations(self, state, operations, native, mids):
        for i, op in enumerate(operations):
            if isinstance(op, Conditional):
                if op.meas_val.concretize(mids):
                    self._apply_noisy(state, op.base, native, first=False)
            elif isinstance(op, MidMeasure):
                self._mid_measure(state, op, mids)
            else:
                self._apply_noisy(state, op, native, first=i == 0)

    def _mid_measure(self, state, op, mids):
        wire = int(op.wires[0])
        outcome = bool(state.measure(wire, -1, self._seed()))
        mids[op] = np.int64(outcome)
        if op.reset and outcome:
            state.apply_operation("PauliX", [wire])

    def _apply_noisy(self, state, op, native, first: bool):
        if _is_state_prep(op):
            self._apply_state_prep(state, op, first)
            return
        if native is None:
            self._apply_gate(state, op)
            return
        before, after = _gate_noise(native, op)
        if before:
            state.apply_noise(native, before)
        self._apply_gate(state, op)
        if after:
            state.apply_noise(native, after)

    def _apply_state_prep(self, state, op, first: bool):
        wires = [int(w) for w in op.wires]
        if isinstance(op, qp.QubitDensityMatrix):
            self._apply_density_matrix(state, op, state.num_qubits)
            return
        if not first:
            # decomposed by the preprocessing; kept for direct execute() calls
            for o in op.decomposition():
                self._apply_gate(state, o)
            return
        if isinstance(op, qp.BasisState):
            bits = np.asarray(_unwrap(op.parameters[0])).astype(int).ravel()
            state.set_basis_state([int(b) for b in bits], wires)
        else:
            vec = np.ascontiguousarray(_unwrap(op.state_vector(wire_order=op.wires)), self._c_dtype)
            state.set_pure_state(vec.ravel(), wires)

    def _matrix(self, op):
        return np.ascontiguousarray(_unwrap(qp.matrix(op, wire_order=op.wires)), self._c_dtype)

    def _params(self, op):
        return [float(np.real(_unwrap(p))) for p in op.parameters]

    def _apply_gate(self, state, op):  # pylint: disable=too-many-branches
        if isinstance(op, qp.Identity):
            return
        if isinstance(op, qp.operation.Channel):
            kraus = [np.ascontiguousarray(_unwrap(k), self._c_dtype) for k in op.kraus_matrices()]
            state.apply_channel(kraus, [int(w) for w in op.wires])
            return
        if isinstance(op, qp.ops.Controlled) and isinstance(op.base, qp.operation.Channel):
            # classically controlled channel (from deferred mid-circuit measurements)
            kraus = controlled_kraus(
                op.base.kraus_matrices(), [bool(v) for v in op.control_values], self._c_dtype
            )
            wires = [int(w) for w in op.control_wires] + [int(w) for w in op.base.wires]
            state.apply_channel(kraus, wires)
            return
        inverse = False
        base = op
        while isinstance(base, qp.ops.Adjoint):
            inverse = not inverse
            base = base.base
        wires = [int(w) for w in base.wires]
        name = base.name

        if isinstance(base, qp.PauliRot):
            word = base.hyperparameters["pauli_word"]
            ws = [w for w, p in zip(wires, word) if p != "I"]
            wd = "".join(p for p in word if p != "I")
            if wd:
                state.apply_pauli_rot(ws, inverse, self._params(base), wd)
            return
        if name in _NAMED_KERNELS:
            state.apply_operation(name, wires, inverse, self._params(base))
            return
        if isinstance(base, qp.ops.Controlled):
            target = base.base
            inv = inverse
            while isinstance(target, qp.ops.Adjoint):
                inv = not inv
                target = target.base
            cw = [int(w) for w in base.control_wires]
            cv = [bool(v) for v in base.control_values]
            tw = [int(w) for w in base.target_wires]
            if target.name in _CONTROLLABLE_KERNELS and not isinstance(target, qp.ops.Controlled):
                state.apply_operation(target.name, tw, inv, self._params(target), cw, cv)
            else:
                state.apply_matrix(self._matrix(target), tw, inv, cw, cv)
            return
        state.apply_matrix(self._matrix(base), wires, inverse)

    # ---- measurements -----------------------------------------------------

    def _readout_state(self, state, mp, native):
        """State on which ``mp`` is measured (a noisy copy if readout noise applies)."""
        if native is None or not native.has_readout_noise:
            return state
        kind = readout_kind(mp)
        if kind is None:
            return state
        noise = native.lookup_readout(kind, [int(w) for w in mp.wires])
        if not noise:
            return state
        noisy = state.copy()
        noisy.apply_noise(native, noise)
        return noisy

    def _obs_terms(self, obs):
        """Flattened terms of an observable for the C++ ``expval`` / ``var``."""
        coeffs, wires, paulis, mats = [], [], [], []
        empty = np.zeros(0, dtype=self._c_dtype)

        def add(o, scale):
            ps = o.pauli_rep
            if ps is not None:
                for pw, c in ps.items():
                    ws = list(pw.keys())
                    coeffs.append(complex(scale * _unwrap(c)))
                    wires.append([int(w) for w in ws])
                    paulis.append("".join(pw[w] for w in ws))
                    mats.append(empty)
                return
            if isinstance(o, Sum):
                for t in o.operands:
                    add(t, scale)
                return
            if isinstance(o, SProd):
                add(o.base, scale * _unwrap(o.scalar))
                return
            ws = [int(w) for w in o.wires]
            coeffs.append(complex(scale))
            wires.append(ws)
            paulis.append("")
            mats.append(self._matrix(o).ravel())

        add(obs, 1.0)
        return coeffs, wires, paulis, mats

    @staticmethod
    def _is_sparse(obs) -> bool:
        return isinstance(obs, qp.SparseHamiltonian)

    def _analytic(self, state, mp, native, num_wires):  # pylint: disable=too-many-return-statements
        if getattr(mp, "mv", None) is not None:
            # Statistics of (deferred) mid-circuit measurement values: the value
            # qubits are already in the computational basis.
            # pylint: disable=protected-access
            probs = np.asarray(state.probs([int(w) for w in mp.wires]))
            if isinstance(mp, ExpectationMP):
                return np.asarray(mp._calculate_expectation(probs))
            if isinstance(mp, VarianceMP):
                return np.asarray(mp._calculate_variance(probs))
            if isinstance(mp, ProbabilityMP):
                return probs
            raise DeviceError(f"Measurement {mp} is not supported on {self.name}")
        state = self._readout_state(state, mp, native)
        if isinstance(mp, ExpectationMP):
            if self._is_sparse(mp.obs):
                return np.asarray(self._expval_sparse(state, mp.obs, num_wires))
            return np.asarray(state.expval(*self._obs_terms(mp.obs)))
        if isinstance(mp, VarianceMP):
            if self._is_sparse(mp.obs):
                e = self._expval_sparse(state, mp.obs, num_wires)
                e2 = self._expval_sparse(state, mp.obs, num_wires, square=True)
                return np.asarray(e2 - e * e)
            return np.asarray(state.var(*self._obs_terms(mp.obs)))
        if isinstance(mp, ProbabilityMP):
            wires = [int(w) for w in mp.wires] or list(range(num_wires))
            diag = mp.diagonalizing_gates()
            if diag:
                state = state.copy()
                for g in diag:
                    self._apply_gate(state, g)
            return np.asarray(state.probs(wires))
        if isinstance(mp, (StateMP, DensityMatrixMP, PurityMP, VnEntropyMP, MutualInfoMP)):
            return self._state_measurement(state, mp, num_wires)
        raise DeviceError(f"Measurement {mp} is not supported on {self.name}")

    def _measure(self, state, circuit, native, num_wires, mids):
        if not circuit.shots:
            res = tuple(self._analytic(state, mp, native, num_wires) for mp in circuit.measurements)
            return res[0] if len(res) == 1 else res

        results = self._measure_with_samples(
            state, list(circuit.measurements), circuit.shots, native, num_wires, mids
        )
        if len(circuit.measurements) == 1:
            if circuit.shots.has_partitioned_shots:
                return tuple(res[0] for res in results)
            return results[0]
        return results

    def _measure_with_samples(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, state, measurements, shots: Shots, native, num_wires, mids=None
    ):
        mps = measurements[0 : -len(mids)] if mids else list(measurements)
        mps = [
            (
                mp.__class__(wires=Wires(range(num_wires)))
                if mp.obs is None and getattr(mp, "mv", None) is None and len(mp.wires) == 0
                else mp
            )
            for mp in mps
        ]
        readout = native is not None and native.has_readout_noise
        if readout:
            groups, indices = [[mp] for mp in mps], [[i] for i in range(len(mps))]
        else:
            groups, indices = _group_measurements(mps)

        all_res = []
        for group in groups:
            first = group[0]
            if isinstance(first, (ExpectationMP, VarianceMP)) and self._is_sparse(first.obs):
                raise TypeError(
                    "ExpectationMP/VarianceMP of sparse observables cannot be computed with samples."
                )
            if isinstance(first, VarianceMP) and isinstance(first.obs, Sum):
                raise TypeError("VarianceMP(Sum) cannot be computed with samples.")
            if isinstance(first, (ClassicalShadowMP, ShadowExpvalMP)):
                raise TypeError("Classical shadows are not supported by this device.")
            s = self._readout_state(state, first, native) if readout else state
            if isinstance(first, ExpectationMP) and isinstance(first.obs, Sum):
                all_res.extend(self._measure_sum_with_samples(s, first, shots, num_wires))
            else:
                all_res.extend(self._sample_group(s, group, shots))

        flat = [i for row in indices for i in row]
        sorted_res = tuple(res for _, res in sorted(enumerate(all_res), key=lambda r: flat[r[0]]))
        if mids:
            sorted_res += tuple(mids.values())
        if shots.has_partitioned_shots:
            sorted_res = tuple(zip(*sorted_res))
        return sorted_res

    def _measure_sum_with_samples(self, state, mp, shots: Shots, num_wires):
        def single(s):
            res = self._measure_with_samples(
                state, [ExpectationMP(t) for t in mp.obs], Shots(s), None, num_wires
            )
            return sum(res)

        results = tuple(single(s) for s in shots)
        return [results] if shots.has_partitioned_shots else [results[0]]

    def _sample_group(self, state, mps, shots: Shots):
        if len(mps) == 1:
            diag = mps[0].diagonalizing_gates()
        elif all(mp.obs for mp in mps):
            diag = qp.pauli.diagonalize_qwc_pauli_words([mp.obs for mp in mps])[0]
        else:
            diag = []
        if diag:
            state = state.copy()
            for g in diag:
                self._apply_gate(state, g)
        wires = Wires.all_wires([mp.wires for mp in mps])
        samples = np.asarray(
            state.sample([int(w) for w in wires], shots.total_shots, self._seed())
        ).astype(int, copy=False)

        processed = []
        for lower, upper in shots.bins():
            processed.append(tuple(mp.process_samples(samples[lower:upper], wires) for mp in mps))
        return tuple(zip(*processed)) if shots.has_partitioned_shots else processed[0]

    # ---- Catalyst ---------------------------------------------------------

    @classmethod
    def get_c_interface(cls):
        """Device name and shared library of the Catalyst runtime device."""
        # pylint: disable=import-outside-toplevel
        from pennylane_lightning.lightning_base.lightning_base import LightningBase

        return LightningBase.get_c_interface_impl(cls._catalyst_device_name, cls._catalyst_lib_name)


__all__ = [
    "LightningNoisyBase",
    "NativeNoiseError",
    "apply_noise_model",
    "expand_wireless_measurements",
    "is_native_spec",
    "readout_kind",
    "simplify_adjoints",
]
