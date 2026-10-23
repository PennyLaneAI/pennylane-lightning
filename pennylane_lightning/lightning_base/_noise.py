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
Lowering of a PennyLane :class:`~pennylane.NoiseModel` to the native noise
format (``LNM1``) of the ``lightning.tensornet`` backend.

A ``NoiseModel`` is a set of Python callables (conditions and noise
functions). The C++ backends and their Catalyst runtime devices cannot call
Python, so the model is *tabulated*: every gate the backend can execute is
probed on the device wires, the conditions and noise functions are evaluated
exactly as :func:`pennylane.noise.add_noise` would, and the resulting Kraus
channels and placements are written into a flat rule table (see
``pennylane_lightning/core/noise/NoiseModel.hpp`` for the format).

What is exact and what is approximated:

* Conditions built from :func:`~pennylane.noise.op_eq`,
  :func:`~pennylane.noise.op_in`, :func:`~pennylane.noise.wires_in`,
  :func:`~pennylane.noise.wires_eq` and their ``&``, ``|``, ``^``, ``~``
  combinations are evaluated exactly for every gate and wire tuple that is
  probed. Arbitrary Python conditions are evaluated on a probe operation.
* Gates of up to ``exhaustive_arity`` wires are probed on every ordered wire
  tuple as long as there are at most ``max_tuples`` of them; beyond that a
  sample is probed and the most common result is used as a wildcard (a
  warning is raised if the sampled results are not uniform).
* Noise that depends on the gate *parameters* (e.g. over-rotations) cannot be
  tabulated and raises :class:`NativeNoiseError`.
* Noise functions that fail on a wire-less ``GlobalPhase`` (e.g.
  :func:`~pennylane.noise.partial_wires` under a ``wires_in`` condition, which
  matches the empty wire set) add no noise to it. ``add_noise`` raises in that
  case.
* Readout noise (``meas_map``) is keyed by measurement type and wires; if it
  depends on the measured observable, the ``PauliZ`` result is used and a
  warning is raised.
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-branches
# pylint: disable=too-many-return-statements,too-many-instance-attributes,too-few-public-methods
from __future__ import annotations

import itertools
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pennylane as qp
from pennylane.boolean_fn import And, Not, Or, Xor
from pennylane.core.qscript import make_qscript
from pennylane.noise.conditionals import OpEq, OpIn, WiresEq, WiresIn

LNM_HEADER = "LNM1"
READOUT_KINDS = ("expval", "var", "probs", "sample", "counts")
_FORBIDDEN_CHARS = set(",:{} '\"")

# Channels that the backends can apply when they appear as operations.
CHANNEL_NAMES = (
    "AmplitudeDamping",
    "GeneralizedAmplitudeDamping",
    "PhaseDamping",
    "DepolarizingChannel",
    "BitFlip",
    "PhaseFlip",
    "PauliError",
    "ResetError",
    "ThermalRelaxationError",
    "QubitChannel",
)

# Controlled gates that the runtime receives under their own name.
NAMED_CONTROLLED_GATES = frozenset(
    {
        "CNOT",
        "CY",
        "CZ",
        "CH",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "CSWAP",
        "ControlledPhaseShift",
        "Toffoli",
    }
)


class NativeNoiseError(ValueError):
    """Raised when a noise model cannot be lowered to the native format."""


def is_native_spec(model) -> bool:
    """Return ``True`` if ``model`` is an already serialized native noise model."""
    return isinstance(model, str) and model.startswith(LNM_HEADER)


def validate_native_spec(spec: str) -> str:
    """Check the characters of a serialized native model (Catalyst's kwargs
    parser does not accept ``,:{}``, spaces or quotes)."""
    if not is_native_spec(spec):
        raise NativeNoiseError(f"A native noise model must start with '{LNM_HEADER}'.")
    bad = _FORBIDDEN_CHARS.intersection(spec)
    if bad:
        raise NativeNoiseError(f"Invalid characters {sorted(bad)} in native noise model.")
    return spec


# ---------------------------------------------------------------------------
# Runtime keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateKey:
    """A gate as seen by the native backend: ``name``, ``Adjoint(name)``,
    ``C(name)`` or ``C(Adjoint(name))``."""

    base: str
    adjoint: bool = False
    controlled: bool = False

    @property
    def key(self) -> str:
        """Key used in the rule table."""
        k = f"Adjoint({self.base})" if self.adjoint else self.base
        return f"C({k})" if self.controlled else k


def runtime_key(op) -> tuple[str, list]:
    """Key and wires (controls first) under which the backend looks up the noise
    of ``op``. This mirrors how Catalyst lowers operations to the runtime: named
    controlled gates (``CNOT``, ``CRX``, ...) keep their name, other controlled
    operations become ``C(<base>)`` and matrix operations ``QubitUnitary``.
    """
    inverse = False
    while isinstance(op, qp.ops.Adjoint):
        inverse = not inverse
        op = op.base
    ctrls = []
    if op.name not in NAMED_CONTROLLED_GATES and isinstance(op, qp.ops.Controlled):
        ctrls = list(op.control_wires)
        base = op.base
        if isinstance(op, qp.ControlledQubitUnitary):
            name = "QubitUnitary"
        else:
            while isinstance(base, qp.ops.Adjoint):
                inverse = not inverse
                base = base.base
            name = base.name
            if name.startswith("C(") or isinstance(base, qp.ops.Controlled):
                name = "QubitUnitary"
        targets = list(op.target_wires)
    else:
        name = op.name
        targets = list(op.wires)
    key = GateKey(name, inverse, bool(ctrls)).key
    return key, ctrls + targets


# ---------------------------------------------------------------------------
# Probe operations
# ---------------------------------------------------------------------------


def _fixed_num_wires(cls) -> Optional[int]:
    for klass in type.mro(cls):
        if "num_wires" in klass.__dict__:
            nw = klass.__dict__["num_wires"]
            return nw if isinstance(nw, int) else None
    return None


def _random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random unitary (Haar for up to two qubits, a product of random one-qubit
    unitaries beyond that; probes only need a generic matrix)."""
    if dim > 4:
        u = np.ones((1, 1), dtype=complex)
        for _ in range(int(round(math.log2(dim)))):
            u = np.kron(u, _random_unitary(2, rng))
        return u
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _op_class(name: str):
    cls = getattr(qp, name, None)
    if cls is None:
        cls = getattr(qp.ops, name, None)
    return cls


@dataclass
class _GateSpec:
    """How to build a PennyLane operation for a base gate name."""

    name: str
    arity: Optional[int]  # None: variable number of target wires
    min_arity: int
    build: Callable[[Sequence, np.random.Generator], object]
    max_arity: Optional[int] = None


def _make_spec(name: str) -> Optional[_GateSpec]:
    cls = _op_class(name)
    if cls is None or not isinstance(cls, type):
        return None
    fixed = _fixed_num_wires(cls)

    def ang(rng):
        return float(rng.uniform(0.1, 3.0))

    def prob(rng):
        return float(rng.uniform(0.01, 0.2))

    custom = {
        "QubitUnitary": (None, 1, lambda w, r: cls(_random_unitary(2 ** len(w), r), wires=w)),
        "DiagonalQubitUnitary": (
            None,
            1,
            lambda w, r: cls(np.exp(1j * r.uniform(0, 2 * np.pi, 2 ** len(w))), wires=w),
        ),
        "PauliRot": (None, 1, lambda w, r: cls(ang(r), "X" * len(w), wires=w)),
        "MultiRZ": (None, 1, lambda w, r: cls(ang(r), wires=w)),
        "PCPhase": (None, 1, lambda w, r: cls(ang(r), dim=1, wires=w)),
        "GlobalPhase": (0, 0, lambda w, r: cls(ang(r))),
        "Identity": (None, 1, lambda w, r: cls(wires=w)),
        "MultiControlledX": (None, 2, lambda w, r: cls(wires=w)),
        "ControlledQubitUnitary": (
            None,
            2,
            lambda w, r: cls(_random_unitary(2, r), wires=w),
        ),
        "BlockEncode": (None, 1, lambda w, r: cls(float(r.uniform(0.1, 0.9)), wires=w)),
        "PauliError": (None, 1, lambda w, r: cls("X" * len(w), prob(r), wires=w)),
        "QubitChannel": (
            None,
            1,
            lambda w, r: cls([np.eye(2 ** len(w))], wires=w),
        ),
        "ThermalRelaxationError": (
            1,
            1,
            lambda w, r: cls(prob(r), 100.0, 80.0, float(r.uniform(0.1, 1.0)), wires=w),
        ),
        "GeneralizedAmplitudeDamping": (1, 1, lambda w, r: cls(prob(r), prob(r), wires=w)),
        "ResetError": (1, 1, lambda w, r: cls(prob(r), prob(r), wires=w)),
    }
    if name in custom:
        arity, min_arity, build = custom[name]
        max_arity = 3 if name in ("QubitChannel", "PauliError") else None
        return _GateSpec(name, arity, min_arity, build, max_arity)
    if fixed is None:
        return None
    nparams = cls.__dict__.get("num_params", None)
    if not isinstance(nparams, int):
        for klass in type.mro(cls):
            v = klass.__dict__.get("num_params", None)
            if isinstance(v, int):
                nparams = v
                break
    if not isinstance(nparams, int):
        return None
    is_channel = name in CHANNEL_NAMES

    def build_default(w, r, cls=cls, nparams=nparams, is_channel=is_channel):
        params = [prob(r) if is_channel else ang(r) for _ in range(nparams)]
        return cls(*params, wires=w)

    return _GateSpec(name, fixed, fixed, build_default)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def _collect_condition_wires(cond, out: set) -> bool:
    """Collect the wires named by WiresIn / WiresEq conditions. Returns ``True``
    if the condition contains callables that are not structured conditionals."""
    if isinstance(cond, (And, Or, Xor, Not)):
        opaque = False
        for c in cond.operands:
            opaque |= _collect_condition_wires(c, out)
        return opaque
    if isinstance(cond, (WiresIn, WiresEq)):
        out.update(cond.condition)
        return False
    if isinstance(cond, (OpEq, OpIn)):
        return False
    return True


class _CondEvaluator:
    """Three-valued evaluation of a condition for (key, wires) without building
    the operation, falling back to a probe operation for opaque callables."""

    def __init__(self, cond):
        self.cond = cond
        self._op_cache = {}

    def _op_match(self, c, key, probe) -> bool:
        k = (id(c), key)
        if k not in self._op_cache:
            try:
                self._op_cache[k] = bool(c(probe()))
            except Exception:  # pylint: disable=broad-except
                self._op_cache[k] = False
        return self._op_cache[k]

    def _eval(self, c, key, labels, probe):
        if isinstance(c, And):
            a = self._eval(c.operands[0], key, labels, probe)
            if a is False:
                return False
            b = self._eval(c.operands[1], key, labels, probe)
            if b is False:
                return False
            return True if (a and b) else None
        if isinstance(c, Or):
            a = self._eval(c.operands[0], key, labels, probe)
            if a is True:
                return True
            b = self._eval(c.operands[1], key, labels, probe)
            if b is True:
                return True
            return False if (a is False and b is False) else None
        if isinstance(c, Xor):
            a = self._eval(c.operands[0], key, labels, probe)
            b = self._eval(c.operands[1], key, labels, probe)
            return None if (a is None or b is None) else (a != b)
        if isinstance(c, Not):
            a = self._eval(c.operands[0], key, labels, probe)
            return None if a is None else (not a)
        if isinstance(c, WiresIn):
            return set(labels).issubset(c.condition)
        if isinstance(c, WiresEq):
            return set(labels) == set(c.condition)
        if isinstance(c, (OpEq, OpIn)):
            return self._op_match(c, key, probe)
        return None

    def static(self, key, probe) -> Optional[bool]:
        """Result that holds for every wire tuple (``None`` if wire dependent)."""
        return self._static(self.cond, key, probe)

    def _static(self, c, key, probe):
        if isinstance(c, (WiresIn, WiresEq)):
            return None
        if isinstance(c, (OpEq, OpIn)):
            return self._op_match(c, key, probe)
        if isinstance(c, And):
            a, b = (self._static(x, key, probe) for x in c.operands)
            if a is False or b is False:
                return False
            return True if (a and b) else None
        if isinstance(c, Or):
            a, b = (self._static(x, key, probe) for x in c.operands)
            if a is True or b is True:
                return True
            return False if (a is False and b is False) else None
        if isinstance(c, Xor):
            a, b = (self._static(x, key, probe) for x in c.operands)
            return None if (a is None or b is None) else (a != b)
        if isinstance(c, Not):
            a = self._static(c.operands[0], key, probe)
            return None if a is None else (not a)
        return None

    def structured(self, key, labels, probe_canonical) -> Optional[bool]:
        """Result for a wire tuple when it can be decided without a probe."""
        return self._eval(self.cond, key, labels, probe_canonical)

    def __call__(self, key, labels, probe_canonical, probe_op) -> bool:
        r = self._eval(self.cond, key, labels, probe_canonical)
        if r is not None:
            return r
        try:
            return bool(self.cond(probe_op()))
        except Exception:  # pylint: disable=broad-except
            return False


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class NoiseModelConverter:
    """Tabulate a PennyLane ``NoiseModel`` into the native ``LNM1`` format.

    Args:
        noise_model (~pennylane.NoiseModel): the noise model.
        wires (Sequence): device wire labels; position ``i`` is backend wire ``i``.
        gates (dict[str, set[str]]): base gate names and their capability
            properties (``"invertible"``, ``"controllable"``).
        max_tuples (int): maximum number of wire tuples probed per gate and arity.
        exhaustive_arity (int): gates on up to this many wires are probed on all
            wire tuples (subject to ``max_tuples``); larger ones are sampled.
        max_arity (int): largest number of wires of a probed operation.
        sample_uniform (int): within a group of wire tuples matched by the same
            rules, evaluate the noise functions on this many tuples and reuse the
            result if they agree (``0`` evaluates every tuple).
        seed (int): seed for the random probe parameters and tuple samples.
    """

    def __init__(
        self,
        noise_model,
        wires: Sequence,
        gates: dict,
        *,
        max_tuples: int = 2000,
        exhaustive_arity: int = 3,
        sample_uniform: int = 4,
        max_arity: int = 8,
        seed: int = 1234,
    ):
        if not hasattr(noise_model, "model_map") or not hasattr(noise_model, "metadata"):
            raise NativeNoiseError("Expected a qml.NoiseModel instance.")
        self.model = noise_model
        self.labels = list(wires)
        self.index = {w: i for i, w in enumerate(self.labels)}
        self.n = len(self.labels)
        self.gates = gates
        self.max_tuples = max_tuples
        self.exhaustive_arity = exhaustive_arity
        self.sample_uniform = sample_uniform
        self.max_arity = max_arity
        self.seed = seed
        self.metadata = dict(noise_model.metadata)
        self.rules = [
            (_CondEvaluator(c), make_qscript(f)) for c, f in noise_model.model_map.items()
        ]
        self.meas_rules = [(c, make_qscript(f)) for c, f in (noise_model.meas_map or {}).items()]
        self._interesting = set()
        self._opaque = False
        for c in list(noise_model.model_map) + list(noise_model.meas_map or {}):
            self._opaque |= _collect_condition_wires(c, self._interesting)
        self._interesting = [w for w in self.labels if w in self._interesting]
        self._channels = []
        self._channel_ids = {}
        self._kraus_cache = {}
        self._warnings = []

    # -- channels ---------------------------------------------------------

    def _channel_id(self, op) -> int:
        nq = len(op.wires)
        try:
            ck = (op.name, nq, tuple(np.asarray(d).tobytes() for d in op.data))
        except Exception:  # pylint: disable=broad-except
            ck = None
        if ck is not None and ck in self._kraus_cache:
            return self._kraus_cache[ck]
        cid = self._channel_id_uncached(op, nq)
        if ck is not None:
            self._kraus_cache[ck] = cid
        return cid

    def _channel_id_uncached(self, op, nq) -> int:
        kraus = None
        if isinstance(op, qp.operation.Channel):
            kraus = [np.asarray(k, dtype=complex) for k in op.kraus_matrices()]
        else:
            try:
                kraus = [np.asarray(qp.matrix(op, wire_order=op.wires), dtype=complex)]
            except Exception as e:  # pylint: disable=broad-except
                raise NativeNoiseError(
                    f"Noise operation {op} has neither Kraus operators nor a matrix."
                ) from e
        dim = 2**nq
        for k in kraus:
            if k.shape != (dim, dim):
                raise NativeNoiseError(f"Noise operation {op} has an invalid Kraus shape.")
        sig = (nq, np.round(np.stack(kraus), 12).tobytes())
        if sig not in self._channel_ids:
            self._channel_ids[sig] = len(self._channels)
            self._channels.append((nq, kraus))
        return self._channel_ids[sig]

    def _ref(self, w, op_wires) -> str:
        if w in op_wires:
            return f"r{op_wires.index(w)}"
        if w not in self.index:
            raise NativeNoiseError(f"Noise acts on wire {w!r}, which is not a device wire.")
        return str(self.index[w])

    def _encode(self, placed, op_wires) -> str:
        acts = []
        for where, nop in placed:
            if isinstance(nop, qp.ops.MidMeasure) or not isinstance(nop, qp.operation.Operator):
                raise NativeNoiseError(f"Unsupported noise operation {nop}.")
            ch = self._channel_id(nop)
            refs = "_".join(self._ref(w, op_wires) for w in nop.wires)
            if not refs:
                # wire-less noise (e.g. a global phase) has no effect on a state
                continue
            acts.append(f"{where}{ch}@{refs}")
        return "+".join(acts) if acts else "-"

    # -- gates --------------------------------------------------------------

    def _build(self, gk: GateKey, spec: _GateSpec, labels, rng, target_arity):
        ctrl = list(labels[: len(labels) - target_arity]) if gk.controlled else []
        targets = list(labels[len(ctrl) :])
        with qp.QueuingManager.stop_recording():
            op = spec.build(targets, rng)
            if gk.adjoint:
                op = qp.adjoint(op, lazy=False)
            if gk.controlled:
                op = qp.ctrl(op, control=ctrl)
        return op

    def _canonical_probe(self, gk: GateKey, spec: _GateSpec, k: int, target_arity: int):
        """Lazily built probe on the first ``k`` wires (used for the wire-independent
        parts of the conditions)."""
        store = {}

        def probe():
            if "op" not in store:
                store["op"] = self._build(
                    gk, spec, self._pad_labels(k), np.random.default_rng(self.seed), target_arity
                )
            return store["op"]

        return probe

    def _gate_pattern(self, gk: GateKey, spec: _GateSpec, labels, rng, target_arity, canon) -> str:
        cache = {}

        def probe():
            if "op" not in cache:
                cache["op"] = self._build(gk, spec, labels, rng, target_arity)
            return cache["op"]

        ckey = (gk.key, len(labels))
        curr = [None]
        pos = 0
        for cond, fn in self.rules:
            if not cond(ckey, labels, canon, probe):
                continue
            op = probe()
            try:
                noise_ops = list(fn(op, **self.metadata).operations)
            except Exception as e:  # pylint: disable=broad-except
                if not labels:
                    # A wire-less operation (GlobalPhase) matched a wire condition such
                    # as wires_in (the empty set is a subset); wire-relative noise
                    # functions (partial_wires) cannot act on it: no noise.
                    continue
                raise NativeNoiseError(
                    f"The noise function {fn} failed on the probe operation {op}: {e}"
                ) from e
            if any(qp.equal(op, o) for o in noise_ops):
                idx = noise_ops.index(op)
                curr = noise_ops[:idx] + curr + noise_ops[idx + 1 :]
                pos += idx
            else:
                curr = curr + noise_ops
        placed = [("b", o) for o in curr[:pos]] + [("a", o) for o in curr[pos + 1 :]]
        return self._encode(placed, list(labels))

    def _pad_labels(self, k):
        labels = list(self.labels[:k])
        extra = 0
        while len(labels) < k:
            labels.append(f"__probe_{extra}")
            extra += 1
        return labels

    def _tuples(self, k, rng, allow_exhaustive: bool):
        n = self.n
        if k == 0:
            return [()], True
        if k > n:
            return [], True
        total = math.perm(n, k)
        if allow_exhaustive and total <= self.max_tuples:
            return list(itertools.permutations(self.labels, k)), True
        # sampled: tuples through every interesting wire and position + random ones
        picked = set()
        picked.add(tuple(self.labels[:k]))
        picked.add(tuple(self.labels[-k:][::-1]))
        budget = self.max_tuples if k <= self.exhaustive_arity else 8
        for w in self._interesting:
            others = [x for x in self.labels if x != w]
            for p in range(k):
                sel = list(rng.choice(len(others), size=k - 1, replace=False))
                t = [others[i] for i in sel]
                t.insert(p, w)
                picked.add(tuple(t))
        if len(self._interesting) >= k:
            for t in itertools.islice(itertools.permutations(self._interesting, k), budget):
                picked.add(t)
        while len(picked) < min(budget, total):
            sel = rng.choice(n, size=k, replace=False)
            picked.add(tuple(self.labels[i] for i in sel))
        return sorted(picked, key=lambda t: [self.index[w] for w in t]), False

    def _gate_keys(self):
        keys = []
        for name, props in self.gates.items():
            spec = _make_spec(name)
            if spec is None:
                continue
            keys.append((GateKey(name), spec))
            if "invertible" in props:
                keys.append((GateKey(name, adjoint=True), spec))
            if "controllable" in props:
                keys.append((GateKey(name, controlled=True), spec))
                if "invertible" in props:
                    keys.append((GateKey(name, adjoint=True, controlled=True), spec))
        return keys

    def _arities(self, gk: GateKey, spec: _GateSpec):
        """(total arity, target arity) pairs to probe."""
        top = min(self.n, self.max_arity, spec.max_arity or self.n)
        return [(k, t) for k, t in self._all_arities(gk, spec) if k <= top or k == 0]

    def _all_arities(self, gk: GateKey, spec: _GateSpec):
        if spec.arity is not None:
            if not gk.controlled:
                return [(spec.arity, spec.arity)]
            return [(spec.arity + c, spec.arity) for c in range(1, self.n - spec.arity + 1)]
        if not gk.controlled:
            return [(k, k) for k in range(spec.min_arity, self.n + 1)]
        # variable-arity bases under control: a single target wire (see module docs)
        t = max(spec.min_arity, 1)
        return [(k, t) for k in range(t + 1, self.n + 1)]

    def _gate_entries(self) -> list[str]:
        entries = []
        truncated = []
        rng = np.random.default_rng(self.seed)
        if not self.rules:
            return entries
        for gk, spec in self._gate_keys():
            for k, t_arity in self._arities(gk, spec):
                if k > max(self.n, 0) or (k == 0 and gk.controlled):
                    continue
                canon = self._canonical_probe(gk, spec, k, t_arity)
                try:
                    canon()
                except Exception:  # pylint: disable=broad-except
                    continue  # this arity cannot be built for this gate
                ckey = (gk.key, k)
                if all(cond.static(ckey, canon) is False for cond, _ in self.rules):
                    continue
                fixed = spec.arity is not None and not gk.controlled
                tuples, exhaustive = self._tuples(
                    k, rng, allow_exhaustive=fixed or k <= self.exhaustive_arity
                )
                if not tuples:
                    continue
                pats = self._probe_tuples(gk, spec, tuples, t_arity, canon, ckey, rng)
                # parameter dependence: re-probe noisy tuples with other parameters
                noisy = [tp for tp, p in pats.items() if p != "-"]
                for tup in noisy[:3]:
                    p2 = self._gate_pattern(
                        gk, spec, tup, np.random.default_rng(self.seed + 7919), t_arity, canon
                    )
                    if p2 != pats[tup]:
                        raise NativeNoiseError(
                            f"The noise of {gk.key} on wires {list(tup)} depends on the gate "
                            "parameters; it cannot be lowered to a native noise model. Use "
                            "noise_mode='transform' (Python execution only)."
                        )
                entries.extend(self._emit("G", gk.key, k, pats, exhaustive))
                if k == self.max_arity < self.n and any(p != "-" for p in pats.values()):
                    truncated.append(gk.key)
        if truncated:
            self._warnings.append(
                f"Noise is only tabulated for operations on up to {self.max_arity} wires "
                f"(affects {', '.join(truncated[:5])}{', ...' if len(truncated) > 5 else ''}); "
                "increase max_arity to cover larger operations."
            )
        return entries

    def _probe_tuples(self, gk, spec, tuples, t_arity, canon, ckey, rng) -> dict:
        """Noise pattern of every wire tuple.

        Tuples are grouped by which rules match them (decided without building
        an operation when the conditions are structured). Within a group the
        noise functions are evaluated on ``sample_uniform`` tuples; if these give
        the same wire-relative pattern it is used for the whole group, otherwise
        every tuple of the group is evaluated.
        """
        pattern = lambda tup: self._gate_pattern(  # pylint: disable=unnecessary-lambda-assignment
            gk, spec, tup, np.random.default_rng(self.seed), t_arity, canon
        )
        groups = {}
        for tup in tuples:
            sig = []
            for cond, _ in self.rules:
                r = cond.structured(ckey, tup, canon)
                if r is None:
                    sig = None
                    break
                sig.append(r)
            key = ("opaque", tup) if sig is None else tuple(sig)
            groups.setdefault(key, []).append(tup)
        pats = {}
        for sig, tups in groups.items():
            if sig and sig[0] != "opaque" and not any(sig):
                for tup in tups:
                    pats[tup] = "-"
                continue
            if self.sample_uniform and len(tups) > self.sample_uniform and sig[0] != "opaque":
                idx = {0, len(tups) - 1}
                while len(idx) < self.sample_uniform:
                    idx.add(int(rng.integers(len(tups))))
                probed = {tups[i]: pattern(tups[i]) for i in sorted(idx)}
                if len(set(probed.values())) == 1:
                    only = next(iter(probed.values()))
                    for tup in tups:
                        pats[tup] = only
                    continue
                pats.update(probed)
                for tup in tups:
                    if tup not in pats:
                        pats[tup] = pattern(tup)
                continue
            for tup in tups:
                pats[tup] = pattern(tup)
        return pats

    def _emit(self, kind, key, k, pats: dict, exhaustive: bool) -> list[str]:
        if all(p == "-" for p in pats.values()):
            return []
        counts = Counter(pats.values())
        best = max(counts.values())
        majority = "-" if counts.get("-", 0) == best else counts.most_common(1)[0][0]
        if not exhaustive and len(counts) > 1:
            self._warnings.append(
                f"Noise for {key} on {k} wires is not uniform over the sampled wire tuples; "
                "tuples that were not sampled use the most common result."
            )
        out = []
        if majority != "-":
            out.append(f"{kind}={key}/*{k}/{majority}")
        for tup, p in pats.items():
            if p != majority:
                w = "_".join(str(self.index[x]) for x in tup) if tup else ""
                if not w:
                    # arity-0 key: the wildcard is the only representation
                    out.append(f"{kind}={key}/*0/{p}")
                    continue
                out.append(f"{kind}={key}/{w}/{p}")
        return out

    # -- readout ------------------------------------------------------------

    def _make_mp(self, kind, wires, obs="Z"):
        with qp.QueuingManager.stop_recording():
            if kind in ("expval", "var"):
                ob = (
                    qp.prod(*[getattr(qp, obs)(w) for w in wires])
                    if len(wires) > 1
                    else getattr(qp, obs)(wires[0])
                )
                return qp.expval(ob) if kind == "expval" else qp.var(ob)
            return {"probs": qp.probs, "sample": qp.sample, "counts": qp.counts}[kind](wires=wires)

    def _readout_ops(self, kind, wires, obs="Z"):
        mp = self._make_mp(kind, list(wires), obs)
        ops = []
        for cond, fn in self.meas_rules:
            try:
                ok = bool(cond(mp))
            except Exception:  # pylint: disable=broad-except
                ok = False
            if ok:
                ops.extend(fn(mp, **self.metadata).operations)
        return ops, list(mp.wires)

    def _readout_pattern(self, kind, wires, obs="Z"):
        ops, mw = self._readout_ops(kind, wires, obs)
        return self._encode([("a", o) for o in ops], mw), ops

    def _resolved(self, ops):
        """Per-wire channel sequences (for comparing with the union of singles)."""
        seq = []
        for o in ops:
            ch = self._channel_id(o)
            seq.append((ch, tuple(self.index[w] for w in o.wires)))
        return seq

    @staticmethod
    def _same_action(a, b) -> bool:
        if a == b:
            return True
        if all(len(w) == 1 for _, w in a + b):
            per = lambda s: sorted(  # pylint: disable=unnecessary-lambda-assignment
                (w[0], [c for c, ww in s if ww == w]) for w in {x for _, x in s}
            )
            return per(a) == per(b)
        return False

    def _readout_entries(self) -> list[str]:
        entries = []
        if not self.meas_rules or self.n == 0:
            return entries
        rng = np.random.default_rng(self.seed + 1)
        for kind in READOUT_KINDS:
            singles = {}
            single_ops = {}
            for w in self.labels:
                p, ops = self._readout_pattern(kind, [w])
                singles[(w,)] = p
                single_ops[w] = ops
                if kind in ("expval", "var"):
                    px, _ = self._readout_pattern(kind, [w], obs="PauliX")
                    if px != p:
                        self._warnings.append(
                            f"Readout noise for {kind} depends on the observable; the "
                            "native model uses the PauliZ result."
                        )
            entries.extend(self._emit("M", kind, 1, singles, True))
            for k in range(2, self.n + 1):
                total = math.comb(self.n, k)
                limit = self.max_tuples if k == 2 else 500
                if k <= 3 and total <= limit:
                    combos = list(itertools.combinations(self.labels, k))
                    exhaustive = True
                else:
                    combos = set()
                    combos.add(tuple(self.labels[:k]))
                    for _ in range(min(8, total)):
                        sel = sorted(rng.choice(self.n, size=k, replace=False))
                        combos.add(tuple(self.labels[i] for i in sel))
                    combos = sorted(combos, key=lambda t: [self.index[w] for w in t])
                    exhaustive = False
                mismatched = {}
                for tup in combos:
                    p, ops = self._readout_pattern(kind, list(tup))
                    union = [o for w in tup for o in single_ops[w]]
                    if not self._same_action(self._resolved(ops), self._resolved(union)):
                        mismatched[tup] = p
                if mismatched:
                    if exhaustive:
                        for tup, p in mismatched.items():
                            w = "_".join(str(self.index[x]) for x in tup)
                            entries.append(f"M={kind}/{w}/{p}")
                    else:
                        self._warnings.append(
                            f"Readout noise for {kind} on {k} wires is not the union of "
                            "the single-wire noise; only the sampled tuples are exact."
                        )
                        for tup, p in mismatched.items():
                            w = "_".join(str(self.index[x]) for x in tup)
                            entries.append(f"M={kind}/{w}/{p}")
                if k >= 3 and not mismatched and not exhaustive:
                    break
        return entries

    # -- serialize ----------------------------------------------------------

    def convert(self) -> str:
        """Return the serialized ``LNM1`` model."""
        gate_entries = self._gate_entries()
        meas_entries = self._readout_entries()
        parts = [LNM_HEADER]
        for nq, kraus in self._channels:
            vals = []
            for k in kraus:
                for z in np.asarray(k).reshape(-1):
                    vals.append(repr(float(z.real)))
                    vals.append(repr(float(z.imag)))
            parts.append(f"K={nq}/{len(kraus)}/" + "/".join(vals))
        parts.extend(gate_entries)
        parts.extend(meas_entries)
        for w in dict.fromkeys(self._warnings):
            warnings.warn(w, UserWarning)
        return validate_native_spec(";".join(parts))


def gates_from_capabilities(capabilities, include_channels: bool = True) -> dict:
    """Base gate names and properties from a ``DeviceCapabilities`` object."""
    gates = {}
    for name, op in capabilities.operations.items():
        props = set()
        if getattr(op, "invertible", False):
            props.add("invertible")
        if getattr(op, "controllable", False):
            props.add("controllable")
        gates[name] = props
    if include_channels:
        for name in CHANNEL_NAMES:
            gates.setdefault(name, set())
    return gates


def to_native_noise_model(
    noise_model,
    wires: Optional[Iterable],
    capabilities,
    *,
    max_tuples: int = 2000,
    exhaustive_arity: int = 3,
    sample_uniform: int = 4,
    max_arity: int = 8,
    default_num_wires: int = 8,
) -> str:
    """Serialize ``noise_model`` for the native backends.

    Args:
        noise_model (~pennylane.NoiseModel | str | None): the model; an ``LNM1``
            string is validated and returned unchanged.
        wires (Iterable | None): device wires. If ``None``, integer wires
            ``0..default_num_wires-1`` (or up to the largest wire named in the
            conditions) are probed.
        capabilities (DeviceCapabilities): device capabilities (gate names and
            properties).
    Returns:
        str: the serialized model (``""`` for no noise).
    """
    if noise_model is None:
        return ""
    if isinstance(noise_model, str):
        return validate_native_spec(noise_model)
    if wires is None:
        named = set()
        for c in list(noise_model.model_map) + list(noise_model.meas_map or {}):
            _collect_condition_wires(c, named)
        ints = [w for w in named if isinstance(w, int) and w >= 0]
        n = max([default_num_wires] + [w + 1 for w in ints])
        wires = list(range(n))
    conv = NoiseModelConverter(
        noise_model,
        list(wires),
        gates_from_capabilities(capabilities),
        max_tuples=max_tuples,
        exhaustive_arity=exhaustive_arity,
        sample_uniform=sample_uniform,
        max_arity=max_arity,
    )
    return conv.convert()
