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
This module contains the :class:`~.LightningTensornet` class, a Kokkos
matrix-product-state simulator with native noise-model support.
"""

import ctypes
import numbers
import platform
from pathlib import Path
from typing import Optional
from warnings import warn

import numpy as np
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.exceptions import DeviceError
from pennylane.measurements import (
    DensityMatrixMP,
    MutualInfoMP,
    PurityMP,
    StateMP,
    VnEntropyMP,
)
from pennylane.wires import Wires

from pennylane_lightning.lightning_base._noisy_device import LightningNoisyBase


def _preload_scipy_openblas():
    """Load ``libscipy_openblas`` (used for the SVDs) from ``scipy_openblas32``.

    The backend opens the library by name and finds it through the RPATH
    ``$ORIGIN/../scipy_openblas32/lib``, which only resolves when the binaries
    are installed in site-packages next to ``scipy_openblas32``. Loading the
    library by absolute path first also covers editable installs (the binaries
    then live in the source tree): opening the same soname again returns the
    library that is already loaded.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import scipy_openblas32
    except ImportError:  # pragma: no cover
        return None
    name = "libscipy_openblas.dylib" if platform.system() == "Darwin" else "libscipy_openblas.so"
    path = Path(scipy_openblas32.get_lib_dir()) / name
    try:
        return ctypes.CDLL(str(path)) if path.exists() else None
    except OSError:  # pragma: no cover
        return None


_SCIPY_OPENBLAS = _preload_scipy_openblas()

try:
    # pylint: disable=import-error, no-name-in-module, unused-import
    from pennylane_lightning import lightning_tensornet_ops as _ops
    from pennylane_lightning.lightning_tensornet_ops import backend_info

    LT_CPP_BINARY_AVAILABLE = True
except ImportError as ex:
    warn(str(ex), UserWarning)
    _ops = None
    backend_info = None
    LT_CPP_BINARY_AVAILABLE = False


@simulator_tracking
@single_tape_support
class LightningTensornet(LightningNoisyBase):
    """PennyLane Lightning Tensornet device.

    A matrix-product-state (MPS) simulator built on Kokkos (OpenMP / Serial /
    CUDA / HIP / SYCL, fixed at build time). Noiseless circuits are simulated as
    a pure MPS. The first non-unitary channel switches the simulation, exactly,
    to a matrix-product density operator (MPDO: the vectorized density matrix
    stored as an MPS with physical dimension 4). Noise channels can be used
    directly in the circuit, or supplied as a noise model that is applied by
    the device and, under :func:`~pennylane.qjit`, by the Catalyst runtime.

    Bonds are truncated by SVD after every multi-qubit gate. Only singular
    values above ``cutoff`` are kept, up to ``max_bond_dim`` of them. The
    simulation is exact while no truncation happens, which is always the case
    when ``max_bond_dim >= 2**(N // 2)`` (pure) or ``4**(N // 2)`` (mixed) and
    ``cutoff == 0``.

    Use of this device requires compilation from source. Check out the
    :doc:`/lightning_tensornet/installation` guide for more details, and the
    :doc:`/lightning_tensornet/device` page for the supported features.

    Args:
        wires (int | Iterable | None): the number or labels of the wires; ``None``
            allocates the wires of each circuit.
        shots (int | list | None): default number of shots.
        seed (str | None | int | array_like | SeedSequence | BitGenerator | Generator):
            seed of the device random number generator (``"global"`` draws it from
            NumPy's global generator).
        c_dtype: ``np.complex128`` (default) or ``np.complex64``.
        max_bond_dim (int): largest bond dimension kept by SVD truncation.
        cutoff (float): singular values below the cutoff are discarded.
        cutoff_mode (str): ``"abs"`` (absolute cutoff) or ``"rel"`` (relative to
            the largest singular value of each SVD).
        noise_model (~pennylane.NoiseModel | str | None): noise model or serialized
            native noise model.
        noise_mode (str | None): ``"transform"`` or ``"native"``; see
            :mod:`pennylane_lightning.lightning_base._noisy_device`.
        kokkos_args (dict | None): Kokkos initialization settings.

    **Example**

    .. code-block:: python

        import pennylane as qp

        noise = qp.NoiseModel(
            {qp.noise.op_eq(qp.CNOT): qp.noise.partial_wires(qp.DepolarizingChannel, 0.01)}
        )
        dev = qp.device("lightning.tensornet", wires=40, max_bond_dim=64, noise_model=noise)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, 0)
            for i in range(39):
                qp.CNOT([i, i + 1])
            return qp.expval(qp.Z(39))
    """

    _state_class_name = "MPS"
    _catalyst_device_name = "LightningTensornetSimulator"
    _catalyst_lib_name = "lightning_tensornet"

    # This configuration file declares capabilities of the device
    config_filepath = Path(__file__).parent / "lightning_tensornet.toml"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires=None,
        *,
        shots=None,
        seed="global",
        c_dtype=np.complex128,
        max_bond_dim: int = 128,
        cutoff: float = 0.0,
        cutoff_mode: str = "abs",
        noise_model=None,
        noise_mode: Optional[str] = None,
        kokkos_args: Optional[dict] = None,
    ):
        if (
            isinstance(max_bond_dim, bool)
            or not isinstance(max_bond_dim, numbers.Integral)
            or max_bond_dim < 1
        ):
            raise ValueError(f"max_bond_dim must be a positive integer, got {max_bond_dim!r}.")
        if (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, numbers.Real)
            or not np.isfinite(cutoff)
            or cutoff < 0
        ):
            raise ValueError(f"cutoff must be a non-negative number, got {cutoff!r}.")
        if cutoff_mode not in ("abs", "rel"):
            raise ValueError(f"cutoff_mode must be 'abs' or 'rel', got {cutoff_mode!r}.")
        self._max_bond_dim = int(max_bond_dim)
        self._cutoff = float(cutoff)
        self._cutoff_mode = cutoff_mode
        super().__init__(
            wires=wires,
            shots=shots,
            seed=seed,
            c_dtype=c_dtype,
            noise_model=noise_model,
            noise_mode=noise_mode,
            kokkos_args=kokkos_args,
        )

    @property
    def name(self):
        """The name of the device."""
        return "lightning.tensornet"

    @property
    def max_bond_dim(self) -> int:
        """Largest bond dimension kept by SVD truncation."""
        return self._max_bond_dim

    @property
    def cutoff(self) -> float:
        """Singular-value cutoff."""
        return self._cutoff

    @property
    def cutoff_mode(self) -> str:
        """``"abs"`` or ``"rel"``."""
        return self._cutoff_mode

    @property
    def truncation_error(self) -> float:
        """Truncation error of the last simulated circuit: the sum over all
        truncating SVDs of the discarded fraction of the squared singular values
        (``0.0`` if nothing was truncated or nothing ran yet). For shot-based
        mid-circuit measurements this is the value of the last shot."""
        return 0.0 if self._state is None else float(self._state.truncation_error())

    @property
    def device_kwargs(self) -> dict:
        """Keyword arguments passed to the Catalyst runtime device."""
        kwargs = dict(super().device_kwargs)
        kwargs["max_bond_dim"] = self._max_bond_dim
        kwargs["cutoff"] = self._cutoff
        kwargs["cutoff_mode"] = self._cutoff_mode
        return kwargs

    @staticmethod
    def _ops_module():
        return _ops

    # ---- backend hooks ---------------------------------------------------

    def _new_state(self, num_wires: int):
        cls = getattr(_ops, "MPSC64" if self._c_dtype == np.complex64 else "MPSC128")
        return cls(
            num_wires,
            self._max_bond_dim,
            self._cutoff,
            self._cutoff_mode,
            False,
            self._kokkos_args,
        )

    def _state_measurement(self, state, mp, num_wires):
        if isinstance(mp, DensityMatrixMP):
            wires = [int(w) for w in mp.wires] or list(range(num_wires))
            return state.reduced_density_matrix(wires)
        if isinstance(mp, StateMP):
            if state.is_mixed():
                raise DeviceError(
                    f"The state of {self.name} is mixed (noise was applied), so "
                    "qml.state() is not defined; use qml.density_matrix(wires) instead."
                )
            return state.state_vector()
        if isinstance(mp, (PurityMP, VnEntropyMP, MutualInfoMP)):
            wires = [int(w) for w in mp.wires] or list(range(num_wires))
            rdm = state.reduced_density_matrix(wires)
            local = mp.map_wires({w: i for i, w in enumerate(wires)})
            return local.process_density_matrix(rdm, wire_order=Wires(range(len(wires))))
        raise DeviceError(f"Measurement {mp} is not supported on {self.name}")


_supports_operation = LightningTensornet.capabilities.supports_operation
