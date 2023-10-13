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

r"""
This module contains the :class:`~.LightningQubit` class, a PennyLane simulator device that
interfaces with C++ for fast linear algebra calculations.
"""

from typing import Union, Sequence
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result

QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]

from warnings import warn
import numpy as np

from pennylane_lightning.core.lightning_base import (
    LightningBase,
    LightningBaseFallBack,
)

try:
    # pylint: disable=import-error, no-name-in-module
    from pennylane_lightning.lightning_qubit_ops import (
        StateVectorC64,
        StateVectorC128,
    )

    LQ_CPP_BINARY_AVAILABLE = True
except ImportError:
    LQ_CPP_BINARY_AVAILABLE = False

if LQ_CPP_BINARY_AVAILABLE:
    from ._simulate import get_final_state, measure_final_state

    from pennylane_lightning.core._adjoint_jacobian import AdjointJacobian

    class LightningQubit(LightningBase):
        """PennyLane Lightning device.

        A device that interfaces with C++ to perform fast linear algebra calculations.

        Use of this device requires pre-built binaries or compilation from source. Check out the
        :doc:`/installation` guide for more details.

        Args:
            c_dtype: Datatypes for state vector representation. Must be one of ``np.complex64`` or ``np.complex128``.
        """

        # pylint:disable = too-many-arguments
        def __init__(
            self,
            wires=None,
            shots=None,
            c_dtype=np.complex128,
            batch_obs=False,
        ) -> None:
            super().__init__(wires=wires, shots=shots, c_dtype=c_dtype, batch_obs=batch_obs)

        @property
        def name(self):
            """The name of the device."""
            return "lightning.qubit"

        def simulate(
            self, circuit: QuantumScript, c_dtype=np.complex128, rng=None, debugger=None
        ) -> Result:
            """Simulate a single quantum script.

            Args:
                circuit (QuantumTape): The single circuit to simulate
                rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                    seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                    If no value is provided, a default RNG will be used.
                debugger (_Debugger): The debugger to use

            Returns:
                tuple(TensorLike): The results of the simulation

            Note that this function can return measurements for non-commuting observables simultaneously.

            """
            state, is_state_batched = get_final_state(circuit, c_dtype, debugger=debugger)
            return measure_final_state(circuit, state, is_state_batched, c_dtype, rng=rng)

        def simulate_and_adjoint(
            self, circuit: QuantumScript, c_dtype=np.complex128, rng=None, debugger=None
        ):
            """Simulate a single quantum script and calculates the state gradient with the adjoint Jacobian method.

            Args:
                circuit (QuantumTape): The single circuit to simulate
                rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                    seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                    If no value is provided, a default RNG will be used.
                debugger (_Debugger): The debugger to use

            Returns:
                Results of the simulation and circuit gradient
            """
            state, is_state_batched = get_final_state(circuit, c_dtype, debugger=debugger)
            jac = AdjointJacobian("lightning.qubit").calculate_adjoint_jacobian(
                circuit, c_dtype, state, self._batch_obs
            )
            return measure_final_state(circuit, state, is_state_batched, c_dtype, rng=rng), jac

else:

    class LightningQubit(LightningBaseFallBack):  # pragma: no cover
        # pylint: disable=missing-class-docstring

        def __init__(self, c_dtype=np.complex128):
            warn(
                "Pre-compiled binaries for lightning.qubit are not available. Falling back to "
                "using the Python-based default.qubit implementation. To manually compile from "
                "source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html.",
                UserWarning,
            )
            self.C_DTYPE = c_dtype
            if self.C_DTYPE not in [np.complex64, np.complex128]:
                raise TypeError(f"Unsupported complex Type: {c_dtype}")

            super().__init__()
