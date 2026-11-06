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
This module contains the LightningCliffish class, the PennyLane entry point for a
near-Clifford simulator that runs only through Catalyst.
"""

from pathlib import Path

from pennylane.devices import Device, ExecutionConfig
from pennylane.exceptions import DeviceError
from pennylane.tape import QuantumScriptOrBatch
from pennylane.typing import Result, ResultBatch

from pennylane_lightning.lightning_base.lightning_base import LightningBase


class LightningCliffish(Device):
    """PennyLane Lightning Cliffish device.

    A near-Clifford simulator for programs of about a thousand qubits. The simulator
    is a Catalyst device library; this class only tells PennyLane and Catalyst where
    to find it and what it supports. The device runs only under
    ``qp.qjit(capture=True)``.

    Non-Clifford gates run in a small dense register next to the tableau. A program
    whose register would grow past ``max_dense_qubits`` fails at run time.

    Args:
        wires (int, Iterable[Number, str]): number of wires to initialize the device
            with, or iterable that contains unique labels for the wires
        max_dense_qubits (int): the largest dense register, in qubits. A register of
            n qubits holds 2^n complex amplitudes.
        recompression (str): how the register shrinks without reading amplitudes:
            ``"none"``, ``"t1"``, ``"t2"``, or ``"auto"`` to choose by a dry run.
        dense_checks (bool): whether to look for stabilizers of the register during
            the run and before each output, and fold them back into the tableau.
    """

    config_filepath = Path(__file__).parent / "lightning_cliffish.toml"

    _RECOMPRESSION = ("auto", "none", "t1", "t2")

    def __init__(
        self,
        wires=None,
        *,
        max_dense_qubits: int = 24,
        recompression: str = "auto",
        dense_checks: bool = True,
    ):
        if recompression not in self._RECOMPRESSION:
            raise ValueError(
                f"recompression must be one of {self._RECOMPRESSION}, not {recompression!r}"
            )
        if not 0 <= max_dense_qubits <= 40:
            raise ValueError(f"max_dense_qubits must be in [0, 40], not {max_dense_qubits}")
        super().__init__(wires=wires)
        self.device_kwargs = {
            "max_dense_qubits": int(max_dense_qubits),
            "recompression": recompression,
            "dense_checks": bool(dense_checks),
        }

    @property
    def name(self) -> str:
        """The name of the device."""
        return "lightning.cliffish"

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        """Refuse execution outside Catalyst.

        Raises:
            DeviceError: always, because lightning.cliffish runs only under
                ``qp.qjit(capture=True)``.
        """
        raise DeviceError("lightning.cliffish runs only under qp.qjit(capture=True).")

    @staticmethod
    def get_c_interface() -> tuple[str, str]:
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        return LightningBase.get_c_interface_impl(
            "LightningCliffishSimulator", "lightning_cliffish"
        )
