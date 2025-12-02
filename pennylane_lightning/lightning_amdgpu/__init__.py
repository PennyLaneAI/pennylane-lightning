# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PennyLane lightning_amdgpu package."""

import sys

from pennylane_lightning.core import __version__

try:
    from pennylane_lightning import lightning_amdgpu_ops

    sys.modules["pennylane_lightning.lightning_kokkos_ops"] = lightning_amdgpu_ops
    sys.modules["pennylane_lightning.lightning_kokkos_ops.algorithms"] = (
        lightning_amdgpu_ops.algorithms
    )
    sys.modules["pennylane_lightning.lightning_kokkos_ops.observables"] = (
        lightning_amdgpu_ops.observables
    )
except ImportError:
    pass

from .lightning_amdgpu import LightningAmdgpu
