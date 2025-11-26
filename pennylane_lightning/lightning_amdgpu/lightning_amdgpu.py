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
from pennylane_lightning.lightning_kokkos import LightningKokkos


class LightningAmdgpu(LightningKokkos):
    """PennyLane-Lightning AMDGPU device.

    A device alias for LightningKokkos targeting AMDGPU platforms.
    """

    name = "lightning.amdgpu"
    short_name = "lightning.amdgpu"

    def __init__(self, wires=None, *args, **kwargs):
        # Pass all arguments through to the parent Kokkos device
        super().__init__(wires, *args, **kwargs)
