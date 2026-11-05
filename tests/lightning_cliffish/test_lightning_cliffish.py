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
Unit tests for the LightningCliffish device.
"""

from pathlib import Path

import pennylane as qp
import pytest
from conftest import LightningDevice, device_name
from pennylane.devices.capabilities import DeviceCapabilities
from pennylane.exceptions import DeviceError

if device_name != "lightning.cliffish":
    pytest.skip("Skipping tests for the LightningCliffish class.", allow_module_level=True)


def test_device_name():
    """Test the device name."""
    assert LightningDevice(wires=2).name == "lightning.cliffish"


def test_execute_is_refused():
    """Test that the device refuses execution outside Catalyst."""
    with pytest.raises(DeviceError, match=r"runs only under qp.qjit\(capture=True\)"):
        LightningDevice(wires=2).execute([])


def test_c_interface():
    """Test that the C interface names the Catalyst device and an existing library."""
    device, library = LightningDevice.get_c_interface()
    assert device == "LightningCliffishSimulator"
    assert Path(library).is_file()


def test_capabilities():
    """Test that the TOML file declares a qjit-only device with no operations yet."""
    capabilities = DeviceCapabilities.from_toml_file(LightningDevice.config_filepath)
    assert capabilities.qjit_compatible
    assert not capabilities.operations
    assert not capabilities.measurement_processes
    assert not capabilities.supported_mcm_methods


def test_unsupported_measurement_fails_at_compile_time():
    """Test that Catalyst rejects measurements the TOML file does not declare."""
    pytest.importorskip("catalyst")
    from catalyst.utils.exceptions import CompileError

    @qp.qjit(capture=True)
    @qp.set_shots(10)
    @qp.qnode(LightningDevice(wires=2))
    def circuit():
        qp.Hadamard(0)
        return qp.sample(wires=[0, 1])

    with pytest.raises(CompileError, match="does not support"):
        circuit()


def test_runtime_fails_loudly(tmp_path: Path):
    """Test that Catalyst loads the device library and that the runtime fails loudly
    on the first operation."""
    pytest.importorskip("catalyst")

    toml = LightningDevice.config_filepath.read_text(encoding="utf-8")
    toml = toml.replace(
        "[operators.gates]\n", "[operators.gates]\nHadamard = { properties = [ ] }\n"
    )
    toml = toml.replace(
        "[measurement_processes]\n",
        '[measurement_processes]\nSampleMP = { conditions = [ "finiteshots" ] }\n',
    )
    config_filepath = tmp_path / "lightning_cliffish.toml"
    config_filepath.write_text(toml, encoding="utf-8")

    class ProbeDevice(LightningDevice):
        """LightningCliffish with a TOML file that declares one gate and sampling."""

    ProbeDevice.config_filepath = config_filepath

    @qp.qjit(capture=True)
    @qp.set_shots(10)
    @qp.qnode(ProbeDevice(wires=2))
    def circuit():
        qp.Hadamard(0)
        return qp.sample(wires=[0, 1])

    with pytest.raises(RuntimeError, match="cannot apply 'Hadamard' yet"):
        circuit()
