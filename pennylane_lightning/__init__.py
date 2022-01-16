# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Top level PennyLane-Lightning module."""

import platform, os, sys

def load_lightning_ops():
    """Load lightning_qubit_ops shared object. Supports different Python versions as well as
    different platforms. We suppose this function is only called once.
    """
    import importlib

    # Add the current directory to DLL path.
    # See https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew
    if platform.system() == "Windows" and sys.version_info[:2] >= (3, 8):  # pragma: no cover
        libdir = os.path.dirname(os.path.abspath(__file__))
        os.add_dll_directory(libdir)
        try:
            return importlib.import_module("lightning_qubit_ops")
        except ModuleNotFoundError:
            pass  # Just cannot find a module
        except ImportError:
            # This error means Python cannot load lightning_qubit_ops dependencies
            warn(
                "Pre-compiled binaries are found but failed to load DLLs "
                "the library depends on. Check DLL paths. ",
                UserWarning,
            )
            return None
        # Even after add_dll_directory, some environment does not find dll object
        # (see e.g. https://github.com/zeromq/pyzmq/pull/1498). In This case,
        # we add libdir directly to env["PATH"].
        try:
            sys.path.insert(0, libdir)
            lightning_ops_module = importlib.import_module("lightning_qubit_ops")
            sys.path = sys.path[1:]
            return lightning_ops_module
        except ModuleNotFoundError:
            return None
    try:
        return importlib.import_module(".lightning_qubit_ops", __name__)
    except ModuleNotFoundError:  # ImportError is raises when DLL load is failed
        pass
    return None

lightning_ops_module = load_lightning_ops()

from ._version import __version__
from .lightning_qubit import LightningQubit
