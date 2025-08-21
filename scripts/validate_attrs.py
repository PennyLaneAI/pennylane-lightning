# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Internal logic to validate Lightning packages.
"""
import pprint
import re
from pathlib import Path
from zipfile import ZipFile

for path in Path().glob("*.whl"):  
    additional_packages = []

    with ZipFile(path, mode="r") as zf:
        names = zf.namelist()
        for name in names:

            # Skip development and documentation files
            if (
                name.endswith((".cpp", ".hpp", ".cu", ".py", ".txt", ".md", ".toml"))
                or ".dist-info/" in name
                or ".clang-tidy" in name
            ):
                continue

            # Specific libraries

            # Skip OpenMP libraries
            if re.match(r".*libgomp.*\.so.*", name) or re.match(r".*libomp.*\.dylib.*", name):
                continue
            # Skip Catalyst libraries
            if "catalyst.so" in name or "catalyst.dylib" in name or "catalyst.dll" in name:
                continue
            # Skip Lightning libraries
            if (
                re.match(
                    r"pennylane_lightning/lightning_.*_ops.cpython-3.?.?-.*-linux-gnu\.so", name
                )
                or re.match(r"pennylane_lightning/lightning_.*_ops.cpython-3.?.?-darwin\.so", name)
                or re.match(r"pennylane_lightning/lightning_.*_ops.pdb", name)
            ):
                continue

            # Skip directories paths
            if name.endswith("/"):
                continue

            additional_packages.append(name)
    # end with ZipFile

    if additional_packages:
        print(f"❌ Additional packages in {path.name}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pprint.pprint(additional_packages)
        print("===================================================================================")
    else:
        print(f"✅ No additional packages in {path.name}")
