import pprint
import re
from pathlib import Path
from zipfile import ZipFile

paths = Path().glob("**/*.whl")
for path in paths:
    additional_packages = []
    names = ZipFile(path).namelist()
    for name in names:

        # Development files
        if ".cpp" in name or ".hpp" in name or ".cu" in name or ".py" in name:
            continue
        # Documentation files
        if ".txt" in name or ".md" in name or ".dist-info/" in name:
            continue
        # Additional files
        if ".clang-tidy" in name or "toml" in name:
            continue

        # Specific libraries
        if re.match(r".*libgomp.*\.so.*", name) or re.match(r".*libomp.*\.dylib.*", name):
            continue
        if "catalyst.so" in name or "catalyst.dylib" in name or "catalyst.dll" in name:
            continue
        if (
            re.match(r"pennylane_lightning/lightning_.*_ops.cpython-3.?.?-.*-linux-gnu\.so", name)
            or re.match(r"pennylane_lightning/lightning_.*_ops.cpython-3.?.?-darwin\.so", name)
            or re.match(r"pennylane_lightning/lightning_.*_ops.pdb", name)
        ):
            continue

        # Directories paths
        if name.endswith("/"):
            continue

        additional_packages.append(name)

    if additional_packages:
        print(f"❌ Additional packages in {path.name}")
        pprint.pprint(additional_packages)
        print("=================================================")
    else:
        print(f"✅ No additional packages in {path.name}")
