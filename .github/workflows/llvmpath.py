import os
import shutil
import subprocess
from pathlib import Path


if __name__ == "__main__":
    if os.getenv("BREW_LLVM_VERSION") and shutil.which("brew"):
        brew_llvm_version = os.getenv("BREW_LLVM_VERSION")
        llvmpath = subprocess.run(
            [
                "brew",
                "--prefix",
                "llvm" + f"@{brew_llvm_version}" if brew_llvm_version else "",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    else:
        # No brew, use the default clang++ install provided by MacOS
        llvmpath = shutil.which("clang++")
        llvmpath = Path(llvmpath).parent.parent
    print(llvmpath)
