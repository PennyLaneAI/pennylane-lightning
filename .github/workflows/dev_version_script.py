# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re
from pathlib import Path

VERSION_FILE_PATH = Path("pennylane_lightning/core/_version.py")

rgx_ver = re.compile(pattern=r"^__version__ = \"(.*)\"$", flags=re.MULTILINE)
rgx_dev_ver = re.compile(pattern=r"^(\d+)\.(\d+)\.(\d+)-dev(\d+)$")


def extract_version(repo_root_path: Path) -> str:
    """
    Given the repository root for pennylane-lightning, this function extracts the version from
    pennylane_lightning/core/_version.py.

    :param repo_root_path: Path to the repository root.
    :return: The extracted version string.
    """
    version_file_path = repo_root_path / VERSION_FILE_PATH
    if not version_file_path.exists():
        raise FileNotFoundError(f"Unable to find version file at location {version_file_path}")

    with version_file_path.open() as f:
        for line in f:
            if line.startswith("__version__"):
                if (m := rgx_ver.match(line.strip())) is not None:
                    return m.group(1)
                raise ValueError(f"Unable to find valid version for __version__. Got: '{line}'")
    raise ValueError("Cannot parse version")


def parse_dev_version(version: str) -> tuple[int, int, int, int] | None:
    """
    Splits an `X.Y.Z-devN` version into its numeric parts.

    :param version: The version string to parse.
    :return: A `(major, minor, patch, dev)` tuple, or None if the version is not a dev version.
    """
    match = rgx_dev_ver.match(version)
    return tuple(int(part) for part in match.groups()) if match else None


def update_prerelease_version(repo_root_path: Path, version: str):
    """
    Updates the version file within pennylane_lightning/core/_version.py.

    :param repo_root_path: Path to the repository root.
    :param version: The new version to use within the file.
    :return:
    """
    version_file_path = repo_root_path / VERSION_FILE_PATH
    if not version_file_path.exists():
        raise FileNotFoundError(f"Unable to find version file at location {version_file_path}")

    with version_file_path.open() as f:
        lines = [rgx_ver.sub(f'__version__ = "{version}"', line) for line in f]

    with version_file_path.open("w") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-path", dest="pr", type=Path, required=True, help="Path to the PR dir")
    parser.add_argument(
        "--main-path", dest="main", type=Path, required=True, help="Path to the main dir"
    )

    args = parser.parse_args()

    pr_version = extract_version(args.pr)
    main_version = extract_version(args.main)

    print("Got Package Version from 'main' ->", main_version)
    print("Got Package Version from 'PR' ->", pr_version)

    pr_parts = parse_dev_version(pr_version)
    main_parts = parse_dev_version(main_version)

    # Only attempt to bump the version if both the pull request and `main` are on an
    # `X.Y.Z-devN` version. We do not want to auto bump for non-dev versions.
    # However,
    #  If a PR is of a higher release AND the dev tag is reset, then do nothing
    #  This captures the case during release where we might bump the release version
    #  within a PR and reset the tag back to dev0
    if pr_parts is None:
        print("PR is not a dev prerelease ... Nothing to do!")
    elif main_parts is None:
        print(f"'main' is not on a dev prerelease ('{main_version}') ... Nothing to do!")
    elif pr_parts[:3] > main_parts[:3] and pr_parts[3] == 0:
        print(
            "This Pull Request is upgrading the package version to next release ... skipping bumping!"
        )
        print("If this is happening in error, please report it to the PennyLane team!")
    else:
        major, minor, patch, dev = main_parts
        new_version = f"{major}.{minor}.{patch}-dev{dev + 1}"
        if pr_version != new_version:
            print(f"Updating PR package version from -> '{pr_version}', to -> {new_version}")
            update_prerelease_version(args.pr, new_version)
        else:
            print(f"PR is on the expected version '{new_version}' ... Nothing to do!")
