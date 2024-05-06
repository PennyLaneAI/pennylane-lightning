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

import re
import argparse
from pathlib import Path

try:
    from semver import Version
except ImportError as exc:
    raise ImportError("Unable to import semver. Install semver by running `pip install semver`") from exc

DEV_PRERELEASE_TAG_PREFIX = "dev"
DEV_PRERELEASE_TAG_START = "dev0"
VERSION_FILE_PATH = Path("pennylane_lightning/core/_version.py")

rgx_ver = re.compile(pattern=r"^__version__ = \"(.*)\"$", flags=re.MULTILINE)


def extract_version(repo_root_path: Path) -> Version:
    """
    Given the repository root for pennylane-lightning, this function extracts the semver version from
    pennylane_lightning/core/_version.py.

    :param repo_root_path: Path to the repository root.
    :return: Extracted version a semver.Version object.
    """
    version_file_path = repo_root_path / VERSION_FILE_PATH
    if not version_file_path.exists():
        raise FileNotFoundError(f"Unable to find version file at location {version_file_path}")

    with version_file_path.open() as f:
        for line in f:
            if line.startswith("__version__"):
                if (m := rgx_ver.match(line.strip())) is not None:
                    if not m.groups():
                        raise ValueError(f"Unable to find valid semver for __version__. Got: '{line}'")
                    parsed_semver = m.group(1)
                    if not Version.is_valid(parsed_semver):
                        raise ValueError(f"Invalid semver for __version__. Got: '{parsed_semver}' from line '{line}'")
                    return Version.parse(parsed_semver)
                raise ValueError(f"Unable to find valid semver for __version__. Got: '{line}'")
    raise ValueError("Cannot parse version")


def update_prerelease_version(repo_root_path: Path, version: Version):
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
        lines = [
            rgx_ver.sub(f"__version__ = \"{str(version)}\"", line)
            for line in f
        ]

    with version_file_path.open("w") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-path", dest="pr", type=Path, required=True, help="Path to the PR dir")
    parser.add_argument(
        "--master-path", dest="master", type=Path, required=True, help="Path to the master dir"
    )

    args = parser.parse_args()

    pr_version = extract_version(args.pr)
    master_version = extract_version(args.master)

    print("Got Package Version from 'master' ->", str(master_version))
    print("Got Package Version from 'PR' ->", str(pr_version))

    # Only attempt to bump the version if the pull_request is:
    #  - A prerelease, has `X.Y.Z-prerelease` in _version.py
    #  - The prerelease startswith `dev`. We do not want to auto bump for non-dev prerelease.
    # However,
    #  If a PR is of a higher version AND the prerelease tag is reset, then do nothing
    #  This captures the case during release where we might bump the release version
    #  within a PR and reset tag back to dev0
    if pr_version > master_version and pr_version.prerelease and pr_version.prerelease == DEV_PRERELEASE_TAG_START:
        print("This Pull Request is upgrading the package version to next release ... skipping bumping!")
        print("If this is happening in error, please report it to the PennyLane team!")
    elif pr_version.prerelease and pr_version.prerelease.startswith(DEV_PRERELEASE_TAG_PREFIX):
        # If master branch does not have a prerelease (for any reason) OR does not have an ending number
        # Then default to the starting tag
        if not master_version.prerelease or master_version.prerelease == DEV_PRERELEASE_TAG_PREFIX:
            next_prerelease_version = DEV_PRERELEASE_TAG_START
        else:
            # Generate the next prerelease version (eg: dev1 -> dev2). Sourcing from master version.
            next_prerelease_version = master_version.next_version("prerelease").prerelease
        new_version = master_version.replace(prerelease=next_prerelease_version)
        if pr_version != new_version:
            print(f"Updating PR package version from -> '{pr_version}', to -> {new_version}")
            update_prerelease_version(args.pr, new_version)
        else:
            print(f"PR is on the expected version '{new_version}' ... Nothing to do!")
    else:
        print("PR is not a dev prerelease ... Nothing to do!")
