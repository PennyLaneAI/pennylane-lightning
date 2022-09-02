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
from pathlib import Path
import importlib

import re

VERSION_FILE_PATH = 'pennylane_lightning/_version.py'

rgx_ver = re.compile('^__version__ = \"(.*?)\"$')

rgx_dev_ver = re.compile('^(\d*\.\d*\.\d*)-dev(\d*)$')

def extract_version(package_path):
    with package_path.joinpath(VERSION_FILE_PATH).open('r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                line = line.strip()
                m = rgx_ver.match(line)
                return m.group(1)
    raise ValueError("Cannot parse version")

def is_dev(version_str):
    m = rgx_dev_ver.fullmatch(version_str)
    return m is not None

def update_dev_version(package_path, version_str):
    m = rgx_dev_ver.fullmatch(version_str)
    if m.group(2) == '':
        curr_dev_ver = 0
    else:
        curr_dev_ver = int(m.group(2))

    new_version_str = '{}-dev{}'.format(m.group(1), str(curr_dev_ver + 1))

    lines = []
    with package_path.joinpath(VERSION_FILE_PATH).open('r') as f:
        for line in f.readlines():
            if not line.startswith('__version__'):
                lines.append(line)
            else:
                lines.append(f'__version__ = \"{new_version_str}\"\n')

    with package_path.joinpath(VERSION_FILE_PATH).open('w') as f:
        f.write(''.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-path", dest = "pr", type=str, required=True, help="Path to the PR dir")
    parser.add_argument("--master-path", dest = "master", type=str, required=True, help="Path to the master dir")

    args = parser.parse_args()

    pr_version = extract_version(Path(args.pr))
    master_version = extract_version(Path(args.master))
    
    if pr_version == master_version:
        if is_dev(pr_version):
            print("Automatically update version string.")
            update_dev_version(Path(args.pr), pr_version)
        else:
            print("Even though version of this PR is different from the master, as the PR is not dev, we do nothing.")
    else:
        print("Version of this PR is already different from master. Do nothing.")
