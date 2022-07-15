# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
import pennylane as qml
pl_version = '"' + qml.version() + '"'


def remove_quotes(my_str):
    """ A helper function to remove the quote chars (',")
    from the provided str.
    """
    clean_str = my_str.replace('"', '')  # remove "
    clean_str = clean_str.replace("'", "")  # remove '
    return clean_str


def bump_version(version_line, pre_release):
    """ A helper function which takes the current version string and
    replaces it with the bumped version depending on the pre/post
    release flag.

    Args:
         version_line (string): The string containing the current
            version of the plugin.
         pre_release (bool): A flag which determines if this is a
            pre-release or post-release version bump.

    Returns:
        resultant_line (string): A string of the same form as the version line
        with the version number replaced with the bumped version.
        bumped_version (string): The bumped version string.
    """
    data = version_line.split(" ")
    curr_version = data[-1].replace('\n', '')  # remove any newline characters

    if pre_release:
        curr_version = pl_version  # get current Pennylane version

    split_version = curr_version.split(".")  # "0.17.0" --> ["0,17,0"]
    split_version[1] = str(int(split_version[1]) + 1)  # take middle value and cast as int and bump it by 1

    if not pre_release:
        split_version[2] = split_version[2].replace('"', '-dev"')  # add -dev, ["0,18,0"] --> ["0,18,0-dev"]

    bumped_version = ".".join(split_version)
    data[-1] = bumped_version
    clean_bumped_version = remove_quotes(bumped_version)

    return " ".join(data), clean_bumped_version


def update_version_file(path, pre_release=True):
    """ Updates the __version__ attribute in a specific version file.

    Args:
        path (str): The path to the version file.
        pre_release (bool): A flag which determines if this is a
            pre-release or post-release version bump.

    Return:
        new_version (str): The bumped version string.
    """
    with open(path, 'r', encoding="utf8") as f:
        lines = f.readlines()

    with open(path, 'w', encoding="utf8") as f:
        for line in lines:
            if "__version__" in line.split(' '):
                new_line, new_version = bump_version(line, pre_release)
                f.write(new_line + '\n')
            else:
                f.write(line)
    return new_version


def remove_empty_headers(lines):
    """ Takes a paragraph (list of strings) and removes sections which are empty.
    Where a section begins with a header (### Header_Title).

    Args:
        lines (list[string]): The paragraph containing the changelog sections.

    Returns:
        cleaned_lines (list[string]): The paragraph with empty sections removed.
    """
    cleaned_lines = []
    pntr1 = 0

    while pntr1 < len(lines):
        is_empty = True
        for pntr2 in range(pntr1 + 1, len(lines)):
            line2 = lines[pntr2]

            if (len(line2) >= 4) and (line2[:4] == "### "):
                if (pntr1 == 0) or (not is_empty):
                    cleaned_lines.extend(lines[pntr1:pntr2])  # keep these sections!

                pntr1 = pntr2
                is_empty = True  # reset the empty flag

            elif line2 == '\n':
                pass

            else:
                is_empty = False

        cleaned_lines.extend(lines[pntr1:pntr1+1])
        pntr1 += 1

    return cleaned_lines


def update_changelog(path, new_version, pre_release=True):
    """ Updates the Changelog file depending on whether it's a pre-release
    or post-release version bump.

    Args:
        path (str): The path to the changelog file.
        new_version (str): The bumped version string.
        pre_release (bool): A flag which determines if this is a
            pre-release or post-release version bump.
    """
    with open(path, 'r', encoding="utf8") as f:
        lines = f.readlines()
        end_of_section_index = 0
        for index, line in enumerate(lines):
            if (len(line) >= 3) and (line[:3] == "---"):
                end_of_section_index = index
                break

    with open(path, 'w', encoding="utf8") as f:
        if not pre_release:  # post_release append template to top of the changelog
            with open("./.github/workflows/changelog_template.txt", 'r', encoding="utf8") as template_f:
                template_lines = template_f.readlines()
                template_lines[0] = template_lines[0].replace('x.x.x-dev', new_version)
                f.writelines(template_lines)
                f.writelines(lines)

        else:  # pre_release update the release header and remove any empty headers
            # update release header
            line = lines[0]
            split_line = line.split(" ")
            split_line[-1] = new_version  # replace version (split_line = [#, Release, 0.17.0-dev])
            new_line = " ".join(split_line) + '\n'
            f.write(new_line)

            # remover empty headers
            cleaned_lines = remove_empty_headers(lines[1:end_of_section_index])
            f.writelines(cleaned_lines)

            # keep the rest of the changelog
            rest_of_changelog_lines = lines[end_of_section_index:]
            f.writelines(rest_of_changelog_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version_path", type=str, required=True, help="Path to the _version.py file")
    parser.add_argument("--changelog_path", type=str, required=True, help="Path to the changelog")
    parser.add_argument("--pre_release", dest="release_status", action="store_true",
                        help="True if this is a pre-release version bump, False if it is post release")
    parser.add_argument("--post_release", dest="release_status", action="store_false",
                        help="True if this is a pre-release version bump, False if it is post release")

    args = parser.parse_args()
    updated_version = update_version_file(args.version_path, args.release_status)
    update_changelog(args.changelog_path, updated_version, args.release_status)
