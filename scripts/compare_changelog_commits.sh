#!/usr/bin/env bash

# This script compares the list of merged PRs in Git with the list of PRs in the CHANGELOG.
# It prints the authors of the merged PRs and the PRs in the CHANGELOG, and highlights any discrepancies.

# Usage:
# Edit the variable LAST_RELEASE_DATE with the date from https://github.com/PennyLaneAI/pennylane-lightning/releases
# Run the script with 
# bash compare_changelog_commits.sh 

# Last release date
LAST_RELEASE_DATE="2025-05-05"

# Path to ChangeLog
CHANGELOG_FILE="../.github/CHANGELOG.md"

# -------------------------------------------------------------------------------------------------
# Script body
# -------------------------------------------------------------------------------------------------

# Test if CHANGELOG file exists
if [ ! -f "$CHANGELOG_FILE" ]; then
    echo "CHANGELOG file not found at $CHANGELOG_FILE"
    exit 1
fi

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "gh CLI is not installed. Please install it to run this script."
    exit 1
fi

# Find the end of the current version section in the CHANGELOG
changelog_lower_bound=$(grep -n -- "---" "${CHANGELOG_FILE}" | head -n 1 | cut -d: -f1)

# Find the beginning of the Contributors section
contributors_begin=$(grep -n "<h3>Contributors ✍️</h3>" "${CHANGELOG_FILE}" | head -n 1 | cut -d: -f1)

# Print list of Authors from Changelog and Git
echo "Authors in CHANGELOG       /  Authors in Git"
echo "---------------------------|---------------------------"
paste <(sed -n "$((contributors_begin + 4)),$((changelog_lower_bound - 2))p" "${CHANGELOG_FILE}" | sort | sed 's/,//; s/$/,/') <(git log master --since=$LAST_RELEASE_DATE | grep "Author:" | sort | uniq | sed 's|<.*|| ; s|Author: ||' ) | column -t -s ','

echo "--------------------------------------------------------------------------------"

# Create the list of merged PR
gh pr list --state merged --base master --search "merged:>=$LAST_RELEASE_DATE" -L 1000  | sort -h | awk -F 'MERGED' '{print $1}' > release_list_merged_PR.txt

# Create the list of PRs in the CHANGELOG
sed -n "1,${changelog_lower_bound}p" "${CHANGELOG_FILE}" | grep -B 1 'pull/'  | tr -d "\n" | sed 's/--/\n/g; s|pull/|pull/_ |g ; s/)//g'| awk '{print $NF, "  ", $0}' | sed 's|\[(.*||' |   sort -h -k1 > release_list_PR_in_changelog.txt


interleave_rows(){
    file1="$1"
    file2="$2"

    spliter="---------------------------------------------------------------------"

    while IFS= read -r line1; do
    # Extract the first column from each line
    col1=$(echo "$line1" | awk '{print $1}')

    #  Check if col1 is in file2
    line2=$(grep "^$col1" "$file2")
    # If line2 is empty, it means col1 is not in file2
    if [ -z "$line2" ]; then
        # Print the line from file1 with an empty line for file2
        echo "MERGED | $line1"
        echo " "
        echo $spliter
        continue
    fi
    # Extract the first column from line2
    col2=$(echo "$line2" | awk '{print $1}')

        # Print the line from file1 and the corresponding line from file2

        echo "MERGED | $line1"
        echo "CHGLOG | $line2"
        echo $spliter
    done < "$file1"

    while IFS= read -r line2; do
    # Extract the first column from each line
    col2=$(echo "$line2" | awk '{print $1}')

    # Check if col2 is in file1
    line1=$(grep "^$col2" "$file1")
    # If line1 is empty, it means col2 is not in file1
    if [ -z "$line1" ]; then
        # Print the line from file2 with an empty line for file1
        echo " "
        echo "CHGLOG | $line2"
        echo $spliter
        continue
    fi
    done < "$file2"

}

# Compare the two lists
echo "--------------------------------------------------------------------------------"
echo "Merged PRs in Git / PRs in CHANGELOG"
interleave_rows release_list_merged_PR.txt release_list_PR_in_changelog.txt
