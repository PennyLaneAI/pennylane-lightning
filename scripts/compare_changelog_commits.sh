#!/usr/bin/env bash

# This script compares the list of merged PRs in Git with the list of PRs in the CHANGELOG.
# It prints the authors of the merged PRs and the PRs in the CHANGELOG, and highlights any discrepancies.

# Usage:
# Run the script with 
# bash compare_changelog_commits.sh 

# Root directory, should be the location of the pennylane-lightning repo
ROOT_DIR="."

# --------------------------------------------------------------------------------------------------
# Script body
# --------------------------------------------------------------------------------------------------

tmp_root_dir=$(pwd)
ROOT_DIR="$tmp_root_dir/$ROOT_DIR"

# Path to ChangeLog
CHANGELOG_FILE="$ROOT_DIR/.github/CHANGELOG.md"

# Test if CHANGELOG file exists
if [ ! -f "$CHANGELOG_FILE" ]; then
    echo "CHANGELOG file not found at $CHANGELOG_FILE"
    echo "Please make sure you are in the root of the pennylane-lightning repository"
    exit 1
fi

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "gh CLI is not installed. Please install it to run this script."
    exit 1
fi

# Last release date YYYY-MM-DD
LAST_RELEASE_DATE=$(gh release view --json publishedAt | jq -r '.publishedAt | split("T")[0]')

# Find the end of the current version section in the CHANGELOG
changelog_lower_bound=$(grep -n -- "---" "${CHANGELOG_FILE}" | head -n 1 | cut -d: -f1)

# Find the beginning of the Contributors section
contributors_begin=$(grep -n "<h3>Contributors ✍️</h3>" "${CHANGELOG_FILE}" | head -n 1 | cut -d: -f1)

extract_contributors_from_changelog() {
    # Extract the list of contributors from the CHANGELOG
    contribution_list=$(sed -n "$((contributors_begin + 4)),$((changelog_lower_bound - 2))p" "${CHANGELOG_FILE}") 
    # Sort and format the contribution list
    contribution_list=$(echo "$contribution_list" | sort | sed 's/.$/,/ ; s/,//; s/$/,/')
    echo "$contribution_list"
}

extract_contributors_from_git() {
    # Extract the list of contributors from Git
    contribution_list=$(git log master --since=$LAST_RELEASE_DATE --pretty=format:"%an <%ae>")
    # Sort, find unique entries, and format the contribution list
    contribution_list=$(echo "$contribution_list" | sort | uniq | sed 's|<.*|| ; s|Author: ||')
    echo "$contribution_list"
}

# Print list of Authors from Changelog and Git
echo "Authors in CHANGELOG       /  Authors in Git"
echo "---------------------------|---------------------------"
paste <(extract_contributors_from_changelog) \
      <(extract_contributors_from_git) \
      | column -t -s ','
echo "---------------------------|---------------------------"
echo "--------------------------------------------------------------------------------"

# Create the list of merged PR
list_merged_PRs(){
    # Get the list of merged PRs to master 
    list_PRs=$(gh pr list --state merged --base master --search "merged:>=$LAST_RELEASE_DATE" -L 1000 )
    # Extract the PR number and title, format it, and sort it
    list_PRs=$(echo "$list_PRs" | awk -F 'MERGED' '{print $1}' | sort -h)
    echo "$list_PRs" > release_list_merged_PR.txt
}

list_entries_in_changelog(){
    # Create a list of PRs in the CHANGELOG
    list_entries=$( sed -n "1,${changelog_lower_bound}p" "${CHANGELOG_FILE}" )
    # Extract only the PRs
    list_entries=$(echo "$list_entries" | grep -B 1 'pull/'  | tr -d "\n" )
    # Format the list of PRs, replace 'pull/' with 'pull/_', and remove trailing parentheses
    list_entries=$(echo "$list_entries" | sed 's/--/\n/g; s|pull/|pull/_ |g ; s/)//g')
    # Add the PR number at the beginning of the line
    list_entries=$(echo "$list_entries" |  awk '{print $NF, "  ", $0}') 
    # Remove the PR link and sort the entries
    list_entries=$(echo "$list_entries" | sed 's|\[(.*||' | sort -h -k1)
    echo "$list_entries" > release_list_PR_in_changelog.txt
}

# Create the list of entries in the CHANGELOG and the merged PRs
list_entries_in_changelog
list_merged_PRs 

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
