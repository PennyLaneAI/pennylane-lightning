#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# This script automates the creation of a release candidate branch for PennyLane-Lightning.
#
# Prerequisites:
# - Be on the latest master branch in the root directory of pennylane-lightning/
# - Have 'gh' CLI installed and authenticated with GitHub
# - Have 'jq' installed for JSON processing
#
# Usage Instructions:
# The script supports different phases of the release process. The script should run on the root
# directory of the pennylane-lightning repository and in master branch.
# Run them in sequence:
#
# 1. Create Release Candidate (creates branches and PRs):
#    bash scripts/create_lightning_rc.sh -s 0.42.0 -r 0.43.0 -n 0.44.0 --create_rc
#
# 2. Test Lightning Installation (validates RC build):
#    bash scripts/create_lightning_rc.sh -s 0.42.0 -r 0.43.0 -n 0.44.0 --lightning_test
#
# 3. Create Release (creates GitHub release):
#    bash scripts/create_lightning_rc.sh -s 0.42.0 -r 0.43.0 -n 0.44.0 --release
#
# 4. Handle Release Assets (upload wheels and source distributions):
#    bash scripts/create_lightning_rc.sh -s 0.42.0 -r 0.43.0 -n 0.44.0 --release_assets
#
# Version flags:
# -s/--stable_version: Current stable release (e.g., 0.42.0)
# -r/--release_version: Version being released (e.g., 0.43.0)
# -n/--next_version: Next development version (e.g., 0.44.0)
#
# Use the --help option to see all available options.

# Set version numbers
STABLE_VERSION=0.42.0     # Current stable version | https://github.com/PennyLaneAI/pennylane-lightning/releases
RELEASE_VERSION=0.43.0    # Upcoming release version | https://test.pypi.org/project/pennylane-lightning/#history
NEXT_VERSION=0.44.0       # Next version to be developed | RELEASE_VERSION + 1

IS_TEST=true

# Debug option
# To avoid pushing any branch or PR to GitHub. Set to true
LOCAL_TEST=true

# Check if gh CLI, and jq are installed
if ! command -v gh &> /dev/null; then
    echo "gh CLI could not be found"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "jq could not be found"
    exit 1
fi

# --------------------------------------------------------------------------------------------------
# Script functions
# --------------------------------------------------------------------------------------------------
help(){

    echo "Usage: create_lightning_rc.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --stable_version [version]    Specify the stable version. Default $STABLE_VERSION"
    echo "  -r, --release_version [version]   Specify the release version. Default $RELEASE_VERSION"
    echo "  -n, --next_version [version]      Specify the new version. Default $NEXT_VERSION"
    echo "  -t, --test                        Run on test version, gh pr create with --dry-run. Default $IS_TEST"
    echo "  --create_rc                       Create a release candidate"
    echo "  --lightning_test                  Run Lightning tests"
    echo "  --release                         Perform release actions on GitHub"
    echo "  --release_assets                  Handle release assets to upload"
    echo "  -h, --help                        Show this help message"
}

ROOT_DIR="."

if [ ! -d "${ROOT_DIR}/.git" ]; then
    echo "You should to run the script on the root directory of the repository"
    exit 1
fi

CHANGELOG_FILE="$ROOT_DIR/.github/CHANGELOG.md"
PL_VERSION_FILE="$ROOT_DIR/pennylane_lightning/core/_version.py"

# Utils functions
use_dry_run(){
    # Check if the script is running for testing. If so, use the --dry-run flag.
    dry_run="--draft"

    if [ "$LOCAL_TEST" == "true" ]; then
        dry_run="--dry-run"
    fi

    echo $dry_run
}

branch_name() {
    local version=$1
    local suffix=$2
    local branch

    # The expression ${suffix:+_${suffix}} adds "_$suffix" ONLY if suffix is not empty.
    branch="v${version}${suffix:+_${suffix}}"

    if [ "$IS_TEST" == "true" ]; then
        # The same logic applies here for the test branch.
        branch="test_v${version}${suffix:+_${suffix}}_alpha"
    fi

    # Convert to lowercase at the end.
    echo "$branch" | tr '[:upper:]' '[:lower:]'
}

test_pennylane_version(){
    # Test if the Pennylane Lightning version is correct

    backends=(
        "lightning.qubit"
        "lightning.gpu"
        "lightning.kokkos"
        "lightning.tensor"
    )

    while IFS= read -r line; do
        backend=$(echo "$line" | awk '{print $2}')
        if [[ " ${backends[*]} " == *" $backend "* ]]; then
            if [[ "$line" == *"$RELEASE_VERSION"* ]]; then
                echo "✅  Correct version for backend: $backend"
            else
                echo "❌  Wrong version for backend: $backend"
            fi
            # Remove backend from list
            backends=("${backends[@]/$backend}")
            backends=($(printf "%s\n" "${backends[@]}" | grep -v '^$'))
        else
            echo "Unknown backend: $backend"
            continue
        fi
    done <<< $(python -c "import pennylane as qml; qml.about(); exit()"  | grep -- '- lightning')

    # If list is not empty, print remaining backends
    if [[ ${#backends[@]} -gt 0 ]]; then
        echo "⚠️  Missing backends: " "${backends[@]}"
    fi
}

create_release_notes(){
    # Create the release notes for the release
    changelog_lower_bound=$(grep -n -- "---" "${CHANGELOG_FILE}" | head -n 1 | cut -d: -f1)
    sed -n "1,${changelog_lower_bound}p" $CHANGELOG_FILE | sed ':a;N;$!ba;s/\.\n *\[(#/\. \[(#/g' > ${ROOT_DIR}/release_notes.md
    sed -i 's|^- |* |' ${ROOT_DIR}/release_notes.md
}

add_CHANGELOG_entry(){
    # Add a new entry to the CHANGELOG
    local changelog_entry=$1
    local PR_number=$2

sed -i "/<h3>Internal changes ⚙️<\/h3>/a \\
\\
- $changelog_entry\\
  [(#${PR_number})](https://github.com/PennyLaneAI/pennylane-lightning/pull/${PR_number})" $CHANGELOG_FILE
}

# Release functions

create_release_candidate_branch() {
    # Create a new branch for the release candidate

    # Clean up and checkout master
    git reset --hard
    git checkout master
    git pull origin master

    # Create branches
    for branch in base docs rc; do
        git checkout -b $(branch_name ${RELEASE_VERSION} ${branch})
        if [ "$LOCAL_TEST" == "false" ]; then
        git push --set-upstream origin $(branch_name ${RELEASE_VERSION} ${branch})
        fi
    done
    git checkout $(branch_name ${RELEASE_VERSION} rc)

    # Update lightning version
    sed -i "/__version__/d" $PL_VERSION_FILE
    if [ "$IS_TEST" == "true" ]; then
        dev_number=$(git show master:$PL_VERSION_FILE | grep "version" | grep -oP 'dev\K[0-9]+')
        echo '__version__ = "'${RELEASE_VERSION}'-rc0-dev'${dev_number}'"' >> $PL_VERSION_FILE
    else
        echo '__version__ = "'${RELEASE_VERSION}'-rc0"' >> $PL_VERSION_FILE
    fi
    sed -i "s|Release ${RELEASE_VERSION}-dev (development release)|Release ${RELEASE_VERSION}|g" $CHANGELOG_FILE

    git add $PL_VERSION_FILE $CHANGELOG_FILE
    git commit -m "Create v${RELEASE_VERSION} RC branch."

    # Update PennyLane dependency
    pushd $ROOT_DIR
    for file in requirements-dev.txt requirements-tests.txt; do
        sed -i "s|pennylane.git@master|pennylane.git@v${RELEASE_VERSION}-rc0|g" $file
        git add $file
    done
    popd
    git commit -m "Target PennyLane v${RELEASE_VERSION}-rc0 in requirements-[dev|tests].txt."

    # Update Catalyst dependency
    last_catalyst_commit=$(git ls-remote git@github.com:PennyLaneAI/catalyst.git HEAD | cut -f 1)
    sed -i 's|CATALYST_GIT_TAG "main"|CATALYST_GIT_TAG "'${last_catalyst_commit}'"|g' ${ROOT_DIR}/cmake/support_catalyst.cmake
    git add ${ROOT_DIR}/cmake/support_catalyst.cmake
    git commit -m "Set Catalyst dependency in cmake to commit ${last_catalyst_commit}."

    if [ "$LOCAL_TEST" == "false" ]; then
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} rc)
    fi
}

create_release_candidate_PR(){
    # Create a PR for the release candidate branch

    git checkout $(branch_name ${RELEASE_VERSION} rc)
    if [ "$LOCAL_TEST" == "false" ]; then
    gh pr create $(use_dry_run) \
        --title "Create v${RELEASE_VERSION} RC branch" \
        --body "v${RELEASE_VERSION} RC branch." \
        --head $(branch_name ${RELEASE_VERSION} rc) \
        --base $(branch_name ${RELEASE_VERSION} base) \
        --label 'do not merge','ci:build_wheels','ci:use-multi-gpu-runner','ci:use-gpu-runner','urgent'
    fi
}

create_docs_review_PR(){
    # Create a PR for the docs review

    git checkout $(branch_name ${RELEASE_VERSION} docs)

    git commit -m "Modify docs for v${RELEASE_VERSION}" --allow-empty

    if [ "$LOCAL_TEST" == "false" ]; then
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} docs)
    fi

    if [ "$LOCAL_TEST" == "false" ]; then
    gh pr create $(use_dry_run) \
        --title "Create v${RELEASE_VERSION} Doc branch" \
        --body "v${RELEASE_VERSION} Doc branch." \
        --head $(branch_name ${RELEASE_VERSION} docs) \
        --base $(branch_name ${RELEASE_VERSION} rc) \
        --draft \
        --label 'do not merge','documentation'
    fi
}

create_docker_PR(){
    # Create a PR for the Docker test in PTM

    git checkout master
    git checkout -b $(branch_name ${RELEASE_VERSION} docker)

    sed -i "s|v${STABLE_VERSION}|v${RELEASE_VERSION}|g" ${ROOT_DIR}/.github/workflows/compat-docker-release.yml

    git add ${ROOT_DIR}/.github/workflows/compat-docker-release.yml
    git commit -m "Update compat-docker-release.yml to use v${RELEASE_VERSION}"

    add_CHANGELOG_entry "Test Docker images for v${RELEASE_VERSION} RC branch." "0000"
    git add $CHANGELOG_FILE
    git commit -m "Add CHANGELOG entry for Docker test"

    if [ "$LOCAL_TEST" == "false" ]; then
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} docker)

    gh pr create $(use_dry_run) \
        --title "Docker test for v${RELEASE_VERSION} RC branch" \
        --body "Docker test for v${RELEASE_VERSION} RC branch." \
        --head $(branch_name ${RELEASE_VERSION} docker) \
        --base master \
        --label 'urgent'
    fi
}

new_changelog_entry(){

PR_number=$1
PR_author=$2

new_changelog_text=$(
cat <<EOF
# Release ${NEXT_VERSION}-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Internal changes ⚙️</h3>

- Bumped the version.
    [(#${PR_number})](https://github.com/PennyLaneAI/pennylane-lightning/pull/${PR_number})

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

${PR_author}

---

EOF
)
echo "$new_changelog_text"
}

create_version_bump_PR(){
    # Create a PR for the new version

    git checkout master
    git checkout -b $(branch_name ${RELEASE_VERSION} bump)

    # Update lightning version
    sed -i "/__version__/d" $PL_VERSION_FILE
    if [ "$IS_TEST" == "true" ]; then
        echo '__version__ = "'${NEXT_VERSION}'-alpha1-dev0"' >> $PL_VERSION_FILE
    else
        echo '__version__ = "'${NEXT_VERSION}'-dev0"' >> $PL_VERSION_FILE
    fi

    git add $PL_VERSION_FILE
    git commit -m "Bump version to v${NEXT_VERSION}."

    if [ "$LOCAL_TEST" == "false" ]; then
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} bump)

    gh pr create $(use_dry_run) \
        --title "Bump version to v${NEXT_VERSION}-dev" \
        --body "Bump version to v${NEXT_VERSION}-dev." \
        --head $(branch_name ${RELEASE_VERSION} bump) \
        --base master \
        --label 'urgent'
    fi

    if [ "$LOCAL_TEST" == "true" ]; then
        PR_number="0000"
        PR_author="PennyLaneAI"
    else
        PR_author=$(gh pr view --json author --jq '.author.login')
        PR_number=$(gh pr view --json number --jq .number)
    fi

    new_changelog_text=$(new_changelog_entry "${PR_number}" "${PR_author}")

    # Update CHANGELOG with new_changelog_entry
    {
        echo "$new_changelog_text"
        echo ""
        cat $CHANGELOG_FILE
    } > ${ROOT_DIR}/temp_changelog.md && mv ${ROOT_DIR}/temp_changelog.md $CHANGELOG_FILE

    sed -i "s|Release ${RELEASE_VERSION}-dev (development release)|Release ${RELEASE_VERSION}|g" $CHANGELOG_FILE

    git add $CHANGELOG_FILE
    git commit -m "Update CHANGELOG.md with new version entry."

    # Update minimum PennLane version in requirements.txt and configure_pyproject_toml.py
    sed -i "s/pennylane>=v\?[0-9\.]\+/pennylane>=${STABLE_VERSION%??}/" ${ROOT_DIR}/requirements.txt
    sed -i "s/pennylane>=v\?[0-9\.]\+/pennylane>=${STABLE_VERSION%??}/" ${ROOT_DIR}/scripts/configure_pyproject_toml.py
    sed -i "s/pennylane>=v\?[0-9\.]\+/pennylane>=${STABLE_VERSION%??}/" ${ROOT_DIR}/pyproject.toml

    git add ${ROOT_DIR}/requirements.txt
    git add ${ROOT_DIR}/scripts/configure_pyproject_toml.py
    git add ${ROOT_DIR}/pyproject.toml

    git commit -m "Update minimum PennyLane version to ${RELEASE_VERSION%??}"

    # Update RNG salt
    for i in ${ROOT_DIR}/tests/pytest.ini ${ROOT_DIR}/mpitests/pytest.ini ; do
        sed -i "/rng_salt = /d" $i
        echo "rng_salt = v${RELEASE_VERSION}" >> $i
        git add $i
    done
    git commit -m "Set rng_salt to v${RELEASE_VERSION} in tests/pytest.ini and mpitests/pytest.ini."

    if [ "$LOCAL_TEST" == "false" ]; then
    git push origin $(branch_name ${RELEASE_VERSION} bump)
    fi
}

test_install_lightning(){
    # Test Lightning installation

    git checkout master
    git checkout $(branch_name ${RELEASE_VERSION} rc)

    # Test installation of lightning default backends
    pip install -r requirements-dev.txt
    for backend in qubit gpu kokkos tensor; do
        PL_BACKEND=lightning_${backend} python ${ROOT_DIR}/scripts/configure_pyproject_toml.py
        PL_BACKEND=lightning_${backend} python -m pip install . -v
    done

    # Test import
    is_installed_backend=$(test_pennylane_version)

    # Test installation of lightning custom compiles options

    # Lightning Kokkos and MPI
    pip uninstall -y pennylane_lightning_kokkos
    PL_BACKEND="lightning_kokkos" python ${ROOT_DIR}/scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DENABLE_MPI=ON -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_OPENMP=ON" python -m pip install . -v

    # Test import
    is_installed_kokkos_mpi=$(test_pennylane_version | grep lightning.kokkos)

    # Lightning GPU with MPI
    pip uninstall -y pennylane_lightning_gpu
    PL_BACKEND="lightning_gpu" python ${ROOT_DIR}/scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DENABLE_MPI=ON" python -m pip install . -v

    # Test import
    is_installed_gpu_mpi=$(test_pennylane_version| grep lightning.gpu)

    echo "Installed backends:"
    echo "- Lightning Default:"
    echo "$is_installed_backend"
    echo "- Lightning Kokkos (MPI):"
    echo "$is_installed_kokkos_mpi"
    echo "- Lightning GPU (MPI):"
    echo "$is_installed_gpu_mpi"
}

download_artifacts_gh(){
    # Download the artifacts from the GitHub Actions runs

    wheels_runners=$(gh run list --branch $(branch_name ${RELEASE_VERSION} rc) \
        --json status,workflowName,databaseId | jq '.[] | select(.workflowName | contains("Wheel"))')

    completed_runners=$(echo "$wheels_runners" | jq -r '. | select(.status == "completed") | .databaseId')

    incomplete_runners=$(echo "$wheels_runners" | jq -r '. | select(.status != "completed") ')

    mkdir -p ${ROOT_DIR}/Wheels

    for runner in $completed_runners; do
        echo "Downloading artifacts for runner: $runner"
        gh run download --dir ${ROOT_DIR}/Wheels $runner
    done

    echo "Incomplete runner found:"
    echo "$incomplete_runners" | jq .
}

test_wheels_for_unwanted_libraries(){
    # Test for unwanted libraries in the wheels

    pushd Wheels

    for wheel in *.zip/*.whl; do
        cp "$wheel" .
    done

    python ${ROOT_DIR}/scripts/validate_attrs.py

    popd
}

create_release_branch(){
    # Create the release branch

    git checkout $(branch_name ${RELEASE_VERSION} rc)

    if [ "$LOCAL_TEST" == "false" ]; then
    gh pr comment $(branch_name ${RELEASE_VERSION} rc) \
        --body "Forked as v${RELEASE_VERSION}_release to be released with tag v${RELEASE_VERSION}"
    fi

    # Create the release branch
    git checkout -b $(branch_name ${RELEASE_VERSION} release)

    # Update version
    sed -i "/__version__/d" $PL_VERSION_FILE
    if [ "$IS_TEST" == "true" ]; then
        echo '__version__ = "'${RELEASE_VERSION}'-alpha"' >> $PL_VERSION_FILE
    else
        echo '__version__ = "'${RELEASE_VERSION}'"' >> $PL_VERSION_FILE
    fi

    git add $PL_VERSION_FILE
    git commit -m "Pre-release updates"
    if [ "$LOCAL_TEST" == "false" ]; then
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} release)
    fi
}

create_GitHub_release(){
    # Create the GitHub release as draft
    git checkout $(branch_name ${RELEASE_VERSION} release)

    create_release_notes

    # Create tag
    git tag -a "$(branch_name ${RELEASE_VERSION})" -m "Release ${RELEASE_VERSION}"
    if [ "$LOCAL_TEST" == "false" ]; then
    git push origin "$(branch_name ${RELEASE_VERSION})"

    gh release create $(branch_name ${RELEASE_VERSION}) \
        --target $(branch_name ${RELEASE_VERSION} release) \
        --title "Release ${RELEASE_VERSION}" \
        --notes-file ${ROOT_DIR}/release_notes.md \
        --draft --latest
    fi
    rm ${ROOT_DIR}/release_notes.md
}

download_release_artifacts_gh(){
    # Download the artifacts from the GitHub Actions runs
    wheels_runners=$(gh run list --event release --branch $(branch_name ${RELEASE_VERSION}) \
        --json status,workflowName,databaseId | jq '.[] | select(.workflowName | contains("Wheel"))')

     completed_runners=$(echo "$wheels_runners" | jq -r '. | select(.status == "completed") | .databaseId')
     incomplete_runners=$(echo "$wheels_runners" | jq -r '. | select(.status != "completed") ')

     mkdir -p ${ROOT_DIR}/Release_Assets

     for runner in $completed_runners; do
         echo "Downloading artifacts for runner: $runner"
         gh run download --dir ${ROOT_DIR}/Release_Assets $runner
     done

     echo "Incomplete runner found:"
     echo "$incomplete_runners" | jq .
}

create_sdist(){
    # Create the source distribution

    git checkout $(branch_name ${RELEASE_VERSION} "release")

    mkdir -p ${ROOT_DIR}/Release_Assets

    for backend in qubit gpu kokkos tensor; do
        PL_BACKEND=lightning_${backend} python ${ROOT_DIR}/scripts/configure_pyproject_toml.py
        python setup.py sdist
    done

    cp dist/*.tar.gz ${ROOT_DIR}/Release_Assets/
}

upload_release_assets_gh(){
    # Upload the release assets
    gh release upload $(branch_name ${RELEASE_VERSION}) ${ROOT_DIR}/Release_Assets/*.whl --clobber
    gh release upload $(branch_name ${RELEASE_VERSION}) ${ROOT_DIR}/Release_Assets/*.tar.gz --clobber
}

create_merge_branch(){
    # Create the merge branch to merge the RC into master and bump the version with NEXT_VERSION-dev

    git checkout $(branch_name ${RELEASE_VERSION} "release")
    git checkout -b $(branch_name ${RELEASE_VERSION} "rc_merge")

    pushd $ROOT_DIR
    for file in requirements-dev.txt requirements-tests.txt; do
        sed -i "s|pennylane.git@v${RELEASE_VERSION}-rc0|pennylane.git@master|g" $file
        git add $file
    done
    popd
    git commit -m "Target PennyLane master in requirements-[dev|tests].txt."

    sed -i "/__version__/d" $PL_VERSION_FILE
    if [ "$IS_TEST" == "true" ]; then
        echo '__version__ = "'${NEXT_VERSION}'-alpha1-dev0"' >> $PL_VERSION_FILE
    else
        echo '__version__ = "'${NEXT_VERSION}'-dev0"' >> $PL_VERSION_FILE
    fi
    git add $PL_VERSION_FILE
    git commit -m "Bump version to ${NEXT_VERSION}-dev0"

    sed -i 's/set(CATALYST_GIT_TAG *"[^"]*" *CACHE STRING "GIT_TAG value to build Catalyst")/set(CATALYST_GIT_TAG "main" CACHE STRING "GIT_TAG value to build Catalyst")/' ${ROOT_DIR}/cmake/support_catalyst.cmake
    git add ${ROOT_DIR}/cmake/support_catalyst.cmake
    git commit -m "Restore Catalyst GIT_TAG to main"

    for i in release stable; do
        sed -i "s|v${STABLE_VERSION}|v${RELEASE_VERSION}|g" ${ROOT_DIR}/.github/workflows/compat-docker-${i}.yml
        git add ${ROOT_DIR}/.github/workflows/compat-docker-${i}.yml
    done
    git commit -m "Update Docker workflows for new release version"

    add_CHANGELOG_entry "Merge RC v${RELEASE_VERSION} rc to master" "0000"
    git add $CHANGELOG_FILE
    git commit -m "Add CHANGELOG entry for RC merge"

    if [ "$LOCAL_TEST" == "false" ]; then
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} "rc_merge")
    fi
}

create_merge_PR(){
    # Create a PR to merge the RC into master and bump the version with NEXT_VERSION-dev
    if [ "$LOCAL_TEST" == "false" ]; then
    git checkout $(branch_name ${RELEASE_VERSION} "rc_merge")

    gh pr create $(use_dry_run) \
    --title "Merge RC v${RELEASE_VERSION}_rc to v${NEXT_VERSION}-dev" \
    --body "v${RELEASE_VERSION} RC merge branch." \
    --head $(branch_name ${RELEASE_VERSION} "rc_merge") \
    --base master \
    --label 'ci:build_wheels','ci:use-multi-gpu-runner','ci:use-gpu-runner','urgent'
    fi
}


# --------------------------------------------------------------------------------------------------
# Script body
# --------------------------------------------------------------------------------------------------

POSITIONAL_ARGS=()

CREATE_RC="false"
LIGHTNING_TEST="false"
RELEASE_ACTION="false"
RELEASE_ASSETS="false"

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      help
      exit 0
      ;;
    -s|--stable_version)
      STABLE_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -r|--release_version)
      RELEASE_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--next_version)
      NEXT_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--test)
      IS_TEST="$2"
      shift # past argument
      shift # past value
      ;;
    --create_rc)
      CREATE_RC="true"
      shift # past argument
      ;;
    --lightning_test)
      LIGHTNING_TEST="true"
      shift # past argument
      ;;
    --release)
      RELEASE_ACTION="true"
      shift # past argument
      ;;
    --release_assets)
      RELEASE_ASSETS="true"
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      echo "Use the option --help' for more information."
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

# If not a test run
if [ "$IS_TEST" == "false" ]; then
    LOCAL_TEST=false
fi

echo "STABLE_VERSION: $STABLE_VERSION"
echo "RELEASE_VERSION: $RELEASE_VERSION"
echo "NEXT_VERSION: $NEXT_VERSION"
echo "IS_TEST: $IS_TEST"
echo "CREATE_RC: $CREATE_RC"
echo "LIGHTNING_TEST: $LIGHTNING_TEST"
echo "RELEASE_ACTION: $RELEASE_ACTION"
echo "RELEASE_ASSETS: $RELEASE_ASSETS"


if [ "$CREATE_RC" == "true" ]; then
    create_release_candidate_branch
    create_release_candidate_PR
    create_docs_review_PR
    create_docker_PR
    create_version_bump_PR
    git checkout master
fi

if [ "$LIGHTNING_TEST" == "true" ]; then
    test_install_lightning
    download_artifacts_gh
    test_wheels_for_unwanted_libraries
fi

if [ "$RELEASE_ACTION" == "true" ]; then
    create_release_branch
    create_GitHub_release
    create_merge_branch
    create_merge_PR
fi

if [ "$RELEASE_ASSETS" == "true" ]; then
    download_release_artifacts_gh
    create_sdist
    upload_release_assets_gh
fi
