#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# This script creates a release candidate branch for PennyLane-Lightning

# Set version numbers
STABLE_VERSION=0.42.0
RELEASE_VERSION=0.43.0
NEW_VERSION=0.44.0

IS_TEST=true

LOCAL_TEST=false

PUSH_TESTPYPI=false

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "gh CLI could not be found"
    exit
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
    echo "  -n, --new_version [version]       Specify the new version. Default $NEW_VERSION"
    echo "  -t, --test                        Run on test version, gh pr create with --dry-run. Default $IS_TEST"
    echo "  --create_rc                       Create a release candidate"
    echo "  --lightning_test                  Run Lightning tests"
    echo "  --release                         Perform release actions on GitHub"
    echo "  --release_assets                  Handle release assets to upload"
    echo "  -h, --help                        Show this help message"
}


# Utils functions
use_dry_run(){
    # Check if the script is running for testing. If so, use the --dry-run flag.
    dry_run="--draft"

    if [ "$LOCAL_TEST" == "true" ]; then
        dry_run="--dry-run"
    fi
    
    echo $dry_run
}

branch_name(){
    version=$1
    suffix=$2

    branch=$(echo "v${version}_${suffix}" | tr '[:upper:]' '[:lower:]')

    if [ "$IS_TEST" == "true" ]; then
        branch="v${version}_${suffix}_alpha"
    fi

    # Warning: delete the following line before merging
    branch="test_v${version}_${suffix}_alpha"

    echo $branch
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
        echo "⚠️  Missing backends: ${backends[@]}"
    fi
}

create_release_notes(){
    # Create the release notes for the release
    CHANGELOG_FILE=".github/CHANGELOG.md"
    changelog_lower_bound=$(grep -n -- "---" "${CHANGELOG_FILE}" | head -n 1 | cut -d: -f1)
    sed -n "1,${changelog_lower_bound}p" $CHANGELOG_FILE | sed ':a;N;$!ba;s/\.\n *\[(#/\. \[(#/g' > release_notes.md
}

# Release functions

create_release_candidate_branch() {
    # Create a new branch for the release candidate

    echo "Warning: uncomment the following lines before merge master"
    # clean up and checkout master
    # git reset --hard
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
    sed -i "/__version__/d" pennylane_lightning/core/_version.py
    if [ "$IS_TEST" == "true" ]; then
        echo '__version__ = "'${RELEASE_VERSION}'-rc0-dev99"' >> pennylane_lightning/core/_version.py
    else
        echo '__version__ = "'${RELEASE_VERSION}'-rc0"' >> pennylane_lightning/core/_version.py
    fi
    sed -i "s|Release ${RELEASE_VERSION}-dev (development release)|Release ${RELEASE_VERSION}|g" .github/CHANGELOG.md

    git add pennylane_lightning/core/_version.py .github/CHANGELOG.md
    git commit -m "Create v${RELEASE_VERSION} RC branch."

    # Update PennyLane dependency
    for file in requirements-dev.txt requirements-tests.txt; do
        sed -i "s|pennylane.git@master|pennylane.git@v${RELEASE_VERSION}-rc0|g" $file
        git add $file
    done
    git commit -m "Target PennyLane v${RELEASE_VERSION}-rc0 in requirements-[dev|tests].txt."

    # Update Catalyst dependency
    last_catalyst_commit=$(git ls-remote git@github.com:PennyLaneAI/catalyst.git HEAD | cut -f 1)
    sed -i 's|CATALYST_GIT_TAG "main"|CATALYST_GIT_TAG "'${last_catalyst_commit}'"|g' cmake/support_catalyst.cmake
    git add cmake/support_catalyst.cmake
    git commit -m "Set Catalyst dependency in cmake to commit ${last_catalyst_commit}."

    # Update RNG salt
    sed -i "/rng_salt = /d" tests/pytest.ini
    echo "rng_salt = v${RELEASE_VERSION}" >> tests/pytest.ini
    git add tests/pytest.ini
    git commit -m "Set rng_salt to v${RELEASE_VERSION} in tests/pytest.ini."

    # Enable to upload the wheels to TestPyPI and GitHub Artifacts
    if [ "$PUSH_TESTPYPI" == "true" ]; then
    sed -i "s|event_name == 'release'|event_name == 'pull_request'|g" .github/workflows/wheel_*
    git add .github/workflows/wheel_*
    git commit -m "Update wheel workflows for pull request"
    fi

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

    sed -i "s|v${STABLE_VERSION}|v${RELEASE_VERSION}|g" .github/workflows/compat-docker-release.yml

    git add .github/workflows/compat-docker-release.yml
    git commit -m "Update compat-docker-release.yml to use v${RELEASE_VERSION}"

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

new_changelog_entry=$(
cat <<EOF
# Release ${NEW_VERSION}-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Internal changes ⚙️</h3>

- Bumped the version.
    [(#xyz)](https://github.com/PennyLaneAI/pennylane-lightning/pull/xyz)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

{Release Manager Name}

---

EOF
)

create_version_bump_PR(){
    # Create a PR for the new version 

    git checkout master
    git checkout -b $(branch_name ${RELEASE_VERSION} bump)

    # Update CHANGELOG with new_changelog_entry
    {
        echo "$new_changelog_entry"
        echo ""
        cat .github/CHANGELOG.md
    } > temp_changelog.md && mv temp_changelog.md .github/CHANGELOG.md

    sed -i "s|Release ${RELEASE_VERSION}-dev (development release)|Release ${RELEASE_VERSION}|g" .github/CHANGELOG.md

    git add .github/CHANGELOG.md
    git commit -m "Update CHANGELOG.md with new version entry."

    # Update lightning version
    sed -i "/__version__/d" pennylane_lightning/core/_version.py
    if [ "$IS_TEST" == "true" ]; then
        echo '__version__ = "'${NEW_VERSION}'-alpha1-dev0"' >> pennylane_lightning/core/_version.py
    else
        echo '__version__ = "'${NEW_VERSION}'-dev0"' >> pennylane_lightning/core/_version.py
    fi

    git add pennylane_lightning/core/_version.py
    git commit -m "Bump version to v${NEW_VERSION}."

    if [ "$LOCAL_TEST" == "false" ]; then
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} bump)

    gh pr create $(use_dry_run) \
        --title "Bump version to v${NEW_VERSION}-dev" \
        --body "Bump version to v${NEW_VERSION}-dev." \
        --head $(branch_name ${RELEASE_VERSION} bump) \
        --base master \
        --label 'urgent'
    fi
}

test_install_lightning(){
    # Test Lightning installation 

    git checkout master
    git checkout $(branch_name ${RELEASE_VERSION} rc)

    # Test installation of lightning default backends
    pip install -r requirements-dev.txt
    for backend in qubit gpu kokkos tensor; do
        PL_BACKEND=lightning_${backend} python scripts/configure_pyproject_toml.py
        PL_BACKEND=lightning_${backend} python -m pip install . -v
    done

    # Test import
    is_installed_backend=$(test_pennylane_version)

    # Test installation of lightning custom compiles options

    # Lightning Kokkos with CUDA and MPI
    pip uninstall -y pennylane_lightning_kokkos
    PL_BACKEND="lightning_kokkos" python scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DENABLE_MPI=ON -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_OPENMP=ON" python -m pip install . -v

    # Test import
    is_installed_kokkos_mpi=$(test_pennylane_version | grep lightning.kokkos)

    # Lightning GPU with MPI
    pip uninstall -y pennylane_lightning_gpu
    PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py
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
        --json status,workflowName,workflowDatabaseId | jq '.[] | select(.workflowName | contains("Wheel"))')

    completed_runners=$(echo "$wheels_runners" | jq -r '. | select(.status == "completed") | .workflowDatabaseId')

    incomplete_runners=$(echo "$wheels_runners" | jq -r '. | select(.status != "completed") ')

    mkdir -p Wheels

    for runner in $completed_runners; do
        echo "Downloading artifacts for runner: $runner"
        gh run download --dir Wheels $runner
    done

    echo "Incomplete runner found:"
    echo "$incomplete_runners" | jq .
}

test_wheels_for_unwanted_libraries(){
    # Test for unwanted libraries in the wheels

    cd Wheels

    for wheel in *.zip; do
        unzip -o -q "$wheel"
    done

    python ../scripts/validate_attrs.py

    cd ..
}

create_release_branch(){
    # Create the release branch

    git checkout $(branch_name ${RELEASE_VERSION} rc)

    gh pr comment $(branch_name ${RELEASE_VERSION} rc) \
        --body "Forked as v${RELEASE_VERSION}_release to be released with tag v${RELEASE_VERSION}"

    # Create the release branch
    git checkout -b $(branch_name ${RELEASE_VERSION} release)

    # Update version
    sed -i "/__version__/d" pennylane_lightning/core/_version.py
    if [ "$IS_TEST" == "true" ]; then
        echo '__version__ = "'${RELEASE_VERSION}'-alpha"' >> pennylane_lightning/core/_version.py
    else
        echo '__version__ = "'${RELEASE_VERSION}'"' >> pennylane_lightning/core/_version.py
    fi

    
    if [ "$PUSH_TESTPYPI" == "true" ]; then
    # Disable to upload the wheels to TestPyPI and GitHub Artifacts
    sed -i "s|event_name == 'pull_request'|event_name == 'release'|g" .github/workflows/wheel_*
    fi
    
    git add pennylane_lightning/core/_version.py
    git add .github/workflows/wheel_*
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
        --notes-file release_notes.md \
        --draft --latest
    fi
}

download_release_artifacts_gh(){
    # Download the artifacts from the GitHub Actions runs
    wheels_runners=$(gh run list --event release --branch $(branch_name ${RELEASE_VERSION}) \
        --json status,workflowName,workflowDatabaseId | jq '.[] | select(.workflowName | contains("Wheel"))')

     completed_runners=$(echo "$wheels_runners" | jq -r '. | select(.status == "completed") | .workflowDatabaseId')
     incomplete_runners=$(echo "$wheels_runners" | jq -r '. | select(.status != "completed") ')

     mkdir -p Release_Assets

     for runner in $completed_runners; do
         echo "Downloading artifacts for runner: $runner"
         gh run download --dir Release_Assets $runner
     done

     echo "Incomplete runner found:"
     echo "$incomplete_runners" | jq .
}

create_sdist(){
    # Create the source distribution

    git checkout $(branch_name ${RELEASE_VERSION} "release")
    PL_BACKEND=lightning_qubit python ./scripts/configure_pyproject_toml.py 
    python setup.py sdist

    mkdir -p Release_Assets
    cp dist/*.tar.gz Release_Assets/
}

upload_release_assets_gh(){
    # Upload the release assets
    gh release upload $(branch_name ${RELEASE_VERSION} "0") Release_Assets/*.whl --clobber
    gh release upload $(branch_name ${RELEASE_VERSION} "0") Release_Assets/*.tar.gz --clobber
}

create_merge_branch(){
    # Create the merge branch to merge the RC into master and bump the version with NEW_VERSION-dev

    git checkout $(branch_name ${RELEASE_VERSION} "release")
    git checkout -b $(branch_name ${RELEASE_VERSION} "rc_merge")

    for file in requirements-dev.txt requirements-tests.txt; do
        sed -i "s|pennylane.git@v${RELEASE_VERSION}-rc0|pennylane.git@master|g" $file
        git add $file
    done
    git commit -m "Target PennyLane master in requirements-[dev|tests].txt."

    sed -i "/__version__/d" pennylane_lightning/core/_version.py
    if [ "$IS_TEST" == "true" ]; then
        echo '__version__ = "'${NEW_VERSION}'-alpha1-dev0"' >> pennylane_lightning/core/_version.py
    else
        echo '__version__ = "'${NEW_VERSION}'-dev0"' >> pennylane_lightning/core/_version.py
    fi
    git add pennylane_lightning/core/_version.py
    git commit -m "Bump version to ${NEW_VERSION}-dev0"

    sed -i 's/set(CATALYST_GIT_TAG *"[^"]*" *CACHE STRING "GIT_TAG value to build Catalyst")/set(CATALYST_GIT_TAG "main" CACHE STRING "GIT_TAG value to build Catalyst")/' cmake/support_catalyst.cmake
    git add cmake/support_catalyst.cmake
    git commit -m "Restore Catalyst GIT_TAG to main"

    for i in release stable; do
        sed -i "s|v${RELEASE_VERSION}|v${NEW_VERSION}|g" .github/workflows/compat-docker-${i}.yml
        git add .github/workflows/compat-docker-${i}.yml
    done
    git commit -m "Update Docker workflows for new release version"

    if [ "$LOCAL_TEST" == "false" ]; then
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} "rc_merge")
    fi
}

create_merge_PR(){
    # Create a PR to merge the RC into master and bump the version with NEW_VERSION-dev
    git checkout $(branch_name ${RELEASE_VERSION} "rc_merge")

    gh pr create $(use_dry_run) \
    --title "Merge RC v${RELEASE_VERSION}_rc to v${NEW_VERSION}-dev" \
    --body "v${RELEASE_VERSION} RC merge branch." \
    --head $(branch_name ${RELEASE_VERSION} "rc_merge") \
    --base master \
    --label 'ci:build_wheels','ci:use-multi-gpu-runner','ci:use-gpu-runner','urgent'
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
    -n|--new_version)
      NEW_VERSION="$2"
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


echo "STABLE_VERSION: $STABLE_VERSION"
echo "RELEASE_VERSION: $RELEASE_VERSION"
echo "NEW_VERSION: $NEW_VERSION"
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
    # test_install_lightning
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
