#!/usr/bin/env bash

# This script creates a release candidate branch for PennyLane-Lightning

# Set version numbers
STABLE_VERSION=0.42.0
RELEASE_VERSION=0.42.0
NEW_VERSION=0.44.0

IS_TEST=true

# Check if gh CLI is installed
if ! command -v gh &> /dev/null
then
    echo "gh CLI could not be found"
    exit
fi

# Utils functions
use_dry_run(){
    # Check if the script is running for testing. If so, use the --dry-run flag.
    dry_run=""

    if [ "$IS_TEST" == "true" ]; then
        dry_run="--dry-run"
    fi
    
    echo $dry_run
}

branch_name(){
    version=$1
    suffix=$2

    branch=$(echo "v${version}_${suffix}" | tr '[:upper:]' '[:lower:]')

    if [ "$IS_TEST" == "true" ]; then
         branch="test_v${version}_${suffix}_test"
    fi
    
    echo $branch
}

rreplace(){
   grep -rl "$1" . | xargs sed -i "s|$1|$2|g"
}

create_release_candidate_branch() {
    # Create a new branch for the release candidate

    # clean up and checkout master
    # git reset --hard
    # git checkout master
    # git pull origin master

    # Create branches 
    for branch in base docs rc; do
        git checkout -b $(branch_name ${RELEASE_VERSION} ${branch})
        git push --set-upstream origin $(branch_name ${RELEASE_VERSION} ${branch})
    done
    git checkout $(branch_name ${RELEASE_VERSION} rc)

    # Update lightning version
    sed -i "/${STABLE_VERSION}/d" pennylane_lightning/core/_version.py
    echo '__version__ = "'${RELEASE_VERSION}'-rc0"' >> pennylane_lightning/core/_version.py
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
    sed -i "s|event_name == 'release'|event_name == 'pull_request'|g" .github/workflows/wheel_*

    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} rc)
}

create_release_candidate_PR(){
    # Create a PR for the release candidate branch

    git checkout $(branch_name ${RELEASE_VERSION} rc)
    gh pr create $(use_dry_run) \
        --title "Create v${RELEASE_VERSION} RC branch" \
        --body "v${RELEASE_VERSION} RC branch." \
        --head $(branch_name ${RELEASE_VERSION} rc) \
        --base $(branch_name ${RELEASE_VERSION} base) \
        --label 'do not merge','ci:build_wheels','ci:use-multi-gpu-runner','ci:use-gpu-runner','urgent'
}

create_docs_review_PR(){
    # Create a PR for the docs review

    git checkout $(branch_name ${RELEASE_VERSION} docs)
    gh pr create $(use_dry_run) \
        --title "Create v${RELEASE_VERSION} Doc branch" \
        --body "v${RELEASE_VERSION} Doc branch." \
        --head $(branch_name ${RELEASE_VERSION} docs) \
        --base $(branch_name ${RELEASE_VERSION} rc) \
        --draft \
        --label 'do not merge','documentation'
}

create_docker_PR(){
    # Create a PR for the Docker test in PTM

    git checkout master
    git checkout -b $(branch_name ${RELEASE_VERSION} docker)

    rreplace "v${STABLE_VERSION}" "v${RELEASE_VERSION}" .github/workflows/compat-docker-release.yml

    git add .github/workflows/compat-docker-release.yml
    git commit -m "Update compat-docker-release.yml to use v${RELEASE_VERSION}"
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} docker)

    gh pr create $dry_run \
        --title "Docker test for v${RELEASE_VERSION} RC branch" \
        --body "Docker test for v${RELEASE_VERSION} RC branch." \
        --head $(branch_name ${RELEASE_VERSION} docker) \
        --base master \
        --label 'urgent'
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

    # Update lightning version
    sed -i "/${RELEASE_VERSION}/d" pennylane_lightning/core/_version.py
    echo '__version__ = "'${NEW_VERSION}'-dev0"' >> pennylane_lightning/core/_version.py

    git add pennylane_lightning/core/_version.py
    git commit -m "Bump version to v${NEW_VERSION}."

    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} bump)

    gh pr create $dry_run \
        --title "Bump version to v${NEW_VERSION}-dev" \
        --body "Bump version to v${NEW_VERSION}-dev." \
        --head $(branch_name ${RELEASE_VERSION} bump) \
        --base master \
        --label 'urgent'
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

test_install_lightning(){
    # Test Lightning installation 

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
    CMAKE_ARGS="-DENABLE_MPI=ON -DKokkos_ENABLE_OPENMP=ON" python -m pip install . -v

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

    wheels_runners=$(gh run list --branch $(branch_name ${RELEASE_VERSION} rc) --json status,workflowName,workflowDatabaseId | jq '.[] | select(.workflowName | contains("Wheel"))')

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
    sed -i "/$RELEASE_VERSION/d" pennylane_lightning/core/_version.py
    echo '__version__ = "'${RELEASE_VERSION}'"' >> pennylane_lightning/core/_version.py

    # Disable to upload the wheels to TestPyPI and GitHub Artifacts
    sed -i "s|event_name == 'pull_request'|event_name == 'release'|g" .github/workflows/wheel_*

    git add -u .
    git commit -m "Pre-release updates"
    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} release)
}

create_GitHub_release(){
    # Create the GitHub release as draft
    git checkout $(branch_name ${RELEASE_VERSION} release)

    create_release_notes

    # Create tag
    git tag -a "$(branch_name ${RELEASE_VERSION})" -m "Release ${RELEASE_VERSION}"
    git push origin "$(branch_name ${RELEASE_VERSION})"

    gh release create $(branch_name ${RELEASE_VERSION}) \
        --target $(branch_name ${RELEASE_VERSION} release) \
        --title "Release ${RELEASE_VERSION}" \
        --notes-file release_notes.md \
        --draft --latest
}

create_changelog_for_release(){
    # Create the release notes for the release
    CHANGELOG_FILE=".github/CHANGELOG.md"
    changelog_lower_bound=$(grep -n -- "---" "${CHANGELOG_FILE}" | head -n 1 | cut -d: -f1)
    sed -n "1,${changelog_lower_bound}p" $CHANGELOG_FILE | sed ':a;N;$!ba;s/\.\n *\[(#/\. \[(#/g' > release_notes.md
}

download_release_artifacts_gh(){
    # Download the artifacts from the GitHub Actions runs
    wheels_runners=$(gh run list --event release --branch $(branch_name ${RELEASE_VERSION}) --json status,workflowName,workflowDatabaseId | jq '.[] | select(.workflowName | contains("Wheel"))')

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

    sed -i "/${RELEASE_VERSION}/d" pennylane_lightning/core/_version.py
    echo '__version__ = "'${RELEASE_VERSION}'-rc0"' >> pennylane_lightning/core/_version.py
    git add pennylane_lightning/core/_version.py
    git commit -m "Bump version to ${RELEASE_VERSION}-rc0"

    sed -i 's/set(CATALYST_GIT_TAG *"[^"]*" *CACHE STRING "GIT_TAG value to build Catalyst")/set(CATALYST_GIT_TAG "main" CACHE STRING "GIT_TAG value to build Catalyst")/' cmake/support_catalyst.cmake
    git add cmake/support_catalyst.cmake
    git commit -m "Restore Catalyst GIT_TAG to main"

    for i in release stable; do
        rreplace "v${RELEASE_VERSION}" "v${NEW_VERSION}" .github/workflows/compat-docker-${i}.yml
        git add .github/workflows/compat-docker-${i}.yml
    done
    git commit -m "Update Docker workflows for new release version"

   git push --set-upstream origin $(branch_name ${RELEASE_VERSION} "rc_merge")
}

create_merge_PR(){
    git checkout $(branch_name ${RELEASE_VERSION} "rc_merge")

    gh pr create $dry_run \
    --title "Merge RC v${RELEASE_VERSION}_rc to v${NEW_VERSION}-dev" \
    --body "v${RELEASE_VERSION} RC merge branch." \
    --head $(branch_name ${RELEASE_VERSION} "rc_merge") \
    --base master \
    --label 'ci:build_wheels','ci:use-multi-gpu-runner','ci:use-gpu-runner','urgent'
}



