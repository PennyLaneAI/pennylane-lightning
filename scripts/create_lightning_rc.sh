#!/usr/bin/env bash

# Create a release candidate branch for PennyLane-Lightning
# This script should be run from the pennylane-lightning repository root.

STABLE_VERSION=0.42.0
RELEASE_VERSION=0.43.0
NEW_VERSION=0.44.0


dry_run="--dry-run"

branch_name(){
    version=$1
    suffix=$2
    echo "test_v${version}_${suffix}_test"
}

rreplace(){
   grep -rl "$1" . | xargs sed -i "s|$1|$2|g"
}

# Branch for the release candidate
create_lightning_rc_branch() {
    # clean up and checkout master
    # git reset --hard
    # git checkout master
    # git pull origin master

    # create branches 
    for branch in base docs rc; do
        git checkout -b $(branch_name ${RELEASE_VERSION} ${branch})
        git push --set-upstream origin $(branch_name ${RELEASE_VERSION} ${branch})
    done
    git checkout $(branch_name ${RELEASE_VERSION} rc)

    # update lightning version
    sed -i "/${STABLE_VERSION}/d" pennylane_lightning/core/_version.py
    echo '__version__ = "'${RELEASE_VERSION}'-rc0"' >> pennylane_lightning/core/_version.py
    sed -i "s|Release ${RELEASE_VERSION}-dev (development release)|Release ${RELEASE_VERSION}|g" .github/CHANGELOG.md

    # commit & push lightning version
    git add pennylane_lightning/core/_version.py .github/CHANGELOG.md
    git commit -m "Create v${RELEASE_VERSION} RC branch."

    # update pennylane dep
    for file in requirements-dev.txt requirements-tests.txt; do
        sed -i "s|pennylane.git@master|pennylane.git@v${RELEASE_VERSION}-rc0|g" $file
        git add $file
    done
    git commit -m "Target PennyLane v${RELEASE_VERSION}-rc0 in requirements-tests.txt."

    # update catalyst dep
    last_catalyst_commit=$(git ls-remote git@github.com:PennyLaneAI/catalyst.git HEAD | cut -f 1)
    sed -i 's|CATALYST_GIT_TAG "main"|CATALYST_GIT_TAG "'${last_catalyst_commit}'"|g' cmake/support_catalyst.cmake
    git add cmake/support_catalyst.cmake
    git commit -m "Set Catalyst dependency in cmake to commit ${last_catalyst_commit}."

    # update rng salt
    sed -i "/rng_salt = /d" tests/pytest.ini
    echo "rng_salt = v${RELEASE_VERSION}" >> tests/pytest.ini
    git add tests/pytest.ini
    git commit -m "Set rng_salt to v${RELEASE_VERSION} in tests/pytest.ini."

    # Enable to upload the wheels to TestPyPI and GitHub Artifacs
    sed -i "s|event_name == 'release'|event_name == 'pull_request'|g" .github/workflows/wheel_*

    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} rc)
}

create_lightning_rc_PR(){

    # create PR
    git checkout $(branch_name ${RELEASE_VERSION} rc)
    gh pr create  $dry_run \
        --title "Create v${RELEASE_VERSION} RC branch" \
        --body "v${RELEASE_VERSION} RC branch." \
        --head $(branch_name ${RELEASE_VERSION} rc) \
        --base $(branch_name ${RELEASE_VERSION} base) \
        --label 'do not merge','ci:build_wheels','ci:use-multi-gpu-runner','ci:use-gpu-runner','urgent'
}

create_lightning_docs_PR(){

    # create PR
    git checkout $(branch_name ${RELEASE_VERSION} docs)
    gh pr create $dry_run \
        --title "Create v${RELEASE_VERSION} Doc branch" \
        --body "v${RELEASE_VERSION} Doc branch." \
        --head $(branch_name ${RELEASE_VERSION} docs) \
        --base $(branch_name ${RELEASE_VERSION} rc) \
        --draft \
        --label 'do not merge','documentation'
}

create_lightning_docker_PR(){

    # create PR
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

create_lightning_version_bump_PR(){

    # create PR
    git checkout master
    git checkout -b $(branch_name ${RELEASE_VERSION} bump)

    # Update CHANGELOG with new_changelog_entry
    {
        echo "$new_changelog_entry"
        echo ""
        cat .github/CHANGELOG.md
    } > temp_changelog.md && mv temp_changelog.md .github/CHANGELOG.md

    # commit & push version bump
    git add pennylane_lightning/core/_version.py
    git commit -m "Bump version to v${RELEASE_VERSION}."

    git push --set-upstream origin $(branch_name ${RELEASE_VERSION} bump)

    gh pr create $dry_run \
        --title "Bump version to v${RELEASE_VERSION}" \
        --body "Bump version to v${RELEASE_VERSION}." \
        --head $(branch_name ${RELEASE_VERSION} bump) \
        --base master \
        --label 'do not merge','urgent'
}

test_pennylane_version(){

    backends=(
        "lightning.qubit"
        "lightning.gpu"
        "lightning.kokkos"
        "lightning.tensor"
    )

    while IFS= read -r line; do
        backend=$(echo "$line" | awk '{print $2}')
        if [[ " ${backends[*]} " == *" $backend "* ]]; then
            # Check the version using the RELEASE_VERSION
            # echo "RELEASE_VERSION: $RELEASE_VERSION"
            # echo "LINE: $line"
            # echo "BACKEND: $backend"
            if [[ "$line" == *"$RELEASE_VERSION"* ]]; then
                echo "Correct version for backend: $backend"
            else
                echo "Wrong version for backend: $backend"
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
        echo "Missing backends: ${backends[@]}"
    fi
}

test_install_lightning(){

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

download_artifacs_gh(){
    
}

# Main script

# create_lightning_rc_branch
# create_lightning_rc_PR
# create_lightning_docs_PR
# create_lightning_docker_PR
# create_lightning_version_bump_PR

test_install_lightning
# test_pennylane_version

# # upload wheel artifacts
# pushd .github/workflows
# rreplace "          github.event_name == 'release'" "          github.event_name == 'pull_request'"
# popd
# git add -u .github/workflows
# git commit -m "Change trigger to upload wheel artifacts."
# git push

# # update compate workflows
# cat > cron<<EOF
#   schedule:
#     - cron: "0 23 * * 0-6"  # Run daily at 11pm everyday
# EOF
# sed -i '/  workflow_dispatch:/ r cron' .github/workflows/compat-check-release-release.yml
# sed -i '/  workflow_dispatch:/ r cron' .github/workflows/compat-docker-release.yml
# sed -i "s|$STABLE_VERSION|$RELEASE_VERSION|g" .github/workflows/compat-docker-release.yml
# git add -u .github/workflows
# git commit -m "Set compat workflow and docker builds to run every night at 11pm."
# git push

# # create PR
# git checkout master
# git checkout -b v${RELEASE_VERSION}_base
# git push --set-upstream origin v${RELEASE_VERSION}_base

# git checkout v${RELEASE_VERSION}_rc
# gh pr create --dry-run --title "Create v${RELEASE_VERSION} RC branch." --head v${RELEASE_VERSION}_rc --body "v${RELEASE_VERSION} RC branch." --base v${RELEASE_VERSION}_base --label 'do not merge' --label 'ci:build_wheels' --label 'ci:use-multi-gpu-runner'
# gh pr create --title "Create v${RELEASE_VERSION} RC branch." --head v${RELEASE_VERSION}_rc --body "v${RELEASE_VERSION} RC branch." --base v${RELEASE_VERSION}_base --label 'do not merge' --label 'ci:build_wheels' --label 'ci:use-multi-gpu-runner'

# # Proof-read CHANGELOG & README

# cat >install.sh<<EOF
# for backend in qubit gpu kokkos tensor; do
#     make python backend=lightning_\$backend
# done
# python -c "import pennylane as qml; qml.about(); exit()"
# CMAKE_ARGS="-DKokkos_ENABLE_CUDA=ON" make python backend=lightning_kokkos
# python -c "import pennylane as qml; qml.about(); exit()"
# EOF
# bash -e install.sh

# cat > validate_attrs.py<<EOF
# import pennylane as qml
# from pennylane_lightning.lightning_gpu import LightningGPU as plg
# from pennylane_lightning.lightning_kokkos import LightningKokkos as plk

# for lclass, name in zip([plg, plk], ["lightning.gpu", "lightning.kokkos"]):
#     dev = qml.device(name, wires=0)

#     print(dev.version)
#     print(lclass.version)
#     print(dev.pennylane_requires)
# EOF
# python validate_attrs.py

# # Create release

# git checkout -b v${RELEASE_VERSION}_release
# sed -i "/$RELEASE_VERSION/d" pennylane_lightning/core/_version.py
# echo '__version__ = "'${RELEASE_VERSION}'"' >> pennylane_lightning/core/_version.py
# pushd .github/workflows
# sed -i "s|event_name == 'pull_request'|event_name == 'release'|g" wheel_*
# popd
# git add -u .github/workflows pennylane_lightning
# git commit -m "Forked as v${RELEASE_VERSION}_release to be released with tag v${RELEASE_VERSION}."
# git push --set-upstream origin v${RELEASE_VERSION}_release

# # Create source dists

# for backend in qubit gpu kokkos tensor; do
#     PL_BACKEND=lightning_${backend} python scripts/configure_pyproject_toml.py
#     PL_BACKEND=lightning_${backend} python setup.py sdist
# done
# # upload ./dist/* to release

