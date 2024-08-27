#!/usr/bin/env bash

OLDVER=0.37.0
LVER=0.38.0
rreplace(){
   grep -rl "$1" . | xargs sed -i "s|$1|$2|g"
}

# clean up and checkout master
git reset .
git checkout .
git fetch
git checkout master
git pull origin master

# create local branch
git branch -d v${LVER}_rc
git checkout -b v${LVER}_rc

# update lightning version
sed -i "/$LVER/d" pennylane_lightning/core/_version.py
echo '__version__ = "'${LVER}'-rc0"' >> pennylane_lightning/core/_version.py
sed -i "s|Release ${LVER}-dev|Release ${LVER}|g" .github/CHANGELOG.md

# commit & push lightning version
git add -u pennylane_lightning/core/_version.py .github/CHANGELOG.md
git commit -m "Create v${LVER} RC branch."
git push --set-upstream origin v${LVER}_rc

# update pennylane dep
for file in requirements-dev.txt requirements-tests.txt; do
    sed -i "s|pennylane.git@master|pennylane.git@v${LVER}-rc0|g" $file
    git add -u $file
done
git commit -m "Target PennyLane v${LVER}-rc0 in requirements-tests.txt."
git push

# upload wheel artifacts
pushd .github/workflows
rreplace "          github.event_name == 'release'" "          github.event_name == 'pull_request'"
popd
git add -u .github/workflows
git commit -m "Change trigger to upload wheel artifacts."
git push

# update compate workflows
cat > cron<<EOF
  schedule:
    - cron: "0 23 * * 0-6"  # Run daily at 11pm everyday
EOF
sed -i '/  workflow_dispatch:/ r cron' .github/workflows/compat-check-release-release.yml
sed -i '/  workflow_dispatch:/ r cron' .github/workflows/compat-docker-release.yml
sed -i "s|$OLDVER|$LVER|g" .github/workflows/compat-docker-release.yml
git add -u .github/workflows
git commit -m "Set compat workflow and docker builds to run every night at 11pm."
git push

# create PR
git checkout master
git checkout -b v${LVER}_base
git push --set-upstream origin v${LVER}_base

git checkout v${LVER}_rc
gh pr create --dry-run --title "Create v${LVER} RC branch." --head v${LVER}_rc --body "v${LVER} RC branch." --base v${LVER}_base --label 'do not merge' --label 'ci:build_wheels' --label 'ci:use-multi-gpu-runner'
gh pr create --title "Create v${LVER} RC branch." --head v${LVER}_rc --body "v${LVER} RC branch." --base v${LVER}_base --label 'do not merge' --label 'ci:build_wheels' --label 'ci:use-multi-gpu-runner'

# Proof-read CHANGELOG & README

cat >install.sh<<EOF
for backend in qubit gpu kokkos tensor; do
    make python backend=lightning_\$backend
done
python -c "import pennylane as qml; qml.about(); exit()"
CMAKE_ARGS="-DKokkos_ENABLE_CUDA=ON" make python backend=lightning_kokkos
python -c "import pennylane as qml; qml.about(); exit()"
EOF
bash -e install.sh

cat > validate_attrs.py<<EOF
import pennylane as qml
from pennylane_lightning.lightning_gpu import LightningGPU as plg
from pennylane_lightning.lightning_kokkos import LightningKokkos as plk

for lclass, name in zip([plg, plk], ["lightning.gpu", "lightning.kokkos"]):
    dev = qml.device(name, wires=0)

    print(dev.version)
    print(lclass.version)
    print(dev.pennylane_requires)
EOF
python validate_attrs.py
